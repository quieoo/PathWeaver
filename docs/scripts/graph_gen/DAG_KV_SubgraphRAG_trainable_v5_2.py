import argparse
import gc
import json
import os
import shutil
import random
import re
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau
from numpy.lib.format import open_memmap


def log(msg: str) -> None:
    print(msg, flush=True)


# ============================================================
# 0) IO
# ============================================================
def _read_concatenated_json_objects(path: str) -> List[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    data: List[Dict[str, Any]] = []
    buf = ''

    with open(path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            while True:
                s = buf.lstrip()
                if not s:
                    buf = ''
                    break
                try:
                    obj, idx = decoder.raw_decode(s)
                except json.JSONDecodeError:
                    break
                if not isinstance(obj, dict):
                    raise ValueError(f'Expected JSON object in {path}, got {type(obj).__name__}')
                data.append(obj)
                buf = s[idx:]

    s = buf.lstrip()
    while s:
        obj, idx = decoder.raw_decode(s)
        if not isinstance(obj, dict):
            raise ValueError(f'Expected JSON object in {path}, got {type(obj).__name__}')
        data.append(obj)
        s = s[idx:].lstrip()
    return data


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith('.jsonl'):
        try:
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data
        except json.JSONDecodeError:
            return _read_concatenated_json_objects(path)

    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and 'data' in obj and isinstance(obj['data'], list):
        return obj['data']
    raise ValueError('Unsupported JSON root format. Expect list or {data:[...]}.')


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# 1) Normalization
# ============================================================
_SPACE_RE = re.compile(r'\s+')
_ZW_RE = re.compile(r'[\u200b\u200c\u200d\uFEFF]')
_PAREN_CONTENT_RE = re.compile(r'\([^)]*\)')
_PUNCT_RE = re.compile(r'[^a-z0-9\s]+')
_STOPWORDS = {
    'the', 'a', 'an', 'of', 'in', 'on', 'at', 'for', 'to', 'from', 'by', 'with', 'is', 'was',
    'were', 'are', 'be', 'been', 'being', 'and', 'or', 'that', 'this', 'these', 'those', 'what',
    'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how', 'as', 'into', 'about'
}


def _fix_unbalanced_parentheses(s: str) -> str:
    if not s:
        return s
    l = s.count('(')
    r = s.count(')')
    if l > r:
        s = s + (')' * (l - r))
    elif r > l:
        extra = r - l
        while extra > 0 and s.endswith(')'):
            s = s[:-1]
            extra -= 1
    return s


def norm_text(x: Any) -> str:
    if x is None:
        return ''
    s = str(x)
    s = _ZW_RE.sub('', s)
    s = s.strip()
    s = _SPACE_RE.sub(' ', s)
    s = _fix_unbalanced_parentheses(s)
    return s


def norm_match(x: Any) -> str:
    return norm_text(x).lower()


def normalize_lex(s: str) -> str:
    s = (s or '').lower()
    s = _PUNCT_RE.sub(' ', s)
    toks = [t for t in s.split() if t and t not in _STOPWORDS]
    return ' '.join(toks)


def token_set(s: str) -> Set[str]:
    s = normalize_lex(s)
    return set(s.split()) if s else set()


def jaccard(a: str, b: str) -> float:
    A = token_set(a)
    B = token_set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A | B))


def entity_key(name: str) -> str:
    s = norm_text(name)
    s = _PAREN_CONTENT_RE.sub('', s)
    s = s.lower()
    s = _PUNCT_RE.sub(' ', s)
    s = _SPACE_RE.sub(' ', s).strip()
    return s


def entity_alias_keys(name: str) -> List[str]:
    k = entity_key(name)
    if not k:
        return [k]
    toks = k.split()
    keys = [k]
    if len(toks) >= 3:
        keys.append(' '.join(toks[:-1]))
    suffixes = {'jr', 'sr', 'ii', 'iii', 'iv'}
    if toks and toks[-1] in suffixes and len(toks) >= 2:
        keys.append(' '.join(toks[:-1]))
    out, seen = [], set()
    for kk in keys:
        if kk not in seen:
            out.append(kk)
            seen.add(kk)
    return out


def contains_mention(question: str, name: str) -> bool:
    q = normalize_lex(question)
    n = normalize_lex(name)
    if not n:
        return False
    if n in q:
        return True
    toks = n.split()
    if len(toks) >= 2 and ' '.join(toks[:2]) in q:
        return True
    if len(toks) == 1 and len(toks[0]) >= 5 and toks[0] in q:
        return True
    return False


def value_matches_answer(value: str, answer: str) -> bool:
    v = norm_match(value)
    a = norm_match(answer)
    if not v or not a:
        return False
    return v == a or a in v or v in a


# ============================================================
# 2) Embedding helpers
# ============================================================
def embed_texts(embedder: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    emb = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if isinstance(emb, list):
        emb = np.array(emb)
    return emb.astype(np.float32)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def cosine_sim_vec(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ============================================================
# 3) Data model
# ============================================================
@dataclass
class KVEdge:
    kid: int
    src: int
    dst: int
    src_name: str
    dst_name: str
    key: str
    value: str
    edge_score: float
    score: float
    title: str
    triple_type: str
    relation: str
    triple_s: str
    triple_o: str
    kv_idx: int


# ============================================================
# 4) Triple iterator / graph build
# ============================================================
def iter_triples(sample: Dict[str, Any], supporting_only: bool) -> Iterable[Tuple[str, Dict[str, Any]]]:
    ctx = sample.get('context', []) or []
    supporting_titles: Optional[Set[str]] = None
    yielded = False
    if supporting_only:
        sf = sample.get('supporting_facts', []) or []
        supporting_titles = {t for t, _ in sf if isinstance(t, str)}

    for para in ctx:
        if isinstance(para, dict):
            title = norm_text(para.get('title', ''))
            tri_list = para.get('triple_list', []) or []
        elif isinstance(para, list):
            title = norm_text(para[0]) if para else ''
            tri_list = para[2] if len(para) >= 3 else []

        if supporting_titles is not None and title not in supporting_titles:
            continue
        for tri in tri_list:
            yielded = True
            yield title, tri

    if yielded:
        return

    # Fallback for merged tripled datasets where triples live at the sample top level.
    top_level_triples = sample.get('triple_list', []) or []
    for tri in top_level_triples:
        title = norm_text(tri.get('title', '') if isinstance(tri, dict) else '')
        yield title, tri


def get_or_add_node(name: str, node_map: Dict[str, int], node_names: List[str]) -> int:
    nn = norm_text(name)
    for k in entity_alias_keys(nn):
        if k in node_map:
            return node_map[k]
    nid = len(node_names)
    node_names.append(nn)
    for k in entity_alias_keys(nn):
        if k:
            node_map[k] = nid
    return nid


def infer_kv_direction(s: str, o: str, key: str, value: str) -> str:
    sN = normalize_lex(s)
    oN = normalize_lex(o)
    kN = normalize_lex(key)
    vN = normalize_lex(value)
    if vN == oN:
        return 'forward'
    if vN == sN:
        return 'backward'
    s_in = sN and (sN in kN)
    o_in = oN and (oN in kN)
    if s_in and not o_in:
        return 'forward'
    if o_in and not s_in:
        return 'backward'
    return 'forward'


def build_kvedge_graph(sample: Dict[str, Any], supporting_only: bool) -> Tuple[List[str], Dict[int, KVEdge], Dict[int, List[int]], Dict[int, List[int]]]:
    node_map: Dict[str, int] = {}
    node_names: List[str] = []
    kvedges: Dict[int, KVEdge] = {}
    out_adj: Dict[int, List[int]] = defaultdict(list)
    in_adj: Dict[int, List[int]] = defaultdict(list)
    seen: Set[Tuple[str, str]] = set()
    kid = 0

    for title, tri in iter_triples(sample, supporting_only=supporting_only):
        ttype = norm_text(tri.get('type', ''))
        rel = norm_text(tri.get('description_type', ''))
        s = norm_text(tri.get('name', ''))
        o = norm_text(tri.get('description', ''))
        if not s or not o:
            continue

        kvs = tri.get('kv_lists', []) or []
        for kv_idx, kv in enumerate(kvs):
            key = norm_text(kv.get('key_string', ''))
            value = norm_text(kv.get('value_string', ''))
            if not key or not value:
                continue
            sig = (key, value)
            if sig in seen:
                continue
            seen.add(sig)

            # direction = infer_kv_direction(s, o, key, value)
            # if direction == 'backward':
            #     src_name, dst_name = o, s
            # else:
            #     src_name, dst_name = s, o
            
            # 第一条forward，第二条backward, 依次类推
            if kv_idx % 2 == 0:
                src_name, dst_name = s, o
            else:
                src_name, dst_name = o, s


            src = get_or_add_node(src_name, node_map, node_names)
            dst = get_or_add_node(dst_name, node_map, node_names)

            e = KVEdge(
                kid=kid,
                src=src,
                dst=dst,
                src_name=node_names[src],
                dst_name=node_names[dst],
                key=key,
                value=value,
                edge_score=0.0,
                score=0.0,
                title=norm_text(title),
                triple_type=ttype,
                relation=rel,
                triple_s=s,
                triple_o=o,
                kv_idx=kv_idx,
            )
            kvedges[kid] = e
            out_adj[src].append(kid)
            in_adj[dst].append(kid)
            kid += 1

    return node_names, kvedges, dict(out_adj), dict(in_adj)



def print_kvedge_graph(
    kvedges: Dict[int, KVEdge],
    out_adj: Dict[int, List[int]],
    in_adj: Dict[int, List[int]],
) -> None:
    if not kvedges:
        print('[Graph] empty kvedge graph')
        return

    node_names: Dict[int, str] = {}
    for e in kvedges.values():
        node_names[e.src] = e.src_name
        node_names[e.dst] = e.dst_name

    print(f'[Graph] nodes={len(node_names)} edges={len(kvedges)}')
    print('[Graph] edge list:')
    for eid in sorted(kvedges.keys()):
        e = kvedges[eid]
        print(
            f'  eid={eid} src={e.src}("{e.src_name}") -> dst={e.dst}("{e.dst_name}") '
            f'key="{e.key}" value="{e.value}" edge_score={e.edge_score:.4f} score={e.score:.4f}'
        )

    print('[Graph] node adjacency:')
    for nid in sorted(node_names.keys()):
        out_eids = sorted(out_adj.get(nid, []))
        in_eids = sorted(in_adj.get(nid, []))
        print(f'  nid={nid} name="{node_names[nid]}"')
        print(f'    in_eids={in_eids}')
        for eid in in_eids:
            e = kvedges[eid]
            print(f'      <- eid={eid} from {e.src}("{e.src_name}") key="{e.key}" value="{e.value}"')
        print(f'    out_eids={out_eids}')
        for eid in out_eids:
            e = kvedges[eid]
            print(f'      -> eid={eid} to {e.dst}("{e.dst_name}") key="{e.key}" value="{e.value}"')


# ============================================================
# 5) Topic entities / DDE / features
# ============================================================
def identify_topic_entities(question: str, node_names: List[str], node_emb: np.ndarray, q_emb: np.ndarray, top_k: int, mention_bonus: float) -> List[int]:
    if len(node_names) == 0:
        return []
    sims = cosine_sim_matrix(q_emb[None, :], node_emb).reshape(-1)
    scores = sims.copy()
    for i, name in enumerate(node_names):
        if contains_mention(question, name):
            scores[i] += mention_bonus
        scores[i] += 0.15 * jaccard(question, name)
    order = np.argsort(-scores)
    return [int(x) for x in order[:max(1, min(top_k, len(order)))]]


def compute_dde(num_nodes: int, out_adj: Dict[int, List[int]], in_adj: Dict[int, List[int]], kvedges: Dict[int, KVEdge], topic_nodes: List[int], max_hops: int) -> np.ndarray:
    if num_nodes == 0:
        return np.zeros((0, 1 + 2 * max_hops), dtype=np.float32)
    s0 = np.zeros((num_nodes,), dtype=np.float32)
    for t in topic_nodes:
        if 0 <= t < num_nodes:
            s0[t] = 1.0
    feats = [s0]
    cur = s0.copy()
    for _ in range(max_hops):
        nxt = np.zeros_like(cur)
        for e in kvedges.values():
            nxt[e.dst] += cur[e.src]
        indeg = np.ones((num_nodes,), dtype=np.float32)
        for v, eids in in_adj.items():
            indeg[v] = max(1.0, float(len(eids)))
        nxt = nxt / indeg
        feats.append(nxt)
        cur = nxt
    cur = s0.copy()
    rev_feats = []
    for _ in range(max_hops):
        nxt = np.zeros_like(cur)
        for e in kvedges.values():
            nxt[e.src] += cur[e.dst]
        outdeg = np.ones((num_nodes,), dtype=np.float32)
        for u, eids in out_adj.items():
            outdeg[u] = max(1.0, float(len(eids)))
        nxt = nxt / outdeg
        rev_feats.append(nxt)
        cur = nxt
    feats.extend(rev_feats)
    return np.stack(feats, axis=1).astype(np.float32)


def weak_label_edge(sample: Dict[str, Any], e: KVEdge) -> int:
    answer = norm_text(sample.get('answer', ''))
    supporting_titles = {norm_text(t) for t, _ in (sample.get('supporting_facts', []) or []) if isinstance(t, str)}
    pos = False
    if answer and value_matches_answer(e.value, answer):
        pos = True
    return int(pos)

def embed_texts_cached(
    embedder: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """
    Encode unique texts once, return {normalized_text: embedding}.
    """
    uniq = []
    seen = set()
    for t in texts:
        t = norm_text(t)
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    if not uniq:
        return {}

    emb = embedder.encode(
        uniq,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if isinstance(emb, list):
        emb = np.array(emb)
    emb = emb.astype(np.float32)
    return {t: emb[i] for i, t in enumerate(uniq)}


def merge_text_embedding_caches(
    embedder: SentenceTransformer,
    text_groups: List[List[str]],
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """
    Encode multiple text groups incrementally while deduplicating across groups.
    Earlier groups take priority if the same normalized text appears again.
    """
    merged: Dict[str, np.ndarray] = {}
    seen: Set[str] = set()
    for texts in text_groups:
        uniq_texts: List[str] = []
        for t in texts:
            t = norm_text(t)
            if t and t not in seen:
                seen.add(t)
                uniq_texts.append(t)
        if not uniq_texts:
            continue
        merged.update(embed_texts_cached(embedder, uniq_texts, batch_size=batch_size))
    return merged


def build_edge_feature_dict(
    question: str,
    node_names: List[str],
    node_emb: Optional[np.ndarray],
    kvedges: Dict[int, KVEdge],
    out_adj: Dict[int, List[int]],
    in_adj: Dict[int, List[int]],
    embedder: Optional[SentenceTransformer],
    batch_size: int,
    topic_top_k: int,
    dde_hops: int,
    mention_bonus: float,
    text_emb_cache: Optional[Dict[str, np.ndarray]] = None,
    q_emb: Optional[np.ndarray] = None,
    rel_emb: Optional[np.ndarray] = None,
    key_emb: Optional[np.ndarray] = None,
    val_emb: Optional[np.ndarray] = None,
) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
    """
    Faster version:
      - use precomputed q/node/rel/key/val embeddings if provided
      - otherwise fall back to old on-the-fly embedding path
    """
    question = norm_text(question)

    if q_emb is None:
        if text_emb_cache is not None:
            q_emb = text_emb_cache[question]
        else:
            assert embedder is not None
            q_emb = embed_texts(embedder, [question], batch_size=batch_size)[0]

    if node_emb is None:
        if text_emb_cache is not None:
            node_emb = np.stack([text_emb_cache[norm_text(x)] for x in node_names]).astype(np.float32)
        else:
            assert embedder is not None
            node_emb = embed_texts(embedder, node_names, batch_size=batch_size)

    topic_nodes = identify_topic_entities(
        question, node_names, node_emb, q_emb, top_k=topic_top_k, mention_bonus=mention_bonus
    )
    dde = compute_dde(len(node_names), out_adj, in_adj, kvedges, topic_nodes, max_hops=dde_hops)

    eids = sorted(kvedges.keys())

    if rel_emb is None:
        rel_texts = [norm_text(kvedges[eid].relation or kvedges[eid].key) for eid in eids]
        if text_emb_cache is not None:
            rel_emb = np.stack([text_emb_cache[t] for t in rel_texts]).astype(np.float32)
        else:
            assert embedder is not None
            rel_emb = embed_texts(embedder, rel_texts, batch_size=batch_size)

    if key_emb is None:
        key_texts = [norm_text(kvedges[eid].key) for eid in eids]
        if text_emb_cache is not None:
            key_emb = np.stack([text_emb_cache[t] for t in key_texts]).astype(np.float32)
        else:
            assert embedder is not None
            key_emb = embed_texts(embedder, key_texts, batch_size=batch_size)

    if val_emb is None:
        val_texts = [norm_text(kvedges[eid].value) for eid in eids]
        if text_emb_cache is not None:
            val_emb = np.stack([text_emb_cache[t] for t in val_texts]).astype(np.float32)
        else:
            assert embedder is not None
            val_emb = embed_texts(embedder, val_texts, batch_size=batch_size)

    feats: Dict[int, Dict[str, Any]] = {}
    topic_set = set(topic_nodes)

    for i, eid in enumerate(eids):
        e = kvedges[eid]
        src_emb = node_emb[e.src]
        dst_emb = node_emb[e.dst]
        r_emb = rel_emb[i]
        k_emb = key_emb[i]
        v_emb = val_emb[i]
        src_dde = dde[e.src]
        dst_dde = dde[e.dst]

        scalar = np.array([
            cosine_sim_vec(q_emb, src_emb),
            cosine_sim_vec(q_emb, dst_emb),
            cosine_sim_vec(q_emb, r_emb),
            cosine_sim_vec(q_emb, k_emb),
            cosine_sim_vec(q_emb, v_emb),
            cosine_sim_vec(src_emb, dst_emb),
            float(cosine_sim_vec(q_emb, dst_emb) - cosine_sim_vec(q_emb, src_emb)),
            1.0 if e.src in topic_set else 0.0,
            1.0 if e.dst in topic_set else 0.0,
            1.0 if contains_mention(question, e.src_name) else 0.0,
            1.0 if contains_mention(question, e.dst_name) else 0.0,
            jaccard(question, e.relation),
            jaccard(question, e.key),
            jaccard(question, e.value),
            1.0 if (e.triple_type or '').upper() == 'ATTRIBUTE' else 0.0,
        ], dtype=np.float32)

        vec = np.concatenate([
            q_emb,
            src_emb,
            r_emb,
            dst_emb,
            k_emb,
            v_emb,
            src_dde.astype(np.float32),
            dst_dde.astype(np.float32),
            scalar,
        ], axis=0).astype(np.float32)

        feats[eid] = {
            'vector': vec,
            'scalar': scalar,
            'q_emb': q_emb,
            'src_emb': src_emb,
            'rel_emb': r_emb,
            'dst_emb': dst_emb,
            'key_emb': k_emb,
            'val_emb': v_emb,
            'src_dde': src_dde,
            'dst_dde': dst_dde,
        }
    return topic_nodes, feats


def build_node_feature_dict(
    question: str,
    node_names: List[str],
    node_emb: np.ndarray,
    kvedges: Dict[int, KVEdge],
    out_adj: Dict[int, List[int]],
    in_adj: Dict[int, List[int]],
    topic_nodes: List[int],
    feat_dict: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """
    为每个 node 构建一个 endness feature。
    目标：预测这个节点是否应该成为最终 sink。
    """
    if feat_dict:
        any_eid = next(iter(feat_dict))
        q_emb = feat_dict[any_eid]['q_emb']
        scalar_dim = len(feat_dict[any_eid]['scalar'])
    else:
        q_emb = np.zeros((node_emb.shape[1],), dtype=np.float32)
        scalar_dim = 15

    topic_set = set(topic_nodes)
    node_feats: Dict[int, Dict[str, Any]] = {}

    for nid, name in enumerate(node_names):
        in_eids = in_adj.get(nid, [])
        out_eids = out_adj.get(nid, [])

        if in_eids:
            in_mean = np.mean([feat_dict[eid]['scalar'] for eid in in_eids], axis=0).astype(np.float32)
        else:
            in_mean = np.zeros((scalar_dim,), dtype=np.float32)

        if out_eids:
            out_mean = np.mean([feat_dict[eid]['scalar'] for eid in out_eids], axis=0).astype(np.float32)
        else:
            out_mean = np.zeros((scalar_dim,), dtype=np.float32)

        in_max_val_sim = max([float(feat_dict[eid]['scalar'][4]) for eid in in_eids], default=0.0)
        in_max_dst_sim = max([float(feat_dict[eid]['scalar'][1]) for eid in in_eids], default=0.0)
        out_max_val_sim = max([float(feat_dict[eid]['scalar'][4]) for eid in out_eids], default=0.0)
        out_max_dst_sim = max([float(feat_dict[eid]['scalar'][1]) for eid in out_eids], default=0.0)

        node_scalar = np.array([
            cosine_sim_vec(q_emb, node_emb[nid]),          # 0: sim(q, node)
            1.0 if nid in topic_set else 0.0,              # 1: is topic node
            1.0 if contains_mention(question, name) else 0.0,  # 2: mentioned in q
            jaccard(question, name),                       # 3: lexical overlap
            float(len(in_eids)),                           # 4: in-degree
            float(len(out_eids)),                          # 5: out-degree
            float(in_max_val_sim),                         # 6
            float(in_max_dst_sim),                         # 7
            float(out_max_val_sim),                        # 8
            float(out_max_dst_sim),                        # 9
        ], dtype=np.float32)

        vec = np.concatenate([
            q_emb.astype(np.float32),
            node_emb[nid].astype(np.float32),
            node_scalar,
            in_mean,
            out_mean,
        ], axis=0).astype(np.float32)

        node_feats[nid] = {
            'vector': vec,
            'scalar': node_scalar,
            'name': name,
        }
    return node_feats


def weak_label_node_end(
    sample: Dict[str, Any],
    nid: int,
    in_adj: Dict[int, List[int]],
    kvedges: Dict[int, KVEdge],
) -> int:
    """
    弱监督：
    若某个节点的 incoming edge 中，有 value 匹配 answer，
    则把这个节点视为正的 end-node。
    """
    answer = norm_text(sample.get('answer', ''))
    if not answer:
        return 0
    for eid in in_adj.get(nid, []):
        if value_matches_answer(kvedges[eid].value, answer):
            return 1
    return 0


def mine_node_end_examples(
    sample: Dict[str, Any],
    node_feats: Dict[int, Dict[str, Any]],
    in_adj: Dict[int, List[int]],
    out_adj: Dict[int, List[int]],
    kvedges: Dict[int, KVEdge],
    neg_pos_ratio: int,
    min_negatives_per_sample: int,
) -> Tuple[List[int], List[int]]:
    """
    给 node-ending scorer 挖训练样本：
    - 正样本：答案节点
    - 难负样本：答案节点的后继节点，以及高 endness 但 out-degree 大的节点
    """
    pos_nodes = [nid for nid in node_feats.keys() if weak_label_node_end(sample, nid, in_adj, kvedges) == 1]
    if not pos_nodes:
        return [], []

    pos_set = set(pos_nodes)
    neg_nodes = [nid for nid in node_feats.keys() if nid not in pos_set]

    hard_negs: Set[int] = set()
    for pn in pos_nodes:
        # 答案节点本身如果有出边，也作为需要压制的对象看待
        hard_negs.add(pn)
        for eid in out_adj.get(pn, []):
            hard_negs.add(kvedges[eid].dst)

    ranked_neg = sorted(
        neg_nodes,
        key=lambda nid: float(
            1.2 * node_feats[nid]['scalar'][6] +   # incoming value sim
            1.0 * node_feats[nid]['scalar'][7] +   # incoming dst sim
            0.6 * node_feats[nid]['scalar'][0] -   # sim(q,node)
            0.4 * node_feats[nid]['scalar'][5]     # suppress big out-degree
        ) + (5.0 if nid in hard_negs else 0.0),
        reverse=True,
    )

    max_negs = max(len(pos_nodes) * neg_pos_ratio, min_negatives_per_sample)
    kept_negs = ranked_neg[:max_negs]
    return pos_nodes, kept_negs

# ============================================================
# 6) Trainable scorer
# ============================================================
class EdgeDataset(Dataset):
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.xs = torch.from_numpy(xs.astype(np.float32))
        self.ys = torch.from_numpy(ys.astype(np.float32))

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx: int):
        return self.xs[idx], self.ys[idx]

class BinaryDataset(Dataset):
    def __init__(self, xs: np.ndarray, ys: np.ndarray, indices: Optional[np.ndarray] = None):
        self.xs = xs
        self.ys = ys
        if indices is None:
            self.indices = np.arange(len(ys), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        y = float(self.ys[real_idx])
        return real_idx, y


def make_binary_collate(xs: np.ndarray) -> Callable[[List[Tuple[int, float]]], Tuple[torch.Tensor, torch.Tensor]]:
    def _collate(batch: List[Tuple[int, float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = np.fromiter((idx for idx, _ in batch), dtype=np.int64, count=len(batch))
        labels = np.fromiter((label for _, label in batch), dtype=np.float32, count=len(batch))
        xb = np.asarray(xs[indices], dtype=np.float32)
        yb = torch.from_numpy(labels)
        return torch.from_numpy(xb), yb

    return _collate


def build_loader(
    dataset: BinaryDataset,
    xs: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    prefetch_batches: int,
) -> DataLoader:
    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": max(0, int(num_workers)),
        "pin_memory": bool(pin_memory),
        "collate_fn": make_binary_collate(xs),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(2, int(prefetch_batches))
    return DataLoader(**loader_kwargs)


class ChunkedArrayWriter:
    def __init__(self, out_dir: str, prefix: str, feature_dtype=np.float16, label_dtype=np.float32, chunk_size: int = 50000):
        self.out_dir = out_dir
        self.prefix = prefix
        self.feature_dtype = feature_dtype
        self.label_dtype = label_dtype
        self.chunk_size = chunk_size
        self._x_buf: List[np.ndarray] = []
        self._y_buf: List[Union[int, float]] = []
        self._chunk_idx = 0
        self.num_rows = 0
        self.feature_dim: Optional[int] = None
        self.x_chunk_paths: List[str] = []
        self.y_chunk_paths: List[str] = []
        os.makedirs(self.out_dir, exist_ok=True)

    def add(self, vec: np.ndarray, label: Union[int, float]) -> None:
        arr = np.asarray(vec, dtype=self.feature_dtype)
        if self.feature_dim is None:
            self.feature_dim = int(arr.shape[0])
        self._x_buf.append(arr)
        self._y_buf.append(label)
        self.num_rows += 1
        if len(self._x_buf) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._x_buf:
            return
        x_arr = np.stack(self._x_buf).astype(self.feature_dtype, copy=False)
        y_arr = np.asarray(self._y_buf, dtype=self.label_dtype)
        x_path = os.path.join(self.out_dir, f'{self.prefix}_x_chunk_{self._chunk_idx:05d}.npy')
        y_path = os.path.join(self.out_dir, f'{self.prefix}_y_chunk_{self._chunk_idx:05d}.npy')
        print(
            f'[ChunkWriter:{self.prefix}] flush chunk={self._chunk_idx} rows={x_arr.shape[0]} '
            f'dim={x_arr.shape[1]} -> {x_path}'
        )
        np.save(x_path, x_arr)
        np.save(y_path, y_arr)
        self.x_chunk_paths.append(x_path)
        self.y_chunk_paths.append(y_path)
        self._chunk_idx += 1
        self._x_buf.clear()
        self._y_buf.clear()
        del x_arr, y_arr
        gc.collect()

    def finalize(self) -> Tuple[str, str]:
        self.flush()
        if self.feature_dim is None:
            raise ValueError(f'No rows written for {self.prefix}')

        final_x = os.path.join(self.out_dir, f'{self.prefix}_X.npy')
        final_y = os.path.join(self.out_dir, f'{self.prefix}_Y.npy')
        x_mm = open_memmap(final_x, mode='w+', dtype=self.feature_dtype, shape=(self.num_rows, self.feature_dim))
        y_mm = open_memmap(final_y, mode='w+', dtype=self.label_dtype, shape=(self.num_rows,))

        start = 0
        for chunk_idx, (x_path, y_path) in enumerate(zip(self.x_chunk_paths, self.y_chunk_paths)):
            x_chunk = np.load(x_path, mmap_mode='r')
            y_chunk = np.load(y_path, mmap_mode='r')
            end = start + x_chunk.shape[0]
            print(
                f'[ChunkWriter:{self.prefix}] merge chunk={chunk_idx} rows={x_chunk.shape[0]} '
                f'range=[{start},{end})'
            )
            x_mm[start:end] = x_chunk
            y_mm[start:end] = y_chunk
            start = end
            del x_chunk, y_chunk
            gc.collect()

        x_mm.flush()
        y_mm.flush()
        del x_mm, y_mm

        for p in self.x_chunk_paths + self.y_chunk_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        return final_x, final_y


class MLPScorer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class NodeEndScorer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        hid2 = max(64, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hid2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def collect_training_examples(
    args,
    samples: List[Dict[str, Any]],
    embedder: SentenceTransformer,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    total_edges = 0
    pos_edges = 0
    total_nodes = 0
    pos_nodes_total = 0

    used_samples = samples[:args.limit] if args.limit else samples
    train_embed_batch_size = args.batch_size
    train_cache_base = args.train_cache_path.strip()
    work_dir = (
        train_cache_base + '.dir'
        if train_cache_base
        else tempfile.mkdtemp(prefix='subgraphrag_train_build_')
    )
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    edge_writer = ChunkedArrayWriter(work_dir, 'edge')
    node_writer = ChunkedArrayWriter(work_dir, 'node')

    print(
        f'[Collect] start samples={len(used_samples)} '
        f'embed_batch={train_embed_batch_size} work_dir={work_dir}'
    )

    prepared = []
    all_texts: List[str] = []

    for sample in tqdm(used_samples, desc='Pass1: collect texts / graphs'):
        question = norm_text(sample.get('question', ''))
        node_names, kvedges, out_adj, in_adj = build_kvedge_graph(sample, supporting_only=False)
        if not kvedges:
            continue

        eids = sorted(kvedges.keys())
        rel_texts = [norm_text(kvedges[eid].relation or kvedges[eid].key) for eid in eids]
        key_texts = [norm_text(kvedges[eid].key) for eid in eids]
        val_texts = [norm_text(kvedges[eid].value) for eid in eids]
        node_names_norm = [norm_text(x) for x in node_names]

        prepared.append({
            'sample': sample,
            'question': question,
            'node_names': node_names,
            'node_names_norm': node_names_norm,
            'kvedges': kvedges,
            'out_adj': out_adj,
            'in_adj': in_adj,
            'eids': eids,
            'rel_texts': rel_texts,
            'key_texts': key_texts,
            'val_texts': val_texts,
        })

        all_texts.append(question)
        all_texts.extend(node_names_norm)
        all_texts.extend(rel_texts)
        all_texts.extend(key_texts)
        all_texts.extend(val_texts)

    if not prepared:
        raise ValueError('No valid samples found for training example construction.')

    print(
        f'[Collect] pass1_done valid_samples={len(prepared)} '
        f'unique_text_candidates={len(all_texts)}'
    )

    text_emb_cache = embed_texts_cached(
        embedder=embedder,
        texts=all_texts,
        batch_size=train_embed_batch_size,
    )
    num_unique_emb_texts = len(text_emb_cache)
    print(f'[Collect] embeddings_ready unique_texts={num_unique_emb_texts}')

    for item in tqdm(prepared, desc='Pass2: build weak labels / features'):
        sample = item['sample']
        question = item['question']
        node_names = item['node_names']
        node_names_norm = item['node_names_norm']
        kvedges = item['kvedges']
        out_adj = item['out_adj']
        in_adj = item['in_adj']
        eids = item['eids']
        rel_texts = item['rel_texts']
        key_texts = item['key_texts']
        val_texts = item['val_texts']

        q_emb = text_emb_cache[question]
        node_emb = np.stack([text_emb_cache[t] for t in node_names_norm]).astype(np.float32)
        rel_emb = np.stack([text_emb_cache[t] for t in rel_texts]).astype(np.float32)
        key_emb = np.stack([text_emb_cache[t] for t in key_texts]).astype(np.float32)
        val_emb = np.stack([text_emb_cache[t] for t in val_texts]).astype(np.float32)

        topic_nodes, feat_dict = build_edge_feature_dict(
            question=question,
            node_names=node_names,
            node_emb=node_emb,
            kvedges=kvedges,
            out_adj=out_adj,
            in_adj=in_adj,
            embedder=None,
            batch_size=train_embed_batch_size,
            topic_top_k=args.topic_top_k,
            dde_hops=args.dde_hops,
            mention_bonus=args.mention_bonus,
            text_emb_cache=None,
            q_emb=q_emb,
            rel_emb=rel_emb,
            key_emb=key_emb,
            val_emb=val_emb,
        )

        # ---------------------------
        # edge examples
        # ---------------------------
        pos_ids, neg_ids = [], []
        for eid, e in kvedges.items():
            y = weak_label_edge(sample, e)
            total_edges += 1
            if y == 1:
                pos_edges += 1
                pos_ids.append(eid)
            else:
                neg_ids.append(eid)

        neg_ids.sort(
            key=lambda eid: float(
                feat_dict[eid]['scalar'][3] + 0.5 * feat_dict[eid]['scalar'][4] + 0.3 * feat_dict[eid]['scalar'][1]
            ),
            reverse=True,
        )
        max_negs = max(len(pos_ids) * args.neg_pos_ratio, args.min_negatives_per_sample)
        kept_neg_ids = neg_ids[:max_negs]

        for eid in pos_ids:
            edge_writer.add(feat_dict[eid]['vector'], 1)
        for eid in kept_neg_ids:
            edge_writer.add(feat_dict[eid]['vector'], 0)

        # ---------------------------
        # node-ending examples
        # ---------------------------
        node_feat_dict = build_node_feature_dict(
            question=question,
            node_names=node_names,
            node_emb=node_emb,
            kvedges=kvedges,
            out_adj=out_adj,
            in_adj=in_adj,
            topic_nodes=topic_nodes,
            feat_dict=feat_dict,
        )

        pos_node_ids, neg_node_ids = mine_node_end_examples(
            sample=sample,
            node_feats=node_feat_dict,
            in_adj=in_adj,
            out_adj=out_adj,
            kvedges=kvedges,
            neg_pos_ratio=args.end_neg_pos_ratio,
            min_negatives_per_sample=args.end_min_negatives_per_sample,
        )

        total_nodes += len(node_feat_dict)
        pos_nodes_total += len(pos_node_ids)

        for nid in pos_node_ids:
            node_writer.add(node_feat_dict[nid]['vector'], 1)
        for nid in neg_node_ids:
            node_writer.add(node_feat_dict[nid]['vector'], 0)

    if edge_writer.num_rows == 0:
        raise ValueError('No edge training examples constructed.')
    if node_writer.num_rows == 0:
        raise ValueError('No node-ending training examples constructed.')
    print(
        f'[Build examples] edge_rows={edge_writer.num_rows} node_rows={node_writer.num_rows} '
        f'edge_dim={edge_writer.feature_dim} node_dim={node_writer.feature_dim}'
    )
    print('[Collect] releasing pass1/pass2 intermediate objects before memmap finalize')

    del prepared
    del all_texts
    del text_emb_cache
    gc.collect()

    print('[Collect] finalizing edge chunks -> memmap')
    edge_x_path, edge_y_path = edge_writer.finalize()
    print('[Collect] finalizing node chunks -> memmap')
    node_x_path, node_y_path = node_writer.finalize()
    print(
        f'[Collect] memmap_ready edge_x={edge_x_path} edge_y={edge_y_path} '
        f'node_x={node_x_path} node_y={node_y_path}'
    )
    X_edge = np.load(edge_x_path, mmap_mode='r')
    Y_edge = np.load(edge_y_path, mmap_mode='r')
    X_node = np.load(node_x_path, mmap_mode='r')
    Y_node = np.load(node_y_path, mmap_mode='r')

    meta = {
        'num_edge_examples': int(len(Y_edge)),
        'num_edge_pos': int(np.asarray(Y_edge).sum()),
        'num_edge_neg': int(len(Y_edge) - np.asarray(Y_edge).sum()),
        'num_node_examples': int(len(Y_node)),
        'num_node_pos': int(np.asarray(Y_node).sum()),
        'num_node_neg': int(len(Y_node) - np.asarray(Y_node).sum()),
        'raw_total_edges': int(total_edges),
        'raw_pos_edges': int(pos_edges),
        'raw_total_nodes': int(total_nodes),
        'raw_pos_nodes': int(pos_nodes_total),
        'edge_input_dim': int(X_edge.shape[1]),
        'node_input_dim': int(X_node.shape[1]),
        'num_unique_emb_texts': int(num_unique_emb_texts),
        'train_embed_batch_size': int(train_embed_batch_size),
        'cache_format': 'npy_memmap',
        'cache_build_dir': work_dir,
    }
    return X_edge, Y_edge, X_node, Y_node, meta

def split_indices(num_rows: int, dev_ratio: float, seed: int):
    idx = np.arange(num_rows)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    split = int(num_rows * (1.0 - dev_ratio))
    split = max(1, min(split, num_rows - 1))
    return idx[:split], idx[split:]


def train_one_binary_model(
    args,
    model: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    train_batch_size: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    threshold: float,
    device: torch.device,
    seed: int,
    tag: str = 'model',
) -> Tuple[Dict[str, torch.Tensor], float]:
    train_idx, dev_idx = split_indices(len(Y), dev_ratio=0.1, seed=seed)
    Y_np = np.asarray(Y, dtype=np.float32)
    log(
        f'[{tag}] dataset_ready total={len(Y_np)} train={len(train_idx)} dev={len(dev_idx)} '
        f'input_dim={X.shape[1]}'
    )

    train_ds = BinaryDataset(X, Y_np, train_idx)
    dev_ds = BinaryDataset(X, Y_np, dev_idx)
    train_loader = build_loader(
        train_ds,
        X,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda' and not args.disable_pin_memory),
        prefetch_batches=args.prefetch_batches,
    )
    dev_loader = build_loader(
        dev_ds,
        X,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda' and not args.disable_pin_memory),
        prefetch_batches=args.prefetch_batches,
    )
    log(
        f'[{tag}] loaders_ready train_batches={len(train_loader)} dev_batches={len(dev_loader)} '
        f'num_workers={args.num_workers} pin_memory={device.type == "cuda" and not args.disable_pin_memory}'
    )

    pos_count = float(Y_np[train_idx].sum())
    neg_count = float(len(train_idx) - pos_count)
    pos_weight = torch.tensor([max(1.0, neg_count / max(1.0, pos_count))], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience//2, verbose=True)

    best_state = None
    best_dev = -1.0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item()) * len(yb)
            if batch_idx == 1 or batch_idx % args.train_log_interval == 0 or batch_idx == len(train_loader):
                log(
                    f'[{tag}][Epoch {epoch}] train_batch={batch_idx}/{len(train_loader)} '
                    f'loss={float(loss.item()):.4f}'
                )

        dev_f1 = eval_binary_f1(
            model,
            dev_loader,
            device,
            threshold=threshold,
            tag=f'{tag}/dev',
            log_interval=args.eval_log_interval,
        )
        scheduler.step(dev_f1)

        train_loss = total_loss / max(1, len(train_ds))
        log(f'[{tag}][Epoch {epoch}] train_loss={train_loss:.4f} dev_f1={dev_f1:.4f}')

        if dev_f1 > best_dev:
            best_dev = dev_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f'[{tag}] early stop, patience={patience}')
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    return best_state, best_dev

class SharedEncoder(nn.Module):
    """
    共享特征编码器 - Edge 和 Node 任务共用
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeHead(nn.Module):
    """Edge 任务头"""
    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, max(64, in_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, in_dim // 2), 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class NodeHead(nn.Module):
    """Node 任务头"""
    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        hid2 = max(64, in_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hid2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class JointScorer(nn.Module):
    """
    联合评分模型 - 独立输入投影 + 共享编码器 + 两个任务头
    """
    def __init__(self, edge_in_dim: int, node_in_dim: int, 
                 hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        
        # 关键修正：为 Edge 和 Node 分别添加输入投影层
        self.edge_input_proj = nn.Linear(edge_in_dim, hidden_dim)
        self.node_input_proj = nn.Linear(node_in_dim, hidden_dim)
        
        # 共享编码器 (输入已经是 hidden_dim)
        self.shared_encoder = SharedEncoder(hidden_dim, hidden_dim, dropout)
        
        # 任务头
        self.edge_head = EdgeHead(self.shared_encoder.output_dim, dropout)
        self.node_head = NodeHead(self.shared_encoder.output_dim, dropout)
        
        # 记录配置 (用于保存/加载)
        self.edge_in_dim = edge_in_dim
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
    
    def forward_edge(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - Edge 任务"""
        # 输入投影 -> 共享编码 -> 任务头
        projected = self.edge_input_proj(x)
        encoded = self.shared_encoder(projected)
        return self.edge_head(encoded)
    
    def forward_node(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - Node 任务"""
        # 输入投影 -> 共享编码 -> 任务头
        projected = self.node_input_proj(x)
        encoded = self.shared_encoder(projected)
        return self.node_head(encoded)
    
    def forward(self, x_edge: torch.Tensor, x_node: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """联合前向传播"""
        return self.forward_edge(x_edge), self.forward_node(x_node)

# ============================================================
# 联合训练函数
# ============================================================

def train_joint_model(
    args,
    X_edge: np.ndarray,
    Y_edge: np.ndarray,
    X_node: np.ndarray,
    Y_node: np.ndarray,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], float, float]:
    """
    联合训练 Edge 和 Node 模型
    """
    # 划分训练/验证集
    rng = np.random.RandomState(args.seed)
    
    # Edge 数据划分
    idx_edge = np.arange(len(Y_edge))
    rng.shuffle(idx_edge)
    split_edge = int(len(Y_edge) * 0.9)
    tr_edge_idx, dv_edge_idx = idx_edge[:split_edge], idx_edge[split_edge:]
    
    # Node 数据划分
    idx_node = np.arange(len(Y_node))
    rng.shuffle(idx_node)
    split_node = int(len(Y_node) * 0.9)
    tr_node_idx, dv_node_idx = idx_node[:split_node], idx_node[split_node:]
    Y_edge_np = np.asarray(Y_edge, dtype=np.float32)
    Y_node_np = np.asarray(Y_node, dtype=np.float32)
    log(
        f'[Joint] dataset_ready edge_total={len(Y_edge_np)} edge_train={len(tr_edge_idx)} edge_dev={len(dv_edge_idx)} '
        f'node_total={len(Y_node_np)} node_train={len(tr_node_idx)} node_dev={len(dv_node_idx)} '
        f'edge_dim={X_edge.shape[1]} node_dim={X_node.shape[1]}'
    )
    
    # 创建 DataLoader
    train_edge_ds = BinaryDataset(X_edge, Y_edge_np, tr_edge_idx)
    train_node_ds = BinaryDataset(X_node, Y_node_np, tr_node_idx)
    dev_edge_ds = BinaryDataset(X_edge, Y_edge_np, dv_edge_idx)
    dev_node_ds = BinaryDataset(X_node, Y_node_np, dv_node_idx)
    
    pin_memory = device.type == 'cuda' and not args.disable_pin_memory
    train_edge_loader = build_loader(
        train_edge_ds,
        X_edge,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_batches=args.prefetch_batches,
    )
    train_node_loader = build_loader(
        train_node_ds,
        X_node,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_batches=args.prefetch_batches,
    )
    dev_edge_loader = build_loader(
        dev_edge_ds,
        X_edge,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_batches=args.prefetch_batches,
    )
    dev_node_loader = build_loader(
        dev_node_ds,
        X_node,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_batches=args.prefetch_batches,
    )
    log(
        f'[Joint] loaders_ready edge_train_batches={len(train_edge_loader)} edge_dev_batches={len(dev_edge_loader)} '
        f'node_train_batches={len(train_node_loader)} node_dev_batches={len(dev_node_loader)} '
        f'num_workers={args.num_workers} pin_memory={pin_memory}'
    )
    
    # 创建联合模型 - 传入正确的输入维度
    model = JointScorer(
        edge_in_dim=X_edge.shape[1],  # 关键：使用实际 edge 特征维度
        node_in_dim=X_node.shape[1],  # 关键：使用实际 node 特征维度
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    
    # 计算类别权重
    pos_count_edge = float(Y_edge_np[tr_edge_idx].sum())
    neg_count_edge = float(len(tr_edge_idx) - pos_count_edge)
    pos_weight_edge = torch.tensor([max(1.0, neg_count_edge / max(1.0, pos_count_edge))], device=device)
    
    pos_count_node = float(Y_node_np[tr_node_idx].sum())
    neg_count_node = float(len(tr_node_idx) - pos_count_node)
    pos_weight_node = torch.tensor([max(1.0, neg_count_node / max(1.0, pos_count_node))], device=device)
    
    criterion_edge = nn.BCEWithLogitsLoss(pos_weight=pos_weight_edge)
    criterion_node = nn.BCEWithLogitsLoss(pos_weight=pos_weight_node)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.patience//2, verbose=True)
    
    # 训练循环
    best_state = None
    best_dev_edge = -1.0
    best_dev_node = -1.0
    best_dev_combined = -1.0
    bad_epochs = 0
    
    # 获取 lambda 参数 (如果存在)
    joint_lambda = getattr(args, 'joint_lambda', 0.5)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss_edge = 0.0
        total_loss_node = 0.0
        
        edge_iter = iter(train_edge_loader)
        node_iter = iter(train_node_loader)
        
        max_batches = max(len(train_edge_loader), len(train_node_loader))
        
        epoch_t0 = time.perf_counter()
        for batch_idx in range(1, max_batches + 1):
            # Edge batch
            loss_edge = 0.0
            loss_edge_value = 0.0
            try:
                xb_edge, yb_edge = next(edge_iter)
                xb_edge = xb_edge.to(device, non_blocking=True)
                yb_edge = yb_edge.to(device, non_blocking=True)
                
                logits_edge = model.forward_edge(xb_edge)
                loss_edge = criterion_edge(logits_edge, yb_edge)
                loss_edge_value = float(loss_edge.item())
                
                total_loss_edge += loss_edge_value * len(yb_edge)
            except StopIteration:
                pass
            
            # Node batch
            loss_node = 0.0
            loss_node_value = 0.0
            try:
                xb_node, yb_node = next(node_iter)
                xb_node = xb_node.to(device, non_blocking=True)
                yb_node = yb_node.to(device, non_blocking=True)
                
                logits_node = model.forward_node(xb_node)
                loss_node = criterion_node(logits_node, yb_node)
                loss_node_value = float(loss_node.item())
                
                total_loss_node += loss_node_value * len(yb_node)
            except StopIteration:
                pass
            
            # 联合 Loss
            total_loss = loss_edge + joint_lambda * loss_node
            
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if (
                batch_idx == 1
                or batch_idx % args.train_log_interval == 0
                or batch_idx == max_batches
            ):
                log(
                    f'[Joint][Epoch {epoch}] train_batch={batch_idx}/{max_batches} '
                    f'edge_loss={loss_edge_value:.4f} node_loss={loss_node_value:.4f}'
                )
        
        # 验证
        dev_f1_edge = eval_binary_f1_for_joint(
            model,
            dev_edge_loader,
            device,
            args.threshold,
            task='edge',
            tag=f'Joint/edge_dev/epoch_{epoch}',
            log_interval=args.eval_log_interval,
        )
        dev_f1_node = eval_binary_f1_for_joint(
            model,
            dev_node_loader,
            device,
            args.end_threshold,
            task='node',
            tag=f'Joint/node_dev/epoch_{epoch}',
            log_interval=args.eval_log_interval,
        )
        dev_f1_combined = (dev_f1_edge + dev_f1_node) / 2.0
        
        scheduler.step(dev_f1_combined)
        
        train_loss_edge = total_loss_edge / max(1, len(train_edge_ds))
        train_loss_node = total_loss_node / max(1, len(train_node_ds))
        
        epoch_s = time.perf_counter() - epoch_t0
        log(
            f'[Joint][Epoch {epoch}] edge_loss={train_loss_edge:.4f} edge_f1={dev_f1_edge:.4f} | '
            f'node_loss={train_loss_node:.4f} node_f1={dev_f1_node:.4f} | combined_f1={dev_f1_combined:.4f} | '
            f'epoch_s={epoch_s:.1f}'
        )
        
        # 早停
        if dev_f1_combined > best_dev_combined:
            best_dev_combined = dev_f1_combined
            best_dev_edge = dev_f1_edge
            best_dev_node = dev_f1_node
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log(f'[Joint] early stop, patience={args.patience}')
                break
    
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    
    return best_state, best_dev_edge, best_dev_node


def eval_binary_f1_for_joint(
    model: JointScorer, 
    loader: DataLoader, 
    device: torch.device, 
    threshold: float,
    task: str = 'edge',
    tag: str = 'joint/dev',
    log_interval: int = 0,
) -> float:
    """联合模型的评估函数"""
    model.eval()
    tp = fp = fn = 0
    
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            if task == 'edge':
                logits = model.forward_edge(xb)
            else:
                logits = model.forward_node(xb)
            
            probs = torch.sigmoid(logits)
            pred = (probs >= threshold).float()
            
            tp += int(((pred == 1) & (yb == 1)).sum().item())
            fp += int(((pred == 1) & (yb == 0)).sum().item())
            fn += int(((pred == 0) & (yb == 1)).sum().item())
            if log_interval > 0 and (
                batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(loader)
            ):
                log(f'[{tag}] eval_batch={batch_idx}/{len(loader)}')
    
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# def train_model(args, samples: List[Dict[str, Any]], embedder: SentenceTransformer) -> None:
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)

#     cache_path = args.train_cache_path.strip()

#     if cache_path and os.path.exists(cache_path) and not args.rebuild_train_cache:
#         print(f"[Load train cache] {cache_path}")
#         with open(cache_path, "rb") as f:
#             cache_obj = pickle.load(f)
#         X_edge = cache_obj["X_edge"]
#         Y_edge = cache_obj["Y_edge"]
#         X_node = cache_obj["X_node"]
#         Y_node = cache_obj["Y_node"]
#         meta = cache_obj["meta"]
#     else:
#         X_edge, Y_edge, X_node, Y_node, meta = collect_training_examples(args, samples, embedder)

#         if cache_path:
#             cache_dir = os.path.dirname(cache_path)
#             if cache_dir:
#                 os.makedirs(cache_dir, exist_ok=True)
#             with open(cache_path, "wb") as f:
#                 pickle.dump(
#                     {
#                         "X_edge": X_edge,
#                         "Y_edge": Y_edge,
#                         "X_node": X_node,
#                         "Y_node": Y_node,
#                         "meta": meta,
#                     },
#                     f,
#                     protocol=pickle.HIGHEST_PROTOCOL,
#                 )
#             print(f"[Saved train cache] {cache_path}")

#     print(json.dumps(meta, indent=2, ensure_ascii=False))

#     device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

#     edge_model = MLPScorer(
#         in_dim=X_edge.shape[1],
#         hidden_dim=args.hidden_dim,
#         dropout=args.dropout,
#     ).to(device)

#     node_model = NodeEndScorer(
#         in_dim=X_node.shape[1],
#         hidden_dim=args.end_hidden_dim,
#         dropout=args.dropout,
#     ).to(device)

#     edge_state, edge_best_dev = train_one_binary_model(
#         model=edge_model,
#         X=X_edge,
#         Y=Y_edge,
#         train_batch_size=args.train_batch_size,
#         lr=args.lr,
#         weight_decay=args.weight_decay,
#         epochs=args.epochs,
#         patience=args.patience,
#         threshold=args.threshold,
#         device=device,
#         seed=args.seed,
#         tag='edge',
#     )

#     node_state, node_best_dev = train_one_binary_model(
#         model=node_model,
#         X=X_node,
#         Y=Y_node,
#         train_batch_size=args.train_batch_size,
#         lr=args.end_lr,
#         weight_decay=args.weight_decay,
#         epochs=args.epochs,
#         patience=args.patience,
#         threshold=args.end_threshold,
#         device=device,
#         seed=args.seed,
#         tag='node_end',
#     )

#     os.makedirs(os.path.dirname(args.model_ckpt) or '.', exist_ok=True)
#     ckpt = {
#         'edge_state_dict': edge_state,
#         'node_state_dict': node_state,
#         'edge_input_dim': int(X_edge.shape[1]),
#         'node_input_dim': int(X_node.shape[1]),
#         'hidden_dim': int(args.hidden_dim),
#         'end_hidden_dim': int(args.end_hidden_dim),
#         'dropout': float(args.dropout),
#         'threshold': float(args.threshold),
#         'end_threshold': float(args.end_threshold),
#         'meta': meta,
#         'train_args': vars(args),
#     }
#     torch.save(ckpt, args.model_ckpt)
#     print(f'[Saved] {args.model_ckpt}')
#     print(json.dumps({
#         'edge_best_dev_f1': edge_best_dev,
#         'node_best_dev_f1': node_best_dev,
#         **meta
#     }, ensure_ascii=False, indent=2))

def train_model(args, samples: List[Dict[str, Any]], embedder: SentenceTransformer) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cache_path = args.train_cache_path.strip()
    cache_dir = cache_path + '.dir' if cache_path else ''
    log(
        f'[Train] start samples={len(samples)} mode={"joint" if args.joint_training else "independent"} '
        f'cache_path={cache_path or "<none>"} rebuild_cache={args.rebuild_train_cache}'
    )
    
    # 加载或构建训练数据
    if cache_path and os.path.isdir(cache_dir) and not args.rebuild_train_cache:
        log(f"[Load train cache dir] {cache_dir}")
        X_edge = np.load(os.path.join(cache_dir, 'edge_X.npy'), mmap_mode='r')
        Y_edge = np.load(os.path.join(cache_dir, 'edge_Y.npy'), mmap_mode='r')
        X_node = np.load(os.path.join(cache_dir, 'node_X.npy'), mmap_mode='r')
        Y_node = np.load(os.path.join(cache_dir, 'node_Y.npy'), mmap_mode='r')
        with open(os.path.join(cache_dir, 'meta.json'), 'r', encoding='utf-8') as f:
            meta = json.load(f)
        log('[Train] cache_dir_loaded memmap arrays are ready')
    elif cache_path and os.path.exists(cache_path) and not args.rebuild_train_cache:
        log(f"[Load legacy train cache] {cache_path}")
        with open(cache_path, "rb") as f:
            cache_obj = pickle.load(f)
        X_edge = cache_obj["X_edge"]
        Y_edge = cache_obj["Y_edge"]
        X_node = cache_obj["X_node"]
        Y_node = cache_obj["Y_node"]
        meta = cache_obj["meta"]
        log('[Train] legacy_cache_loaded arrays are in memory')
    else:
        log('[Train] cache miss -> building training examples')
        X_edge, Y_edge, X_node, Y_node, meta = collect_training_examples(args, samples, embedder)
        
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, 'meta.json'), 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            log(f"[Saved train cache dir] {cache_dir}")
    
    log(json.dumps(meta, indent=2, ensure_ascii=False))
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    log(f'[Train] device={device} edge_shape={tuple(X_edge.shape)} node_shape={tuple(X_node.shape)}')
    
    # ============ 关键改动：选择训练模式 ============
    if args.joint_training:
        log("[Mode] Joint Training with Shared Encoder")
        joint_state, edge_best_dev, node_best_dev = train_joint_model(
            args, X_edge, Y_edge, X_node, Y_node, device
        )
        
        os.makedirs(os.path.dirname(args.model_ckpt) or '.', exist_ok=True)
        ckpt = {
            'joint_state_dict': joint_state,
            'edge_input_dim': int(X_edge.shape[1]),
            'node_input_dim': int(X_node.shape[1]),
            'hidden_dim': int(args.hidden_dim),
            'dropout': float(args.dropout),
            'threshold': float(args.threshold),
            'end_threshold': float(args.end_threshold),
            'meta': meta,
            'train_args': vars(args),
            'is_joint': True,
        }
        torch.save(ckpt, args.model_ckpt)
        log(f'[Saved Joint Model] {args.model_ckpt}')
        
    else:
        log("[Mode] Independent Training")
        # 原有独立训练逻辑
        edge_model = MLPScorer(
            in_dim=X_edge.shape[1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)
        
        node_model = NodeEndScorer(
            in_dim=X_node.shape[1],
            hidden_dim=args.end_hidden_dim,
            dropout=args.dropout,
        ).to(device)
        
        edge_state, edge_best_dev = train_one_binary_model(
            args=args,
            model=edge_model, X=X_edge, Y=Y_edge,
            train_batch_size=args.train_batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            epochs=args.epochs, patience=args.patience,
            threshold=args.threshold, device=device,
            seed=args.seed, tag='edge',
        )
        
        node_state, node_best_dev = train_one_binary_model(
            args=args,
            model=node_model, X=X_node, Y=Y_node,
            train_batch_size=args.train_batch_size,
            lr=args.end_lr, weight_decay=args.weight_decay,
            epochs=args.epochs, patience=args.patience,
            threshold=args.end_threshold, device=device,
            seed=args.seed, tag='node_end',
        )
        
        os.makedirs(os.path.dirname(args.model_ckpt) or '.', exist_ok=True)
        ckpt = {
            'edge_state_dict': edge_state,
            'node_state_dict': node_state,
            'edge_input_dim': int(X_edge.shape[1]),
            'node_input_dim': int(X_node.shape[1]),
            'hidden_dim': int(args.hidden_dim),
            'end_hidden_dim': int(args.end_hidden_dim),
            'dropout': float(args.dropout),
            'threshold': float(args.threshold),
            'end_threshold': float(args.end_threshold),
            'meta': meta,
            'train_args': vars(args),
            'is_joint': False,
        }
        torch.save(ckpt, args.model_ckpt)
        log(f'[Saved] {args.model_ckpt}')
    
    log(json.dumps({
        'edge_best_dev_f1': edge_best_dev,
        'node_best_dev_f1': node_best_dev,
        **meta
    }, ensure_ascii=False, indent=2))


def eval_binary_f1(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    tag: str = 'dev',
    log_interval: int = 0,
) -> float:
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            probs = torch.sigmoid(model(xb))
            pred = (probs >= threshold).float()
            tp += int(((pred == 1) & (yb == 1)).sum().item())
            fp += int(((pred == 1) & (yb == 0)).sum().item())
            fn += int(((pred == 0) & (yb == 1)).sum().item())
            if log_interval > 0 and (
                batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(loader)
            ):
                log(f'[{tag}] eval_batch={batch_idx}/{len(loader)}')
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

# def load_model(model_ckpt: str, cpu: bool = False) -> Tuple[MLPScorer, NodeEndScorer, Dict[str, Any], torch.device]:
#     device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
#     ckpt = torch.load(model_ckpt, map_location='cpu')

#     edge_model = MLPScorer(
#         in_dim=ckpt['edge_input_dim'],
#         hidden_dim=ckpt.get('hidden_dim', 512),
#         dropout=ckpt.get('dropout', 0.1),
#     )
#     node_model = NodeEndScorer(
#         in_dim=ckpt['node_input_dim'],
#         hidden_dim=ckpt.get('end_hidden_dim', 256),
#         dropout=ckpt.get('dropout', 0.1),
#     )

#     edge_model.load_state_dict(ckpt['edge_state_dict'])
#     node_model.load_state_dict(ckpt['node_state_dict'])

#     edge_model.to(device).eval()
#     node_model.to(device).eval()

#     return edge_model, node_model, ckpt, device

def load_model(model_ckpt: str, cpu: bool = False) -> Tuple[Union[MLPScorer, JointScorer], 
                                                             Union[NodeEndScorer, JointScorer], 
                                                             Dict[str, Any], torch.device]:
    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    ckpt = torch.load(model_ckpt, map_location='cpu')
    
    is_joint = ckpt.get('is_joint', False)
    
    if is_joint:
        # 加载联合模型
        model = JointScorer(
            edge_in_dim=ckpt['edge_input_dim'],
            node_in_dim=ckpt['node_input_dim'],
            hidden_dim=ckpt.get('hidden_dim', 512),
            dropout=ckpt.get('dropout', 0.1),
        )
        model.load_state_dict(ckpt['joint_state_dict'])
        model.to(device).eval()
        return model, model, ckpt, device
    else:
        # 加载独立模型
        edge_model = MLPScorer(
            in_dim=ckpt['edge_input_dim'],
            hidden_dim=ckpt.get('hidden_dim', 512),
            dropout=ckpt.get('dropout', 0.1),
        )
        node_model = NodeEndScorer(
            in_dim=ckpt['node_input_dim'],
            hidden_dim=ckpt.get('end_hidden_dim', 256),
            dropout=ckpt.get('dropout', 0.1),
        )
        
        edge_model.load_state_dict(ckpt['edge_state_dict'])
        node_model.load_state_dict(ckpt['node_state_dict'])
        
        edge_model.to(device).eval()
        node_model.to(device).eval()
        
        return edge_model, node_model, ckpt, device

# ============================================================
# 7) Retrieval / DAG post-processing
# ============================================================
def score_edges_with_model(
    model: MLPScorer,
    device: torch.device,
    feat_dict: Dict[int, Dict[str, Any]],
    infer_batch_size: int,
) -> Dict[int, float]:
    if not feat_dict:
        return {}

    eids = sorted(feat_dict.keys())
    X = np.stack([feat_dict[eid]['vector'] for eid in eids]).astype(np.float32)
    scores = _batched_sigmoid_scores(model, device, X, infer_batch_size)
    return {eid: float(scores[i]) for i, eid in enumerate(eids)}


def score_nodes_with_model(
    model: NodeEndScorer,
    device: torch.device,
    node_feat_dict: Dict[int, Dict[str, Any]],
    infer_batch_size: int,
) -> Dict[int, float]:
    if not node_feat_dict:
        return {}

    nids = sorted(node_feat_dict.keys())
    X = np.stack([node_feat_dict[nid]['vector'] for nid in nids]).astype(np.float32)
    scores = _batched_sigmoid_scores(model, device, X, infer_batch_size)
    return {nid: float(scores[i]) for i, nid in enumerate(nids)}


def _batched_sigmoid_scores(
    model: nn.Module,
    device: torch.device,
    X: np.ndarray,
    infer_batch_size: int,
    forward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> np.ndarray:
    if X.size == 0:
        return np.empty((0,), dtype=np.float32)

    X = np.ascontiguousarray(X, dtype=np.float32)
    scores = np.empty((X.shape[0],), dtype=np.float32)
    fwd = model if forward_fn is None else forward_fn

    with torch.no_grad():
        for start in range(0, X.shape[0], infer_batch_size):
            end = min(start + infer_batch_size, X.shape[0])
            xb = torch.from_numpy(X[start:end]).to(device)
            logits = fwd(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            scores[start:end] = probs
    return scores


def apply_two_stage_joint_scores(
    kvedges: Dict[int, KVEdge],
    edge_scores: Dict[int, float],
    node_end_scores: Dict[int, float],
    alpha: float,
    beta: float,
    gamma: float,
) -> None:
    """
    两阶段联合打分：
      joint(u->v) = edge_score + alpha * end(v) - beta * end(u) - gamma * end(u) * best_in(u)
    """
    incoming_best: Dict[int, float] = defaultdict(float)
    for eid, e in kvedges.items():
        incoming_best[e.dst] = max(incoming_best[e.dst], edge_scores.get(eid, 0.0))

    for eid, e in kvedges.items():
        edge_s = float(edge_scores.get(eid, 0.0))
        e.edge_score = edge_s
        dst_end = float(node_end_scores.get(e.dst, 0.0))
        src_end = float(node_end_scores.get(e.src, 0.0))
        src_incoming = float(incoming_best.get(e.src, 0.0))

        joint = edge_s + alpha * dst_end - beta * src_end - gamma * (src_end * src_incoming)
        kvedges[eid].score = float(joint)


def select_subgraph_edges(topic_nodes: List[int], kvedges: Dict[int, KVEdge], out_adj: Dict[int, List[int]], max_edges: int, max_nodes: int, per_src_cap: int, expansion_hops: int, seed_edge_topk: int) -> Tuple[Set[int], Set[int]]:
    selected_edges: Set[int] = set()
    selected_nodes: Set[int] = set(topic_nodes)

    # Stage 1: global top-k edges with per-source sparsification.
    ranked = sorted(kvedges.keys(), key=lambda eid: kvedges[eid].score, reverse=True)
    src_count: Dict[int, int] = defaultdict(int)
    for eid in ranked:
        e = kvedges[eid]
        if src_count[e.src] >= per_src_cap:
            continue
        new_nodes = int(e.src not in selected_nodes) + int(e.dst not in selected_nodes)
        if len(selected_nodes) + new_nodes > max_nodes:
            continue
        selected_edges.add(eid)
        selected_nodes.add(e.src)
        selected_nodes.add(e.dst)
        src_count[e.src] += 1
        if len(selected_edges) >= min(seed_edge_topk, max_edges):
            break

    # Stage 2: topic-centered expansion.
    frontier = deque(topic_nodes)
    seen_frontier = set(topic_nodes)
    cur_hop = 0
    while frontier and cur_hop < expansion_hops and len(selected_edges) < max_edges:
        for _ in range(len(frontier)):
            u = frontier.popleft()
            cand = sorted(out_adj.get(u, []), key=lambda eid: kvedges[eid].score, reverse=True)
            local = 0
            for eid in cand:
                e = kvedges[eid]
                if eid in selected_edges:
                    continue
                if local >= per_src_cap:
                    break
                new_nodes = int(e.src not in selected_nodes) + int(e.dst not in selected_nodes)
                if len(selected_nodes) + new_nodes > max_nodes:
                    continue
                selected_edges.add(eid)
                selected_nodes.add(e.src)
                selected_nodes.add(e.dst)
                local += 1
                if e.dst not in seen_frontier:
                    frontier.append(e.dst)
                    seen_frontier.add(e.dst)
                if len(selected_edges) >= max_edges:
                    break
        cur_hop += 1

    # Stage 3: connectivity patch.
    if len(selected_edges) < max_edges:
        for eid in ranked:
            e = kvedges[eid]
            if eid in selected_edges:
                continue
            if len(selected_edges) >= max_edges:
                break
            if e.src in selected_nodes or e.dst in selected_nodes:
                new_nodes = int(e.src not in selected_nodes) + int(e.dst not in selected_nodes)
                if len(selected_nodes) + new_nodes > max_nodes:
                    continue
                selected_edges.add(eid)
                selected_nodes.add(e.src)
                selected_nodes.add(e.dst)
    return selected_nodes, selected_edges


def reverse_beam_expand_from_sink_edges(
    selected_nodes: Set[int],
    selected_edges: Set[int],
    kvedges: Dict[int, KVEdge],
    in_adj: Dict[int, List[int]],
    max_edges: int,
    max_nodes: int,
    sink_edge_topk: int,
    reverse_hops: int,
    beam_width: int,
) -> Tuple[Set[int], Set[int]]:
    """
    Starting from top sink edges in the current DAG, search backwards on the edge graph:
    prev_edge -> cur_edge iff prev_edge.dst == cur_edge.src.

    Newly added edges are skipped if they exceed the graph budget or would create a cycle
    in the current selected subgraph.
    """
    if sink_edge_topk <= 0 or reverse_hops <= 0 or beam_width <= 0 or not selected_edges:
        return selected_nodes, selected_edges

    entity_outdeg: Dict[int, int] = defaultdict(int)
    for eid in selected_edges:
        entity_outdeg[kvedges[eid].src] += 1

    sink_edges = [
        eid for eid in selected_edges
        if entity_outdeg.get(kvedges[eid].dst, 0) == 0
    ]
    if not sink_edges:
        return selected_nodes, selected_edges

    ranked_sink_edges = sorted(
        sink_edges,
        key=lambda eid: (kvedges[eid].score, eid),
        reverse=True,
    )[:sink_edge_topk]

    kept_out: Dict[int, Set[int]] = defaultdict(set)
    for eid in selected_edges:
        e = kvedges[eid]
        kept_out[e.src].add(e.dst)

    for sink_eid in ranked_sink_edges:
        frontier: List[Tuple[int, float]] = [(sink_eid, float(kvedges[sink_eid].score))]
        visited_eids: Set[int] = {sink_eid}

        for _ in range(reverse_hops):
            if not frontier or len(selected_edges) >= max_edges:
                break

            candidates: List[Tuple[float, int]] = []
            for cur_eid, path_score in frontier:
                cur_edge = kvedges[cur_eid]
                for prev_eid in in_adj.get(cur_edge.src, []):
                    if prev_eid in visited_eids:
                        continue
                    visited_eids.add(prev_eid)
                    prev_edge = kvedges[prev_eid]
                    candidates.append((path_score + float(prev_edge.score), prev_eid))

            if not candidates:
                break

            candidates.sort(key=lambda x: (x[0], kvedges[x[1]].score, -x[1]), reverse=True)
            next_frontier: List[Tuple[int, float]] = []

            for cand_score, cand_eid in candidates[:beam_width]:
                cand_edge = kvedges[cand_eid]

                if cand_eid not in selected_edges:
                    if len(selected_edges) >= max_edges:
                        break
                    new_nodes = int(cand_edge.src not in selected_nodes) + int(cand_edge.dst not in selected_nodes)
                    if len(selected_nodes) + new_nodes > max_nodes:
                        continue
                    if _would_create_cycle(cand_edge.src, cand_edge.dst, kept_out):
                        continue

                    selected_edges.add(cand_eid)
                    selected_nodes.add(cand_edge.src)
                    selected_nodes.add(cand_edge.dst)
                    kept_out[cand_edge.src].add(cand_edge.dst)

                next_frontier.append((cand_eid, cand_score))

            frontier = next_frontier

    return selected_nodes, selected_edges


def _would_create_cycle(src: int, dst: int, kept_out: Dict[int, Set[int]]) -> bool:
    if src == dst:
        return True
    stack = [dst]
    vis = set()
    while stack:
        u = stack.pop()
        if u == src:
            return True
        if u in vis:
            continue
        vis.add(u)
        for v in kept_out.get(u, ()): 
            if v not in vis:
                stack.append(v)
    return False


def break_cycles_to_dag(nodes: Set[int], edges_kept: Set[int], kvedges: Dict[int, KVEdge]) -> Set[int]:
    ordered = sorted(edges_kept, key=lambda eid: kvedges[eid].score, reverse=True)
    kept: Set[int] = set()
    kept_out: Dict[int, Set[int]] = defaultdict(set)
    for eid in ordered:
        e = kvedges[eid]
        if e.src not in nodes or e.dst not in nodes:
            continue
        if not _would_create_cycle(e.src, e.dst, kept_out):
            kept.add(eid)
            kept_out[e.src].add(e.dst)
        # else:
        #     print(f"[WARN] cycle detected: {e.src} -> {e.dst}")
    return kept


def prune_to_reachable(topic_nodes: List[int], kept_nodes: Set[int], kept_edges: Set[int], kvedges: Dict[int, KVEdge]) -> Tuple[Set[int], Set[int]]:
    out_adj: Dict[int, List[int]] = defaultdict(list)
    for eid in kept_edges:
        e = kvedges[eid]
        out_adj[e.src].append(eid)
    vis: Set[int] = set()
    stack = list(topic_nodes)
    while stack:
        u = stack.pop()
        if u in vis or u not in kept_nodes:
            continue
        vis.add(u)
        for eid in out_adj.get(u, []):
            v = kvedges[eid].dst
            if v not in vis:
                stack.append(v)
    vis_edges = {eid for eid in kept_edges if kvedges[eid].src in vis and kvedges[eid].dst in vis}
    return vis, vis_edges


def enforce_max_sinks(topic_nodes: List[int], max_sinks: Optional[int], kept_nodes: Set[int], kept_edges: Set[int], kvedges: Dict[int, KVEdge]) -> Tuple[Set[int], Set[int]]:
    if max_sinks is None or max_sinks <= 0:
        return prune_to_reachable(topic_nodes, kept_nodes, kept_edges, kvedges)

    kept_nodes, kept_edges = prune_to_reachable(topic_nodes, kept_nodes, kept_edges, kvedges)
    if not kept_edges:
        return kept_nodes, kept_edges

    while True:
        outdeg = {n: 0 for n in kept_nodes}
        for eid in kept_edges:
            outdeg[kvedges[eid].src] += 1
        sinks = [n for n in kept_nodes if outdeg.get(n, 0) == 0]
        if len(sinks) <= max_sinks:
            return kept_nodes, kept_edges
        # remove the weakest-supported sink
        sink_in_sum = {}
        for s in sinks:
            in_edges = [eid for eid in kept_edges if kvedges[eid].dst == s]
            sink_in_sum[s] = max(0.0, sum(kvedges[eid].score for eid in in_edges))
        bad_sink = min(sinks, key=lambda s: sink_in_sum.get(s, 0.0))
        in_edges = [eid for eid in kept_edges if kvedges[eid].dst == bad_sink]
        if not in_edges:
            kept_nodes.remove(bad_sink)
        else:
            worst = min(in_edges, key=lambda eid: kvedges[eid].score)
            kept_edges.remove(worst)
        kept_nodes, kept_edges = prune_to_reachable(topic_nodes, kept_nodes, kept_edges, kvedges)
        if not kept_edges:
            return kept_nodes, kept_edges


def export_kv_nodes_and_adj(kept_kvedges: Set[int], kvedges: Dict[int, KVEdge], keep_score: bool) -> Tuple[List[Dict[str, Any]], List[List[int]]]:
    kept_list = sorted(list(kept_kvedges))
    idx_map = {eid: i for i, eid in enumerate(kept_list)}
    kv_nodes: List[Dict[str, Any]] = []
    for eid in kept_list:
        e = kvedges[eid]
        obj = {
            'key': e.key,
            'value': e.value,
            'src_entity': e.src_name,
            'dst_entity': e.dst_name,
            'title': e.title,
            'triple_type': e.triple_type,
            'relation': e.relation,
            'kv_idx': e.kv_idx,
        }
        if keep_score:
            obj['edge_score'] = float(e.edge_score)
            obj['score'] = float(e.score)
        kv_nodes.append(obj)
    n = len(kv_nodes)
    adj = [[0] * n for _ in range(n)]
    for eid_i in kept_list:
        i = idx_map[eid_i]
        dst_i = kvedges[eid_i].dst
        for eid_j in kept_list:
            j = idx_map[eid_j]
            if i != j and kvedges[eid_j].src == dst_i:
                adj[i][j] = 1
    return kv_nodes, adj


def compute_sink_relevance_stats(
    kept_edges: Set[int],
    kvedges: Dict[int, KVEdge],
    feat_dict: Dict[int, Dict[str, Any]],
    answer: str,
) -> Dict[str, Any]:
    """
    On the final KV-node DAG:
    - sink relevance score = sum(sim(question, key)) over all KV nodes that can reach the sink
    - answer sink rank = best rank among terminal KV nodes whose value matches the gold answer
    """
    if not kept_edges:
        return {
            'num_sinks': 0,
            'sink_scores': [],
            'answer_sink_ranks': [],
            'best_answer_rank': None,
        }

    kept_list = sorted(kept_edges)
    kv_out: Dict[int, List[int]] = defaultdict(list)
    kv_rev: Dict[int, List[int]] = defaultdict(list)
    edge_key_scores: Dict[int, float] = {}
    entity_outdeg: Dict[int, int] = defaultdict(int)

    for eid in kept_list:
        entity_outdeg[kvedges[eid].src] += 1
        edge_key_scores[eid] = float(feat_dict[eid]['scalar'][3])

    for eid_i in kept_list:
        mid = kvedges[eid_i].dst
        for eid_j in kept_list:
            if eid_i == eid_j:
                continue
            if kvedges[eid_j].src == mid:
                kv_out[eid_i].append(eid_j)
                kv_rev[eid_j].append(eid_i)

    sink_eids = [eid for eid in kept_list if entity_outdeg.get(kvedges[eid].dst, 0) == 0]
    sink_stats: List[Dict[str, Any]] = []
    answer_sink_ranks: List[int] = []

    for sink_eid in sink_eids:
        vis: Set[int] = set()
        stack = [sink_eid]
        while stack:
            cur = stack.pop()
            if cur in vis:
                continue
            vis.add(cur)
            for prev_eid in kv_rev.get(cur, []):
                if prev_eid not in vis:
                    stack.append(prev_eid)

        sink_stats.append({
            'eid': sink_eid,
            'score': float(sum(edge_key_scores[eid] for eid in vis)),
            'is_answer_sink': value_matches_answer(kvedges[sink_eid].value, answer),
            'has_inbound': len(kv_rev.get(sink_eid, [])) > 0,
        })

    sink_stats.sort(key=lambda x: (-x['score'], x['eid']))

    for rank_idx, item in enumerate(sink_stats, start=1):
        if item['is_answer_sink']:
            answer_sink_ranks.append(rank_idx)

    return {
        'num_sinks': len(sink_stats),
        'sink_scores': sink_stats,
        'answer_sink_ranks': answer_sink_ranks,
        'best_answer_rank': min(answer_sink_ranks) if answer_sink_ranks else None,
    }


def update_sink_rank_buckets(
    rank_buckets: Dict[int, Dict[str, Any]],
    num_sinks: int,
    best_answer_rank: Optional[int],
) -> None:
    if num_sinks <= 0:
        return
    bucket = rank_buckets.setdefault(
        num_sinks,
        {
            'all_samples': 0,
            'answer_sink_samples': 0,
            'topk_hits': [0] * num_sinks,
        },
    )
    bucket['all_samples'] += 1
    if best_answer_rank is None:
        return
    bucket['answer_sink_samples'] += 1
    for k in range(best_answer_rank, num_sinks + 1):
        bucket['topk_hits'][k - 1] += 1


def merge_sink_rank_buckets(rank_buckets: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    if not rank_buckets:
        return {
            'all_samples': 0,
            'answer_sink_samples': 0,
            'topk_hits': [],
            'topk_denoms': [],
            'topk_probs': [],
        }

    max_sinks = max(rank_buckets.keys())
    merged = {
        'all_samples': 0,
        'answer_sink_samples': 0,
        'topk_hits': [0] * max_sinks,
        'topk_denoms': [0] * max_sinks,
        'topk_probs': [0.0] * max_sinks,
    }

    for num_sinks, bucket in rank_buckets.items():
        merged['all_samples'] += bucket['all_samples']
        merged['answer_sink_samples'] += bucket['answer_sink_samples']
        for k in range(1, num_sinks + 1):
            merged['topk_denoms'][k - 1] += bucket['answer_sink_samples']
            merged['topk_hits'][k - 1] += bucket['topk_hits'][k - 1]

    for k in range(max_sinks):
        denom = merged['topk_denoms'][k]
        merged['topk_probs'][k] = (merged['topk_hits'][k] / denom) if denom > 0 else 0.0

    return merged


def enforce_max_terminal_kv_nodes(
    topic_nodes: List[int],
    max_sinks: Optional[int],
    answer: str,
    kept_nodes: Set[int],
    kept_edges: Set[int],
    kvedges: Dict[int, KVEdge],
) -> Tuple[Set[int], Set[int]]:
    """
    Enforce the sink budget on the exported KV-node DAG.

    `max_sinks` is already applied on entity sinks above, but the final output uses
    KV edges as nodes. If a sink entity has multiple incoming KV edges, the exported
    DAG will still contain multiple terminal KV nodes. This pass trims those terminal
    KV nodes directly so the serialized `dag.adj` respects the sink budget.
    """
    if max_sinks is None or max_sinks <= 0:
        return kept_nodes, kept_edges

    kept_nodes, kept_edges = prune_to_reachable(topic_nodes, kept_nodes, kept_edges, kvedges)
    if not kept_edges:
        return kept_nodes, kept_edges

    while True:
        entity_outdeg = {n: 0 for n in kept_nodes}
        for eid in kept_edges:
            entity_outdeg[kvedges[eid].src] += 1

        terminal_kv_edges = [eid for eid in kept_edges if entity_outdeg.get(kvedges[eid].dst, 0) == 0]
        if len(terminal_kv_edges) <= max_sinks:
            return kept_nodes, kept_edges

        answer_terminal_edges = [eid for eid in terminal_kv_edges if value_matches_answer(kvedges[eid].value, answer)]
        removable_terminal_edges = [eid for eid in terminal_kv_edges if eid not in answer_terminal_edges]

        # Prefer keeping answer-matching terminal edges, but enforce the hard cap if needed.
        if removable_terminal_edges:
            bad_eid = min(removable_terminal_edges, key=lambda eid: kvedges[eid].score)
        else:
            bad_eid = min(terminal_kv_edges, key=lambda eid: kvedges[eid].score)

        kept_edges.remove(bad_eid)
        kept_nodes, kept_edges = prune_to_reachable(topic_nodes, kept_nodes, kept_edges, kvedges)
        if not kept_edges:
            return kept_nodes, kept_edges


def score_edges_with_model_joint(
    model: JointScorer,
    device: torch.device,
    feat_dict: Dict[int, Dict[str, Any]],
    infer_batch_size: int,
    task: str = 'edge',
) -> Dict[int, float]:
    """联合模型的 Edge 评分"""
    if not feat_dict:
        return {}

    eids = sorted(feat_dict.keys())
    X = np.stack([feat_dict[eid]['vector'] for eid in eids]).astype(np.float32)
    forward_fn = model.forward_edge if task == 'edge' else model.forward_node
    scores = _batched_sigmoid_scores(model, device, X, infer_batch_size, forward_fn=forward_fn)
    return {eid: float(scores[i]) for i, eid in enumerate(eids)}


def score_nodes_with_model_joint(
    model: JointScorer,
    device: torch.device,
    node_feat_dict: Dict[int, Dict[str, Any]],
    infer_batch_size: int,
    task: str = 'node',
) -> Dict[int, float]:
    """联合模型的 Node 评分"""
    if not node_feat_dict:
        return {}

    nids = sorted(node_feat_dict.keys())
    X = np.stack([node_feat_dict[nid]['vector'] for nid in nids]).astype(np.float32)
    forward_fn = model.forward_edge if task == 'edge' else model.forward_node
    scores = _batched_sigmoid_scores(model, device, X, infer_batch_size, forward_fn=forward_fn)
    return {nid: float(scores[i]) for i, nid in enumerate(nids)}
# ============================================================
# 8) Inference pipeline
# ============================================================

def answer_terminalization(
    topic_nodes: List[int],
    max_sinks: Optional[int],
    answer: str,
    kept_nodes: Set[int],
    kept_edges: Set[int],
    kvedges: Dict[int, KVEdge],
) -> Tuple[Set[int], Set[int]]:
    """
    Post-process the final DAG so that answer-matching nodes are pushed to sinks.

    Minimal-change heuristic:
    1) find nodes reached by edges whose value matches the gold answer
    2) remove all outgoing edges from those answer nodes
    3) prune unreachable nodes/edges
    4) if sink budget is exceeded, preferentially trim NON-answer sinks first

    Note:
    - This is an evaluation-time post-process because it uses the gold answer.
    - It is meant to estimate how much answer recall gap comes purely from
      missing terminalization.
    """
    if not answer or not kept_edges:
        return kept_nodes, kept_edges

    kept_nodes, kept_edges = prune_to_reachable(topic_nodes, kept_nodes, kept_edges, kvedges)
    if not kept_edges:
        return kept_nodes, kept_edges

    # 1) find answer nodes already present in the kept subgraph
    answer_nodes: Set[int] = set()
    for eid in kept_edges:
        e = kvedges[eid]
        if value_matches_answer(e.value, answer):
            answer_nodes.add(e.dst)

    if not answer_nodes:
        return kept_nodes, kept_edges

    # 2) delete all outgoing edges from answer nodes, forcing them toward sink status
    new_kept_edges: Set[int] = set()
    removed_any = False
    for eid in kept_edges:
        e = kvedges[eid]
        if e.src in answer_nodes:
            removed_any = True
            continue
        new_kept_edges.add(eid)

    if not removed_any:
        return kept_nodes, kept_edges

    kept_edges = new_kept_edges
    kept_nodes, kept_edges = prune_to_reachable(topic_nodes, kept_nodes, kept_edges, kvedges)
    if not kept_edges:
        return kept_nodes, kept_edges

    # 3) if max_sinks is enforced, do a protected sink trim:
    #    never delete answer sinks first; trim non-answer sinks first.
    if max_sinks is not None and max_sinks > 0:
        while True:
            outdeg = {n: 0 for n in kept_nodes}
            for eid in kept_edges:
                outdeg[kvedges[eid].src] += 1

            sinks = [n for n in kept_nodes if outdeg.get(n, 0) == 0]
            if len(sinks) <= max_sinks:
                break

            protected_sinks = [s for s in sinks if s in answer_nodes]
            removable_sinks = [s for s in sinks if s not in answer_nodes]

            # if all sinks are protected, stop here rather than destroying answer terminalization
            if not removable_sinks:
                break

            sink_in_sum = {}
            for s in removable_sinks:
                in_edges = [eid for eid in kept_edges if kvedges[eid].dst == s]
                sink_in_sum[s] = max(0.0, sum(kvedges[eid].score for eid in in_edges))

            bad_sink = min(removable_sinks, key=lambda s: sink_in_sum.get(s, 0.0))
            in_edges = [eid for eid in kept_edges if kvedges[eid].dst == bad_sink]

            if not in_edges:
                kept_nodes.remove(bad_sink)
            else:
                worst = min(in_edges, key=lambda eid: kvedges[eid].score)
                kept_edges.remove(worst)

            kept_nodes, kept_edges = prune_to_reachable(topic_nodes, kept_nodes, kept_edges, kvedges)
            if not kept_edges:
                return kept_nodes, kept_edges

    return kept_nodes, kept_edges



def _init_generation_state() -> Dict[str, Any]:
    return {
        'out_samples': [],
        'answer_recall': 0,
        'graph_recall': 0,
        'none_sink_recall': 0,
        'answer_supported_recall': 0,
        'sink_rank_buckets': {},
        'answer_unsupported_sample_ids': [],
        'answer_supported_sample_ids': [],
    }


def _init_latency_stats() -> Dict[str, Any]:
    return {
        'graph_build_s': 0.0,
        'question_encode_s': 0.0,
        'doc_text_encode_s': 0.0,
        'feature_build_s': 0.0,
        'model_scoring_s': 0.0,
        'dag_postprocess_s': 0.0,
        'export_s': 0.0,
        'samples_profiled': 0,
    }


def _report_generation_summary(samples: List[Dict[str, Any]], state: Dict[str, Any]) -> None:
    if len(samples) == 0:
        return

    print(f'Answer recall: {state["answer_recall"] / len(samples):.4f}')
    print(f'Graph  recall: {state["graph_recall"] / len(samples):.4f}')
    print(f'None-sink recall: {state["none_sink_recall"] / len(samples):.4f}')
    print(f'Answer + supported sink ratio: {state["answer_supported_recall"] / len(samples):.4f}')
    if state['sink_rank_buckets']:
        merged_sink_rank_stats = merge_sink_rank_buckets(state['sink_rank_buckets'])
        topk_probs = [
            f'top-{k}={prob:.4f}'
            for k, prob in enumerate(merged_sink_rank_stats['topk_probs'], start=1)
        ]
        print('Sink relevance rank stats (merged across final sink counts):')
        print(
            f'  all_samples={merged_sink_rank_stats["all_samples"]} '
            f'answer_sink_samples={merged_sink_rank_stats["answer_sink_samples"]} '
            + ' '.join(topk_probs)
        )


def _report_latency_summary(latency_stats: Dict[str, Any]) -> None:
    n = latency_stats['samples_profiled']
    if n <= 0:
        return

    online_total = (
        latency_stats['question_encode_s']
        + latency_stats['feature_build_s']
        + latency_stats['model_scoring_s']
        + latency_stats['dag_postprocess_s']
        + latency_stats['export_s']
    )
    offline_total = latency_stats['graph_build_s'] + latency_stats['doc_text_encode_s']
    print('Online latency breakdown:')
    print(
        f'  samples={n} '
        f'question_encode={latency_stats["question_encode_s"]:.4f}s '
        f'feature_build={latency_stats["feature_build_s"]:.4f}s '
        f'model_scoring={latency_stats["model_scoring_s"]:.4f}s '
        f'dag_postprocess={latency_stats["dag_postprocess_s"]:.4f}s '
        f'export={latency_stats["export_s"]:.4f}s '
        f'online_total={online_total:.4f}s'
    )
    print(
        f'  avg_per_sample='
        f'{(online_total / n):.6f}s '
        f'question_encode={(latency_stats["question_encode_s"] / n):.6f}s '
        f'feature_build={(latency_stats["feature_build_s"] / n):.6f}s '
        f'model_scoring={(latency_stats["model_scoring_s"] / n):.6f}s '
        f'dag_postprocess={(latency_stats["dag_postprocess_s"] / n):.6f}s '
        f'export={(latency_stats["export_s"] / n):.6f}s'
    )
    print('Offline/auxiliary breakdown:')
    print(
        f'  graph_build={latency_stats["graph_build_s"]:.4f}s '
        f'doc_text_encode={latency_stats["doc_text_encode_s"]:.4f}s '
        f'offline_total={offline_total:.4f}s'
    )
    print(
        f'  avg_per_sample='
        f'{(offline_total / n):.6f}s '
        f'graph_build={(latency_stats["graph_build_s"] / n):.6f}s '
        f'doc_text_encode={(latency_stats["doc_text_encode_s"] / n):.6f}s'
    )


def _process_sample_dag(
    args,
    sample: Dict[str, Any],
    sample_idx: int,
    answer: str,
    topic_nodes: List[int],
    kvedges: Dict[int, KVEdge],
    out_adj: Dict[int, List[int]],
    in_adj: Dict[int, List[int]],
    feat_dict: Dict[int, Dict[str, Any]],
    edge_scores: Dict[int, float],
    node_end_scores: Dict[int, float],
    state: Dict[str, Any],
    verbose: bool = False,
) -> Tuple[float, float]:
    if verbose:
        print(f"edge_scores: {edge_scores}")
        print(f"node_end_scores: {node_end_scores}")

    dag_t0 = time.perf_counter()
    apply_two_stage_joint_scores(
        kvedges=kvedges,
        edge_scores=edge_scores,
        node_end_scores=node_end_scores,
        alpha=args.end_alpha,
        beta=args.end_beta,
        gamma=args.end_gamma,
    )
    if verbose:
        print_kvedge_graph(kvedges, out_adj, in_adj)

    if any(value_matches_answer(e.value, answer) for e in kvedges.values()):
        state['graph_recall'] += 1

    kept_nodes, kept_edges = select_subgraph_edges(
        topic_nodes=topic_nodes,
        kvedges=kvedges,
        out_adj=out_adj,
        max_edges=args.max_edges,
        max_nodes=args.max_nodes,
        per_src_cap=args.per_src_cap,
        expansion_hops=args.expansion_hops,
        seed_edge_topk=args.seed_edge_topk,
    )

    if verbose:
        print(f"kept nodes after subgraph selection: {kept_nodes}")
        print(f"kept edges after subgraph selection: {kept_edges}")

    kept_edges = break_cycles_to_dag(kept_nodes, kept_edges, kvedges)
    if verbose:
        print(f"kept nodes after cycle breaking: {kept_nodes}")
        print(f"kept edges after cycle breaking: {kept_edges}")

    if args.answer_aware:
        kept_nodes, kept_edges = answer_terminalization(
            topic_nodes=topic_nodes,
            max_sinks=args.max_sinks,
            answer=answer,
            kept_nodes=kept_nodes,
            kept_edges=kept_edges,
            kvedges=kvedges,
        )

    kept_nodes, kept_edges = enforce_max_sinks(topic_nodes, args.max_sinks, kept_nodes, kept_edges, kvedges)
    kept_nodes, kept_edges = enforce_max_terminal_kv_nodes(
        topic_nodes=topic_nodes,
        max_sinks=args.max_sinks,
        answer=answer,
        kept_nodes=kept_nodes,
        kept_edges=kept_edges,
        kvedges=kvedges,
    )
    if verbose:
        print(f"kept nodes after sink enforcement: {kept_nodes}")
        print(f"kept edges after sink enforcement: {kept_edges}")
    kept_nodes, kept_edges = reverse_beam_expand_from_sink_edges(
        selected_nodes=kept_nodes,
        selected_edges=kept_edges,
        kvedges=kvedges,
        in_adj=in_adj,
        max_edges=args.max_edges,
        max_nodes=args.max_nodes,
        sink_edge_topk=args.reverse_sink_edge_topk,
        reverse_hops=args.reverse_sink_hops,
        beam_width=args.reverse_sink_beam_width,
    )

    dag_elapsed = time.perf_counter() - dag_t0
    if not kept_edges:
        sample['dag'] = {'kv_nodes': [], 'adj': [], 'meta': {'reason': 'empty_after_prune'}}
        state['out_samples'].append(sample)
        return dag_elapsed, 0.0

    outdeg = defaultdict(int)
    for eid in kept_edges:
        outdeg[kvedges[eid].src] += 1
    sinks = [n for n in kept_nodes if outdeg.get(n, 0) == 0]

    sink_hit = False
    for sink in sinks:
        for eid in kept_edges:
            if kvedges[eid].dst == sink and value_matches_answer(kvedges[eid].value, answer):
                sink_hit = True
                break
        if sink_hit:
            break
    if sink_hit:
        state['answer_recall'] += 1
    else:
        if args.answer_aware:
            return dag_elapsed, 0.0

    sink_rank_stats = compute_sink_relevance_stats(
        kept_edges=kept_edges,
        kvedges=kvedges,
        feat_dict=feat_dict,
        answer=answer,
    )
    sample_id = sample.get('id', sample.get('_id', f'sample_{sample_idx}'))
    if any(item['is_answer_sink'] and item['has_inbound'] for item in sink_rank_stats['sink_scores']):
        state['answer_supported_recall'] += 1
        state['answer_supported_sample_ids'].append(sample_id)
    else:
        if sink_hit:
            state['answer_unsupported_sample_ids'].append(sample_id)

    if any(value_matches_answer(kvedges[eid].value, answer) for eid in kept_edges):
        state['none_sink_recall'] += 1

    update_sink_rank_buckets(
        rank_buckets=state['sink_rank_buckets'],
        num_sinks=sink_rank_stats['num_sinks'],
        best_answer_rank=sink_rank_stats['best_answer_rank'],
    )

    export_t0 = time.perf_counter()
    kv_nodes, adj = export_kv_nodes_and_adj(kept_edges, kvedges, keep_score=args.keep_score)
    if verbose:
        print(f"Exported KV nodes: {kv_nodes}")
        print(f"Exported adjacency matrix: {adj}")
    goal_ids: List[int] = []
    if answer:
        for j, kv in enumerate(kv_nodes):
            if value_matches_answer(kv.get('value', ''), answer):
                goal_ids.append(j)

    sample['dag'] = {
        'kv_nodes': kv_nodes,
        'adj': adj,
        'meta': {
            'num_entity_nodes': int(len(kept_nodes)),
            'num_kv_edges': int(len(kept_edges)),
            'num_kv_nodes': int(len(kv_nodes)),
            'goal_ids': goal_ids,
            'topic_entity_ids': [int(x) for x in topic_nodes],
            'scorer': 'trainable_subgraphrag_mlp_two_stage_node_ending',
        },
    }
    state['out_samples'].append(sample)
    return dag_elapsed, (time.perf_counter() - export_t0)


def create_dag_with_model_batched(
    args,
    samples: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    edge_model: MLPScorer,
    node_model: NodeEndScorer,
    ckpt: Dict[str, Any],
    device: torch.device,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    if args.limit is not None and args.limit > 0:
        samples = samples[:args.limit]

    if verbose:
        target_sample = 0
        for s in samples:
            if s.get('answer') == "Rome":
                target_sample = s
                break
        samples = [target_sample]
        print(samples)

    state = _init_generation_state()
    is_joint = ckpt.get('is_joint', False)
    infer_batch_size = max(1, int(args.infer_batch_size))
    for batch_start in tqdm(
        range(0, len(samples), infer_batch_size),
        desc='Create DAG (two-stage node-ending aware)',
    ):
        sample_batch = samples[batch_start: batch_start + infer_batch_size]
        prepared: List[Dict[str, Any]] = []
        question_texts: List[str] = []
        doc_texts: List[str] = []

        for sample in sample_batch:
            question = norm_text(sample.get('question', ''))
            answer = norm_text(sample.get('answer', ''))
            if verbose:
                print(f"\nQuestion: {question}")
                print(f"Answer: {answer}")

            node_names, kvedges, out_adj, in_adj = build_kvedge_graph(sample, supporting_only=args.supporting_only)
            if not kvedges:
                sample['dag'] = {'kv_nodes': [], 'adj': [], 'meta': {'reason': 'no_kv_edges'}}
                state['out_samples'].append(sample)
                continue

            question_texts.append(question)
            doc_texts.extend(node_names)
            for e in kvedges.values():
                doc_texts.append(norm_text(e.relation or e.key))
                doc_texts.append(norm_text(e.key))
                doc_texts.append(norm_text(e.value))
            prepared.append({
                'sample': sample,
                'question': question,
                'answer': answer,
                'node_names': node_names,
                'kvedges': kvedges,
                'out_adj': out_adj,
                'in_adj': in_adj,
            })

        if not prepared:
            continue

        text_emb_cache = merge_text_embedding_caches(
            embedder=embedder,
            text_groups=[question_texts, doc_texts],
            batch_size=infer_batch_size,
        )

        for ctx in prepared:
            topic_nodes, feat_dict = build_edge_feature_dict(
                question=ctx['question'],
                node_names=ctx['node_names'],
                node_emb=None,
                kvedges=ctx['kvedges'],
                out_adj=ctx['out_adj'],
                in_adj=ctx['in_adj'],
                embedder=None,
                batch_size=infer_batch_size,
                topic_top_k=args.topic_top_k,
                dde_hops=args.dde_hops,
                mention_bonus=args.mention_bonus,
                text_emb_cache=text_emb_cache,
            )
            node_emb = np.stack([text_emb_cache[norm_text(x)] for x in ctx['node_names']]).astype(np.float32)
            node_feat_dict = build_node_feature_dict(
                question=ctx['question'],
                node_names=ctx['node_names'],
                node_emb=node_emb,
                kvedges=ctx['kvedges'],
                out_adj=ctx['out_adj'],
                in_adj=ctx['in_adj'],
                topic_nodes=topic_nodes,
                feat_dict=feat_dict,
            )
            ctx['topic_nodes'] = topic_nodes
            ctx['feat_dict'] = feat_dict
            ctx['node_feat_dict'] = node_feat_dict

        edge_scores_by_sample: List[Dict[int, float]] = [dict() for _ in prepared]
        node_scores_by_sample: List[Dict[int, float]] = [dict() for _ in prepared]

        edge_blocks: List[np.ndarray] = []
        edge_meta: List[Tuple[int, List[int], int, int]] = []
        cursor = 0
        for i, ctx in enumerate(prepared):
            eids = sorted(ctx['feat_dict'].keys())
            if not eids:
                continue
            x = np.stack([ctx['feat_dict'][eid]['vector'] for eid in eids]).astype(np.float32)
            edge_blocks.append(x)
            edge_meta.append((i, eids, cursor, x.shape[0]))
            cursor += x.shape[0]
        if edge_blocks:
            edge_X = np.concatenate(edge_blocks, axis=0)
            edge_forward = edge_model.forward_edge if is_joint else None
            edge_scores_all = _batched_sigmoid_scores(
                edge_model, device, edge_X, infer_batch_size, forward_fn=edge_forward
            )
            for i, eids, start, length in edge_meta:
                part = edge_scores_all[start:start + length]
                edge_scores_by_sample[i] = {eid: float(part[j]) for j, eid in enumerate(eids)}

        node_blocks: List[np.ndarray] = []
        node_meta: List[Tuple[int, List[int], int, int]] = []
        cursor = 0
        for i, ctx in enumerate(prepared):
            nids = sorted(ctx['node_feat_dict'].keys())
            if not nids:
                continue
            x = np.stack([ctx['node_feat_dict'][nid]['vector'] for nid in nids]).astype(np.float32)
            node_blocks.append(x)
            node_meta.append((i, nids, cursor, x.shape[0]))
            cursor += x.shape[0]
        if node_blocks:
            node_X = np.concatenate(node_blocks, axis=0)
            node_forward = node_model.forward_node if is_joint else None
            node_scores_all = _batched_sigmoid_scores(
                node_model, device, node_X, infer_batch_size, forward_fn=node_forward
            )
            for i, nids, start, length in node_meta:
                part = node_scores_all[start:start + length]
                node_scores_by_sample[i] = {nid: float(part[j]) for j, nid in enumerate(nids)}

        for i, ctx in enumerate(prepared):
            _process_sample_dag(
                args=args,
                sample=ctx['sample'],
                sample_idx=batch_start + i,
                answer=ctx['answer'],
                topic_nodes=ctx['topic_nodes'],
                kvedges=ctx['kvedges'],
                out_adj=ctx['out_adj'],
                in_adj=ctx['in_adj'],
                feat_dict=ctx['feat_dict'],
                edge_scores=edge_scores_by_sample[i],
                node_end_scores=node_scores_by_sample[i],
                state=state,
                verbose=verbose,
            )

    _report_generation_summary(samples, state)
    return state['out_samples'], state['answer_unsupported_sample_ids'], state['answer_supported_sample_ids']


def create_dag_with_model_profiled(
    args,
    samples: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    edge_model: MLPScorer,
    node_model: NodeEndScorer,
    ckpt: Dict[str, Any],
    device: torch.device,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    if args.limit is not None and args.limit > 0:
        samples = samples[:args.limit]

    if verbose:
        target_sample = 0
        for s in samples:
            if s.get('answer') == "Rome":
                target_sample = s
                break
        samples = [target_sample]
        print(samples)

    state = _init_generation_state()
    latency_stats = _init_latency_stats()
    is_joint = ckpt.get('is_joint', False)

    for sample_idx, sample in enumerate(tqdm(samples, desc='Create DAG (strict per-sample profiling)')):
        question = norm_text(sample.get('question', ''))
        answer = norm_text(sample.get('answer', ''))
        if verbose:
            print(f"\nQuestion: {question}")
            print(f"Answer: {answer}")

        graph_build_t0 = time.perf_counter()
        node_names, kvedges, out_adj, in_adj = build_kvedge_graph(sample, supporting_only=args.supporting_only)
        latency_stats['graph_build_s'] += time.perf_counter() - graph_build_t0

        latency_stats['samples_profiled'] += 1
        if not kvedges:
            sample['dag'] = {'kv_nodes': [], 'adj': [], 'meta': {'reason': 'no_kv_edges'}}
            state['out_samples'].append(sample)
            continue

        q_enc_t0 = time.perf_counter()
        question_emb_cache = merge_text_embedding_caches(
            embedder=embedder,
            text_groups=[[question]],
            batch_size=1,
        )
        latency_stats['question_encode_s'] += time.perf_counter() - q_enc_t0

        doc_texts: List[str] = list(node_names)
        for e in kvedges.values():
            doc_texts.append(norm_text(e.relation or e.key))
            doc_texts.append(norm_text(e.key))
            doc_texts.append(norm_text(e.value))
        doc_batch_size = max(1, len(doc_texts))
        doc_enc_t0 = time.perf_counter()
        doc_emb_cache = merge_text_embedding_caches(
            embedder=embedder,
            text_groups=[doc_texts],
            batch_size=doc_batch_size,
        )
        latency_stats['doc_text_encode_s'] += time.perf_counter() - doc_enc_t0
        text_emb_cache = {**doc_emb_cache, **question_emb_cache}

        feature_t0 = time.perf_counter()
        topic_nodes, feat_dict = build_edge_feature_dict(
            question=question,
            node_names=node_names,
            node_emb=None,
            kvedges=kvedges,
            out_adj=out_adj,
            in_adj=in_adj,
            embedder=None,
            batch_size=1,
            topic_top_k=args.topic_top_k,
            dde_hops=args.dde_hops,
            mention_bonus=args.mention_bonus,
            text_emb_cache=text_emb_cache,
        )
        node_emb = np.stack([text_emb_cache[norm_text(x)] for x in node_names]).astype(np.float32)
        node_feat_dict = build_node_feature_dict(
            question=question,
            node_names=node_names,
            node_emb=node_emb,
            kvedges=kvedges,
            out_adj=out_adj,
            in_adj=in_adj,
            topic_nodes=topic_nodes,
            feat_dict=feat_dict,
        )
        latency_stats['feature_build_s'] += time.perf_counter() - feature_t0

        scoring_t0 = time.perf_counter()
        edge_scores: Dict[int, float] = {}
        eids = sorted(feat_dict.keys())
        if eids:
            edge_X = np.stack([feat_dict[eid]['vector'] for eid in eids]).astype(np.float32)
            edge_forward = edge_model.forward_edge if is_joint else None
            edge_scores_all = _batched_sigmoid_scores(
                edge_model,
                device,
                edge_X,
                max(1, edge_X.shape[0]),
                forward_fn=edge_forward,
            )
            edge_scores = {eid: float(edge_scores_all[j]) for j, eid in enumerate(eids)}

        node_end_scores: Dict[int, float] = {}
        nids = sorted(node_feat_dict.keys())
        if nids:
            node_X = np.stack([node_feat_dict[nid]['vector'] for nid in nids]).astype(np.float32)
            node_forward = node_model.forward_node if is_joint else None
            node_scores_all = _batched_sigmoid_scores(
                node_model,
                device,
                node_X,
                max(1, node_X.shape[0]),
                forward_fn=node_forward,
            )
            node_end_scores = {nid: float(node_scores_all[j]) for j, nid in enumerate(nids)}
        latency_stats['model_scoring_s'] += time.perf_counter() - scoring_t0

        dag_elapsed, export_elapsed = _process_sample_dag(
            args=args,
            sample=sample,
            sample_idx=sample_idx,
            answer=answer,
            topic_nodes=topic_nodes,
            kvedges=kvedges,
            out_adj=out_adj,
            in_adj=in_adj,
            feat_dict=feat_dict,
            edge_scores=edge_scores,
            node_end_scores=node_end_scores,
            state=state,
            verbose=verbose,
        )
        latency_stats['dag_postprocess_s'] += dag_elapsed
        latency_stats['export_s'] += export_elapsed

    _report_generation_summary(samples, state)
    _report_latency_summary(latency_stats)
    return state['out_samples'], state['answer_unsupported_sample_ids'], state['answer_supported_sample_ids']


def create_dag_with_model(
    args,
    samples: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    edge_model: MLPScorer,
    node_model: NodeEndScorer,
    ckpt: Dict[str, Any],
    device: torch.device,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    if args.profile_online_latency:
        return create_dag_with_model_profiled(
            args, samples, embedder, edge_model, node_model, ckpt, device, verbose=verbose
        )
    return create_dag_with_model_batched(
        args, samples, embedder, edge_model, node_model, ckpt, device, verbose=verbose
    )


def drop_empty_kv_samples(samples: List[Dict[str, Any]], answerable_only: bool) -> List[Dict[str, Any]]:
    if not answerable_only:
        return samples
    kept: List[Dict[str, Any]] = []
    for s in samples:
        dag = s.get('dag', {}) if isinstance(s, dict) else {}
        kv_nodes = dag.get('kv_nodes', []) if isinstance(dag, dict) else []
        if isinstance(kv_nodes, list) and len(kv_nodes) > 0:
            kept.append(s)
    return kept


def create_dag_baseline_all_triples(
    args,
    samples: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    out_samples: List[Dict[str, Any]] = []
    answer_unsupported_sample_ids: List[int] = []
    answer_supported_sample_ids: List[int] = []

    for sample in tqdm(samples, desc='Baseline DAG'):
        node_names, kvedges, out_adj, in_adj = build_kvedge_graph(
            sample,
            supporting_only=args.supporting_only,
        )

        kept_edges = set(kvedges.keys())
        kept_list = sorted(kept_edges)
        kv_nodes: List[Dict[str, Any]] = []
        answer = norm_text(sample.get('answer', ''))
        goal_ids: List[int] = []

        for idx, eid in enumerate(kept_list):
            e = kvedges[eid]
            obj = {
                'key': e.key,
                'value': e.value,
                'src_entity': e.src_name,
                'dst_entity': e.dst_name,
                'title': e.title,
                'triple_type': e.triple_type,
                'relation': e.relation,
                'kv_idx': e.kv_idx,
            }
            if args.keep_score:
                obj['edge_score'] = 0.0
                obj['score'] = 0.0
            kv_nodes.append(obj)
            if answer and value_matches_answer(e.value, answer):
                goal_ids.append(idx)

        n = len(kv_nodes)
        adj = [[0] * n for _ in range(n)]

        sample['dag'] = {
            'kv_nodes': kv_nodes,
            'adj': adj,
            'meta': {
                'num_entity_nodes': int(len(node_names)),
                'num_kv_edges': int(len(kept_edges)),
                'num_kv_nodes': int(len(kv_nodes)),
                'goal_ids': goal_ids,
                'topic_entity_ids': [],
                'scorer': 'baseline_all_triples_zero_adj',
            },
        }
        out_samples.append(sample)

        sid = sample.get('id', sample.get('_id', None))
        if sid is not None:
            if goal_ids:
                answer_supported_sample_ids.append(sid)
            else:
                answer_unsupported_sample_ids.append(sid)

    return out_samples, answer_unsupported_sample_ids, answer_supported_sample_ids

# ============================================================
# 9) CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'infer', 'baseline'], default='infer')
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', default='')
    ap.add_argument('--model_ckpt', default='subgraphrag_mlp.pt')
    ap.add_argument('--st_model', default='sentence-transformers/all-MiniLM-L6-v2')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--infer_batch_size', type=int, default=4096)
    ap.add_argument('--keep_score', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--supporting_only', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--seed', type=int, default=42)

    # topic + structural features
    ap.add_argument('--topic_top_k', type=int, default=6)
    ap.add_argument('--dde_hops', type=int, default=3)
    ap.add_argument('--mention_bonus', type=float, default=0.20)

    # retrieval / dag size
    ap.add_argument('--seed_edge_topk', type=int, default=18)
    ap.add_argument('--expansion_hops', type=int, default=2)
    ap.add_argument('--per_src_cap', type=int, default=3)
    ap.add_argument('--reverse_sink_edge_topk', type=int, default=3)
    ap.add_argument('--reverse_sink_hops', type=int, default=2)
    ap.add_argument('--reverse_sink_beam_width', type=int, default=4)
    ap.add_argument('--max_nodes', type=int, default=30)
    ap.add_argument('--max_edges', type=int, default=40)
    ap.add_argument('--max_sinks', type=int, default=8)
    ap.add_argument('--answer_aware', action='store_true')

    # training
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--hidden_dim', type=int, default=512)
    ap.add_argument('--dropout', type=float, default=0.10)
    ap.add_argument('--train_batch_size', type=int, default=512)
    ap.add_argument('--num_workers', type=int, default=0,
                    help='Number of DataLoader workers for training and dev evaluation.')
    ap.add_argument('--prefetch_batches', type=int, default=2,
                    help='Prefetch factor for each DataLoader worker.')
    ap.add_argument('--disable_pin_memory', action='store_true',
                    help='Disable pinned host memory when training on GPU.')
    ap.add_argument('--train_log_interval', type=int, default=200,
                    help='Log once every N training batches.')
    ap.add_argument('--eval_log_interval', type=int, default=100,
                    help='Log once every N eval batches.')
    ap.add_argument('--dev_ratio', type=float, default=0.1)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--neg_pos_ratio', type=int, default=4)
    ap.add_argument('--min_negatives_per_sample', type=int, default=16)
    ap.add_argument(
        "--train_cache_path",
        type=str,
        default="",
        help="optional path to save/load cached training examples (X,Y,meta)"
    )
    ap.add_argument(
        "--rebuild_train_cache",
        action="store_true",
        help="force rebuild training cache even if cache file exists"
    )
    ap.add_argument('--verbose', action='store_true')

    # node-ending scorer
    ap.add_argument('--end_hidden_dim', type=int, default=256)
    ap.add_argument('--end_lr', type=float, default=1e-3)
    ap.add_argument('--end_threshold', type=float, default=0.5)
    ap.add_argument('--end_neg_pos_ratio', type=int, default=4)
    ap.add_argument('--end_min_negatives_per_sample', type=int, default=16)

    # joint score
    ap.add_argument('--end_alpha', type=float, default=0.60)
    ap.add_argument('--end_beta', type=float, default=0.35)
    ap.add_argument('--end_gamma', type=float, default=0.25)

    ap.add_argument('--joint_training', action='store_true', 
                help='Enable joint training with shared encoder')
    ap.add_argument('--joint_lambda', type=float, default=0.5,
                    help='Weight for node loss in joint training (lambda_node)')

    ap.add_argument('--answerable_only', action='store_true',
                    help='Only include answerable samples in training')
    ap.add_argument(
        '--keep_type',
        type=str,
        default='',
        help='Optional comma-separated sample types to keep, e.g. bridge or bridge,compositional.',
    )

    ap.add_argument('--dis_out_path', default=None)
    ap.add_argument(
        '--profile_online_latency',
        action='store_true',
        help='Print a latency breakdown for online DAG generation stages.',
    )

    args = ap.parse_args()

    print(args)
    samples = read_json_or_jsonl(args.input)
    if args.limit:
        samples = samples[:args.limit]

    if args.keep_type.strip():
        keep_types = {norm_match(x) for x in args.keep_type.split(',') if x.strip()}
        samples = [s for s in samples if norm_match(s.get('type', '')) in keep_types]

    print(f'Load {len(samples)} samples from {args.input}')

    if args.mode == 'train':
        embedder = SentenceTransformer(args.st_model)
        train_model(args, samples, embedder)
        return

    if args.mode == 'baseline':
        out, answer_unsupported_sample_ids, answer_supported_sample_ids = create_dag_baseline_all_triples(
            args,
            samples,
        )
        out = drop_empty_kv_samples(out, answerable_only=args.answerable_only)

        if args.dis_out_path and out:
            if "islet" in args.dis_out_path:
                evaluated_samples_ids = []
                for s in out:
                    sid = s.get('id', s.get('_id', None))
                    if sid is not None and sid not in answer_unsupported_sample_ids:
                        evaluated_samples_ids.append(sid)
            elif "supported" in args.dis_out_path:
                evaluated_samples_ids = answer_supported_sample_ids
            else:
                evaluated_samples_ids = answer_supported_sample_ids

            print(f"Saving {len(evaluated_samples_ids)} evaluated sample IDs to {args.dis_out_path}")
            with open(args.dis_out_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "evaluated_samples": evaluated_samples_ids,
                }, f, ensure_ascii=False, indent=2)

        if not args.output:
            raise ValueError('--output is required for --mode baseline')
        if args.output.endswith('.jsonl'):
            write_jsonl(args.output, out)
        elif args.output.endswith('.json'):
            write_json(args.output, out)
        else:
            raise ValueError(f'Unknown file format: {args.output}')
        print(f'[DONE] input={len(samples)} output={len(out)} saved_to={args.output}')
        return

    if not args.model_ckpt or not os.path.exists(args.model_ckpt):
        raise FileNotFoundError(f'For --mode infer, model checkpoint is required: {args.model_ckpt}')

    embedder = SentenceTransformer(args.st_model)
    edge_model, node_model, ckpt, device = load_model(args.model_ckpt, cpu=args.cpu)
    print(f'Loaded scorer from {args.model_ckpt}')
    out, answer_unsupported_sample_ids, answer_supported_sample_ids = create_dag_with_model(
            args,
            samples,
            embedder,
            edge_model,
            node_model,
            ckpt,
            device,
            verbose=args.verbose,
        )
    out = drop_empty_kv_samples(out, answerable_only=args.answerable_only)

    if args.dis_out_path and out:
        if "islet" in args.dis_out_path:
            evaluated_samples_ids=[]
            # 取out中样本的id，如果不在answer_unsupported_sample_ids中，则加入evaluated_samples_ids
            for s in out:
                sid = s.get('id', s.get('_id', None))
                if sid is not None and sid not in answer_unsupported_sample_ids:
                    evaluated_samples_ids.append(sid)
        elif "supported" in args.dis_out_path:
            evaluated_samples_ids=answer_supported_sample_ids

        print(f"Saving {len(evaluated_samples_ids)} evaluated sample IDs to {args.dis_out_path}")
        with open(args.dis_out_path, 'w', encoding='utf-8') as f:
            json.dump({
                "evaluated_samples": evaluated_samples_ids,
            }, f, ensure_ascii=False, indent=2)

    if not args.output:
        raise ValueError('--output is required for --mode infer')
    if args.output.endswith('.jsonl'):
        write_jsonl(args.output, out)
    elif args.output.endswith('.json'):
        write_json(args.output, out)
    else:
        raise ValueError(f'Unknown file format: {args.output}')
    print(f'[DONE] input={len(samples)} output={len(out)} saved_to={args.output}')


if __name__ == '__main__':
    main()
