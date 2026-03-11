import argparse
import json
import os
import random
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ============================================================
# 0) IO
# ============================================================
def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith('.jsonl'):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

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
    if supporting_only:
        sf = sample.get('supporting_facts', []) or []
        supporting_titles = {t for t, _ in sf if isinstance(t, str)}

    for para in ctx:
        title = norm_text(para.get('title', ''))
        if supporting_titles is not None and title not in supporting_titles:
            continue
        for tri in (para.get('triple_list', []) or []):
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

            direction = infer_kv_direction(s, o, key, value)
            if direction == 'backward':
                src_name, dst_name = o, s
            else:
                src_name, dst_name = s, o

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
    # if supporting_titles and norm_text(e.title) in supporting_titles:
    #     pos = True
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
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.xs = torch.from_numpy(xs.astype(np.float32))
        self.ys = torch.from_numpy(ys.astype(np.float32))

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx: int):
        return self.xs[idx], self.ys[idx]


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
    edge_xs: List[np.ndarray] = []
    edge_ys: List[int] = []
    node_xs: List[np.ndarray] = []
    node_ys: List[int] = []

    total_edges = 0
    pos_edges = 0
    total_nodes = 0
    pos_nodes_total = 0

    used_samples = samples[:args.limit] if args.limit else samples
    train_embed_batch_size = args.batch_size

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

    text_emb_cache = embed_texts_cached(
        embedder=embedder,
        texts=all_texts,
        batch_size=train_embed_batch_size,
    )

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
            edge_xs.append(feat_dict[eid]['vector'])
            edge_ys.append(1)
        for eid in kept_neg_ids:
            edge_xs.append(feat_dict[eid]['vector'])
            edge_ys.append(0)

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
            node_xs.append(node_feat_dict[nid]['vector'])
            node_ys.append(1)
        for nid in neg_node_ids:
            node_xs.append(node_feat_dict[nid]['vector'])
            node_ys.append(0)

    if not edge_xs:
        raise ValueError('No edge training examples constructed.')
    if not node_xs:
        raise ValueError('No node-ending training examples constructed.')

    X_edge = np.stack(edge_xs).astype(np.float32)
    Y_edge = np.array(edge_ys, dtype=np.float32)
    X_node = np.stack(node_xs).astype(np.float32)
    Y_node = np.array(node_ys, dtype=np.float32)

    meta = {
        'num_edge_examples': int(len(Y_edge)),
        'num_edge_pos': int(Y_edge.sum()),
        'num_edge_neg': int(len(Y_edge) - Y_edge.sum()),
        'num_node_examples': int(len(Y_node)),
        'num_node_pos': int(Y_node.sum()),
        'num_node_neg': int(len(Y_node) - Y_node.sum()),
        'raw_total_edges': int(total_edges),
        'raw_pos_edges': int(pos_edges),
        'raw_total_nodes': int(total_nodes),
        'raw_pos_nodes': int(pos_nodes_total),
        'edge_input_dim': int(X_edge.shape[1]),
        'node_input_dim': int(X_node.shape[1]),
        'num_unique_emb_texts': int(len(text_emb_cache)),
        'train_embed_batch_size': int(train_embed_batch_size),
    }
    return X_edge, Y_edge, X_node, Y_node, meta

def split_xy(X: np.ndarray, Y: np.ndarray, dev_ratio: float, seed: int):
    idx = np.arange(len(Y))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    X = X[idx]
    Y = Y[idx]

    split = int(len(Y) * (1.0 - dev_ratio))
    split = max(1, min(split, len(Y) - 1))
    Xtr, Ytr = X[:split], Y[:split]
    Xdv, Ydv = X[split:], Y[split:]
    return Xtr, Ytr, Xdv, Ydv


def train_one_binary_model(
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
    Xtr, Ytr, Xdv, Ydv = split_xy(X, Y, dev_ratio=0.1, seed=seed)

    train_ds = BinaryDataset(Xtr, Ytr)
    dev_ds = BinaryDataset(Xdv, Ydv)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=train_batch_size, shuffle=False)

    pos_count = float(Ytr.sum())
    neg_count = float(len(Ytr) - pos_count)
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

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item()) * len(yb)

        dev_f1 = eval_binary_f1(model, dev_loader, device, threshold=threshold)
        scheduler.step(dev_f1)

        train_loss = total_loss / max(1, len(train_ds))
        print(f'[{tag}][Epoch {epoch}] train_loss={train_loss:.4f} dev_f1={dev_f1:.4f}')

        if dev_f1 > best_dev:
            best_dev = dev_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f'[{tag}] early stop, patience={patience}')
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    return best_state, best_dev


def train_model(args, samples: List[Dict[str, Any]], embedder: SentenceTransformer) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache_path = args.train_cache_path.strip()

    if cache_path and os.path.exists(cache_path) and not args.rebuild_train_cache:
        print(f"[Load train cache] {cache_path}")
        with open(cache_path, "rb") as f:
            cache_obj = pickle.load(f)
        X_edge = cache_obj["X_edge"]
        Y_edge = cache_obj["Y_edge"]
        X_node = cache_obj["X_node"]
        Y_node = cache_obj["Y_node"]
        meta = cache_obj["meta"]
    else:
        X_edge, Y_edge, X_node, Y_node, meta = collect_training_examples(args, samples, embedder)

        if cache_path:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "X_edge": X_edge,
                        "Y_edge": Y_edge,
                        "X_node": X_node,
                        "Y_node": Y_node,
                        "meta": meta,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"[Saved train cache] {cache_path}")

    print(json.dumps(meta, indent=2, ensure_ascii=False))

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

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
        model=edge_model,
        X=X_edge,
        Y=Y_edge,
        train_batch_size=args.train_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        threshold=args.threshold,
        device=device,
        seed=args.seed,
        tag='edge',
    )

    node_state, node_best_dev = train_one_binary_model(
        model=node_model,
        X=X_node,
        Y=Y_node,
        train_batch_size=args.train_batch_size,
        lr=args.end_lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        threshold=args.end_threshold,
        device=device,
        seed=args.seed,
        tag='node_end',
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
    }
    torch.save(ckpt, args.model_ckpt)
    print(f'[Saved] {args.model_ckpt}')
    print(json.dumps({
        'edge_best_dev_f1': edge_best_dev,
        'node_best_dev_f1': node_best_dev,
        **meta
    }, ensure_ascii=False, indent=2))


def eval_binary_f1(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> float:
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            probs = torch.sigmoid(model(xb))
            pred = (probs >= threshold).float()
            tp += int(((pred == 1) & (yb == 1)).sum().item())
            fp += int(((pred == 1) & (yb == 0)).sum().item())
            fn += int(((pred == 0) & (yb == 1)).sum().item())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def load_model(model_ckpt: str, cpu: bool = False) -> Tuple[MLPScorer, NodeEndScorer, Dict[str, Any], torch.device]:
    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    ckpt = torch.load(model_ckpt, map_location='cpu')

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
    eids = sorted(feat_dict.keys())
    X = np.stack([feat_dict[eid]['vector'] for eid in eids]).astype(np.float32)

    scores = []
    with torch.no_grad():
        for start in range(0, len(eids), infer_batch_size):
            xb = torch.from_numpy(X[start:start + infer_batch_size]).to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            scores.extend(probs.tolist())

    return {eid: float(s) for eid, s in zip(eids, scores)}


def score_nodes_with_model(
    model: NodeEndScorer,
    device: torch.device,
    node_feat_dict: Dict[int, Dict[str, Any]],
    infer_batch_size: int,
) -> Dict[int, float]:
    nids = sorted(node_feat_dict.keys())
    X = np.stack([node_feat_dict[nid]['vector'] for nid in nids]).astype(np.float32)

    scores = []
    with torch.no_grad():
        for start in range(0, len(nids), infer_batch_size):
            xb = torch.from_numpy(X[start:start + infer_batch_size]).to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            scores.extend(probs.tolist())

    return {nid: float(s) for nid, s in zip(nids, scores)}


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


# ============================================================
# 8) Inference pipeline
# ============================================================
def create_dag_with_model(
    args,
    samples: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    edge_model: MLPScorer,
    node_model: NodeEndScorer,
    ckpt: Dict[str, Any],
    device: torch.device,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    if args.limit is not None and args.limit > 0:
        samples = samples[:args.limit]

    if verbose:
        samples = [samples[2]]

    out_samples: List[Dict[str, Any]] = []
    answer_recall = 0
    graph_recall = 0
    none_sink_recall = 0

    for sample in tqdm(samples, desc='Create DAG (two-stage node-ending aware)'):
        question = norm_text(sample.get('question', ''))
        answer = norm_text(sample.get('answer', ''))

        node_names, kvedges, out_adj, in_adj = build_kvedge_graph(sample, supporting_only=args.supporting_only)
        if not kvedges:
            sample['dag'] = {'kv_nodes': [], 'adj': [], 'meta': {'reason': 'no_kv_edges'}}
            out_samples.append(sample)
            continue

        node_emb = embed_texts(embedder, node_names, batch_size=args.batch_size)

        topic_nodes, feat_dict = build_edge_feature_dict(
            question=question,
            node_names=node_names,
            node_emb=node_emb,
            kvedges=kvedges,
            out_adj=out_adj,
            in_adj=in_adj,
            embedder=embedder,
            batch_size=args.batch_size,
            topic_top_k=args.topic_top_k,
            dde_hops=args.dde_hops,
            mention_bonus=args.mention_bonus,
        )

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

        edge_scores = score_edges_with_model(
            edge_model,
            device,
            feat_dict,
            infer_batch_size=args.infer_batch_size,
        )
        node_end_scores = score_nodes_with_model(
            node_model,
            device,
            node_feat_dict,
            infer_batch_size=args.infer_batch_size,
        )

        apply_two_stage_joint_scores(
            kvedges=kvedges,
            edge_scores=edge_scores,
            node_end_scores=node_end_scores,
            alpha=args.end_alpha,
            beta=args.end_beta,
            gamma=args.end_gamma,
        )

        if any(value_matches_answer(e.value, answer) for e in kvedges.values()):
            graph_recall += 1

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
        kept_edges = break_cycles_to_dag(kept_nodes, kept_edges, kvedges)
        kept_nodes, kept_edges = enforce_max_sinks(topic_nodes, args.max_sinks, kept_nodes, kept_edges, kvedges)

        if not kept_edges:
            sample['dag'] = {'kv_nodes': [], 'adj': [], 'meta': {'reason': 'empty_after_prune'}}
            out_samples.append(sample)
            continue

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
            answer_recall += 1

        if any(value_matches_answer(kvedges[eid].value, answer) for eid in kept_edges):
            none_sink_recall += 1

        kv_nodes, adj = export_kv_nodes_and_adj(kept_edges, kvedges, keep_score=args.keep_score)
        goal_ids: List[int] = []
        if answer:
            for i, kv in enumerate(kv_nodes):
                if value_matches_answer(kv.get('value', ''), answer):
                    goal_ids.append(i)

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
        out_samples.append(sample)

    if len(samples) > 0:
        print(f'Answer recall: {answer_recall / len(samples):.4f}')
        print(f'Graph  recall: {graph_recall / len(samples):.4f}')
        print(f'None-sink recall: {none_sink_recall / len(samples):.4f}')

    return out_samples
# ============================================================
# 9) CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'infer'], default='infer')
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
    ap.add_argument('--max_nodes', type=int, default=30)
    ap.add_argument('--max_edges', type=int, default=40)
    ap.add_argument('--max_sinks', type=int, default=8)

    # training
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--hidden_dim', type=int, default=512)
    ap.add_argument('--dropout', type=float, default=0.10)
    ap.add_argument('--train_batch_size', type=int, default=512)
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

    args = ap.parse_args()

    print(args)
    embedder = SentenceTransformer(args.st_model)
    samples = read_json_or_jsonl(args.input)
    if args.limit:
        samples = samples[:args.limit]

    print(f'Load {len(samples)} samples from {args.input}')

    if args.mode == 'train':
        train_model(args, samples, embedder)
        return

    if not args.model_ckpt or not os.path.exists(args.model_ckpt):
        raise FileNotFoundError(f'For --mode infer, model checkpoint is required: {args.model_ckpt}')

    edge_model, node_model, ckpt, device = load_model(args.model_ckpt, cpu=args.cpu)
    print(f'Loaded scorer from {args.model_ckpt}')
    out = create_dag_with_model(
            args,
            samples,
            embedder,
            edge_model,
            node_model,
            ckpt,
            device,
            verbose=args.verbose,
        )
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
