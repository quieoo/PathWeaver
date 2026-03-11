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



def collect_training_examples(
    args,
    samples: List[Dict[str, Any]],
    embedder: SentenceTransformer,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fast version:
      Pass 1: build graph metadata + collect all unique texts
      Pass 2: one-shot large-batch embedding + feature construction
    """
    xs: List[np.ndarray] = []
    ys: List[int] = []
    total_edges = 0
    pos_edges = 0

    used_samples = samples[:args.limit] if args.limit else samples
    train_embed_batch_size = args.batch_size

    # ------------------------------------------------------------
    # Pass 1: preprocess graph objects and collect all unique texts
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Pass 2: global large-batch embedding
    # ------------------------------------------------------------
    text_emb_cache = embed_texts_cached(
        embedder=embedder,
        texts=all_texts,
        batch_size=train_embed_batch_size,
    )

    # ------------------------------------------------------------
    # Pass 3: feature construction using cached embeddings
    # ------------------------------------------------------------
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

        _, feat_dict = build_edge_feature_dict(
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

        pos_ids, neg_ids = [], []
        for eid, e in kvedges.items():
            y = weak_label_edge(sample, e)
            total_edges += 1
            if y == 1:
                pos_edges += 1
                pos_ids.append(eid)
            else:
                neg_ids.append(eid)

        # all positives + semantic hard negatives
        neg_ids.sort(
            key=lambda eid: float(
                feat_dict[eid]['scalar'][3] + 0.5 * feat_dict[eid]['scalar'][4] + 0.3 * feat_dict[eid]['scalar'][1]
            ),
            reverse=True,
        )
        max_negs = max(len(pos_ids) * args.neg_pos_ratio, args.min_negatives_per_sample)
        kept_neg_ids = neg_ids[:max_negs]

        for eid in pos_ids:
            xs.append(feat_dict[eid]['vector'])
            ys.append(1)
        for eid in kept_neg_ids:
            xs.append(feat_dict[eid]['vector'])
            ys.append(0)

    if not xs:
        raise ValueError('No training examples constructed. Check your input data / labels.')

    X = np.stack(xs).astype(np.float32)
    Y = np.array(ys, dtype=np.float32)
    meta = {
        'num_examples': int(len(Y)),
        'num_pos': int(Y.sum()),
        'num_neg': int(len(Y) - Y.sum()),
        'raw_total_edges': int(total_edges),
        'raw_pos_edges': int(pos_edges),
        'input_dim': int(X.shape[1]),
        'num_unique_emb_texts': int(len(text_emb_cache)),
        'train_embed_batch_size': int(train_embed_batch_size),
    }
    return X, Y, meta


def train_model(args, samples: List[Dict[str, Any]], embedder: SentenceTransformer) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # X, Y, meta = collect_training_examples(args, samples, embedder)
    cache_path = args.train_cache_path.strip()

    if cache_path and os.path.exists(cache_path) and not args.rebuild_train_cache:
        print(f"[Load train cache] {cache_path}")
        with open(cache_path, "rb") as f:
            cache_obj = pickle.load(f)
        X = cache_obj["X"]
        Y = cache_obj["Y"]
        meta = cache_obj["meta"]
    else:
        X, Y, meta = collect_training_examples(args, samples, embedder)

        if cache_path:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "X": X,
                        "Y": Y,
                        "meta": meta,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"[Saved train cache] {cache_path}")

    print(json.dumps(meta, indent=2, ensure_ascii=False))


    idx = np.arange(len(Y))
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]

    split = int(len(Y) * (1.0 - args.dev_ratio))
    split = max(1, min(split, len(Y) - 1))
    Xtr, Ytr = X[:split], Y[:split]
    Xdv, Ydv = X[split:], Y[split:]

    train_ds = EdgeDataset(Xtr, Ytr)
    dev_ds = EdgeDataset(Xdv, Ydv)
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.train_batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = MLPScorer(in_dim=X.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    pos_count = float(Ytr.sum())
    neg_count = float(len(Ytr) - pos_count)
    pos_weight = torch.tensor([max(1.0, neg_count / max(1.0, pos_count))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_dev = -1.0
    patience = 0

    for epoch in tqdm(range(1, args.epochs + 1)):
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

        dev_f1 = eval_binary_f1(model, dev_loader, device, threshold=args.threshold)
        train_loss = total_loss / max(1, len(train_ds))
        print(f'[Epoch {epoch}] train_loss={train_loss:.4f} dev_f1={dev_f1:.4f}')

        if dev_f1 > best_dev:
            best_dev = dev_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f'[Early stop] patience={args.patience}')
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    os.makedirs(os.path.dirname(args.model_ckpt) or '.', exist_ok=True)
    ckpt = {
        'state_dict': best_state,
        'input_dim': int(X.shape[1]),
        'hidden_dim': int(args.hidden_dim),
        'dropout': float(args.dropout),
        'threshold': float(args.threshold),
        'meta': meta,
        'train_args': vars(args),
    }
    torch.save(ckpt, args.model_ckpt)
    print(f'[Saved] {args.model_ckpt}')
    print(json.dumps({'best_dev_f1': best_dev, **meta}, ensure_ascii=False, indent=2))


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

def load_model(model_ckpt: str, cpu: bool = False) -> Tuple[MLPScorer, Dict[str, Any], torch.device]:
    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    ckpt = torch.load(model_ckpt, map_location='cpu')
    model = MLPScorer(in_dim=ckpt['input_dim'], hidden_dim=ckpt.get('hidden_dim', 512), dropout=ckpt.get('dropout', 0.1))
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    return model, ckpt, device


# ============================================================
# 7) Retrieval / DAG post-processing
# ============================================================
def score_edges_with_model(model: MLPScorer, device: torch.device, feat_dict: Dict[int, Dict[str, Any]], kvedges: Dict[int, KVEdge], infer_batch_size: int) -> None:
    eids = sorted(kvedges.keys())
    X = np.stack([feat_dict[eid]['vector'] for eid in eids]).astype(np.float32)
    scores = []
    with torch.no_grad():
        for start in range(0, len(eids), infer_batch_size):
            xb = torch.from_numpy(X[start:start + infer_batch_size]).to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            scores.extend(probs.tolist())
    for eid, s in zip(eids, scores):
        kvedges[eid].score = float(s)


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
def create_dag_with_model(args, samples: List[Dict[str, Any]], embedder: SentenceTransformer, model: MLPScorer, ckpt: Dict[str, Any], device: torch.device, verbose: bool = False) -> List[Dict[str, Any]]:
    if args.limit is not None and args.limit > 0:
        samples = samples[:args.limit]

    if verbose:
        samples=[samples[2]]

    out_samples: List[Dict[str, Any]] = []
    answer_recall = 0
    graph_recall = 0
    none_sink_recall = 0
    # --- new stats for sink-miss analysis ---
    miss_sink_but_in_graph = 0
    miss_due_to_answer_node_has_outgoing = 0
    miss_due_to_other_reason = 0

    for sample in tqdm(samples, desc='Create DAG (trainable SubgraphRAG-style)'):
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
        score_edges_with_model(model, device, feat_dict, kvedges, infer_batch_size=args.infer_batch_size)

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

        # ---------- answer recall on sinks ----------
        outdeg = defaultdict(int)
        for eid in kept_edges:
            e = kvedges[eid]
            outdeg[e.src] += 1
        sinks = [n for n in kept_nodes if outdeg.get(n, 0) == 0]

        answer_matched = False
        ansN = norm_match(answer)

        # all answer-bearing nodes in final graph:
        # if an edge value matches answer, regard its dst as an answer node
        answer_nodes_in_graph = set()

        if ansN:
            for eid in kept_edges:
                e = kvedges[eid]
                vN = norm_match(e.value)
                if vN and (vN == ansN or ansN in vN or vN in ansN):
                    answer_nodes_in_graph.add(e.dst)

            # sink-level answer match
            for sink in sinks:
                if sink in answer_nodes_in_graph:
                    answer_matched = True
                    break

        if answer_matched:
            answer_recall += 1

        # answer recall on path
        answer_in_graph = len(answer_nodes_in_graph) > 0
        if answer_in_graph:
            none_sink_recall += 1

        # --- new: analyze why answer in graph but not on sink ---
        if answer_in_graph and not answer_matched:
            miss_sink_but_in_graph += 1

            # if any answer node still has outgoing edges, then the miss is due to
            # "answer node exists, but it is not a sink"
            has_outgoing_answer_node = any(outdeg.get(n, 0) > 0 for n in answer_nodes_in_graph)

            if has_outgoing_answer_node:
                miss_due_to_answer_node_has_outgoing += 1
            else:
                miss_due_to_other_reason += 1
        
        # ----------------------------------------------------------
        

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
                'scorer': 'trainable_subgraphrag_mlp',
            },
        }
        out_samples.append(sample)

    if len(samples) > 0:
        print(f"Answer recall: {answer_recall / len(samples):.4f}")
        print(f"Graph  recall: {graph_recall  / len(samples):.4f}")
        print(f"None-sink recall: {none_sink_recall / len(samples):.4f}")

        print(f"Miss sink but in graph: {miss_sink_but_in_graph}")
        if miss_sink_but_in_graph > 0:
            print(
                "Among sink-miss samples, due to answer node having outgoing edges: "
                f"{miss_due_to_answer_node_has_outgoing} / {miss_sink_but_in_graph} "
                f"= {miss_due_to_answer_node_has_outgoing / miss_sink_but_in_graph:.4f}"
            )
            print(
                "Among sink-miss samples, other reasons: "
                f"{miss_due_to_other_reason} / {miss_sink_but_in_graph} "
                f"= {miss_due_to_other_reason / miss_sink_but_in_graph:.4f}"
            )

    return out_samples




# ============================================================
# 9) HippoRAG / HippoRAG2 modules
# ============================================================
def infer_kv_direction_hippo(s: str, o: str, key: str, value: str) -> Tuple[str, str]:
    sN = norm_match(s)
    oN = norm_match(o)
    vN = norm_match(value)
    kN = norm_match(key)

    if vN == oN:
        return 'forward', 'value==O'
    if vN == sN:
        return 'backward', 'value==S'

    s_in = sN and (sN in kN)
    o_in = oN and (oN in kN)
    if s_in and not o_in:
        return 'forward', 'key_has_S'
    if o_in and not s_in:
        return 'backward', 'key_has_O'
    return 'unknown', 'ambiguous'


def build_kvedge_graph_hippo(
    sample: Dict[str, Any],
    embedder: SentenceTransformer,
    batch_size: int,
    supporting_only: bool,
    pred_weight: float = 0.0,
) -> Tuple[List[str], Dict[int, List[int]], Dict[int, KVEdge]]:
    question = norm_text(sample.get('question', ''))
    q_emb = embed_texts(embedder, [question], batch_size=batch_size)[0:1]

    node_map: Dict[str, int] = {}
    node_names: List[str] = []
    kvedges: Dict[int, KVEdge] = {}
    out_adj: Dict[int, List[int]] = {}

    kv_records: List[Tuple[str, str, str, str, str, str, str, int]] = []
    seen: Set[Tuple[str, str]] = set()

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
            kv_records.append((title, ttype, rel, s, o, key, value, kv_idx))

    if not kv_records:
        return node_names, out_adj, kvedges

    keys = [r[5] for r in kv_records]
    key_emb = embed_texts(embedder, keys, batch_size=batch_size)
    key_sims = cosine_sim_matrix(q_emb, key_emb).reshape(-1)

    if pred_weight != 0.0:
        rels = [r[2] for r in kv_records]
        rel_emb = embed_texts(embedder, rels, batch_size=batch_size)
        rel_sims = cosine_sim_matrix(q_emb, rel_emb).reshape(-1)
    else:
        rel_sims = np.zeros((len(kv_records),), dtype=np.float32)

    kid = 0
    for i, (title, ttype, rel, s, o, key, value, kv_idx) in enumerate(kv_records):
        direction, _ = infer_kv_direction_hippo(s, o, key, value)
        if direction == 'backward':
            src_name, dst_name = o, s
        else:
            src_name, dst_name = s, o

        src = get_or_add_node(src_name, node_map, node_names)
        dst = get_or_add_node(dst_name, node_map, node_names)
        score = float(key_sims[i]) + float(pred_weight) * float(rel_sims[i])

        e = KVEdge(
            kid=kid,
            src=src,
            dst=dst,
            src_name=node_names[src],
            dst_name=node_names[dst],
            key=key,
            value=value,
            score=score,
            title=norm_text(title),
            triple_type=ttype,
            relation=rel,
            triple_s=s,
            triple_o=o,
            kv_idx=kv_idx,
        )
        kvedges[kid] = e
        out_adj.setdefault(src, []).append(kid)
        kid += 1

    for u in out_adj:
        out_adj[u].sort(key=lambda eid: kvedges[eid].score, reverse=True)
    return node_names, out_adj, kvedges


def rebuild_out_adj_hippo(kvedges: Dict[int, KVEdge]) -> Dict[int, List[int]]:
    out_adj: Dict[int, List[int]] = defaultdict(list)
    for eid, e in kvedges.items():
        out_adj[e.src].append(eid)
    for u in out_adj:
        out_adj[u].sort(key=lambda x: kvedges[x].score, reverse=True)
    return dict(out_adj)


def get_seed_nodes_hippo(
    question: str,
    node_names: List[str],
    embedder: SentenceTransformer,
    batch_size: int,
    top_m: int,
    mention_bonus: float,
    lexical_weight: float,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    if not node_names:
        return [], np.zeros((0,), dtype=np.float32), np.zeros((0, 384), dtype=np.float32)

    q_emb = embed_texts(embedder, [question], batch_size=batch_size)[0:1]
    node_emb = embed_texts(embedder, node_names, batch_size=batch_size)
    sem = cosine_sim_matrix(q_emb, node_emb).reshape(-1)

    scores = sem.astype(np.float32).copy()
    for i, name in enumerate(node_names):
        scores[i] += lexical_weight * lexical_overlap(question, name)
        if contains_mention(question, name):
            scores[i] += mention_bonus

    if len(scores) == 0:
        return [], scores, node_emb
    top_m = max(1, min(top_m, len(scores)))
    seed_ids = np.argsort(-scores)[:top_m].tolist()
    return seed_ids, scores, node_emb


def personalized_pagerank_dense_hippo(
    trans: np.ndarray,
    teleport: np.ndarray,
    alpha: float = 0.15,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    n = trans.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    r = teleport.astype(np.float32).copy()
    for _ in range(max_iter):
        new_r = (1.0 - alpha) * (trans.T @ r) + alpha * teleport
        if np.abs(new_r - r).sum() < tol:
            r = new_r
            break
        r = new_r
    s = float(r.sum())
    if s > 0:
        r = r / s
    return r.astype(np.float32)


def build_entity_transition_hippo(num_nodes: int, kvedges: Dict[int, KVEdge]) -> np.ndarray:
    if num_nodes == 0:
        return np.zeros((0, 0), dtype=np.float32)
    mat = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for e in kvedges.values():
        mat[e.src, e.dst] += 1.0
        # allow some reverse spreading like associative memory
        mat[e.dst, e.src] += 1.0
    for i in range(num_nodes):
        row_sum = float(mat[i].sum())
        if row_sum > 0:
            mat[i] /= row_sum
        else:
            mat[i, i] = 1.0
    return mat


def build_bipartite_transition_hippo(num_entities: int, kvedges: Dict[int, KVEdge]) -> Tuple[np.ndarray, Dict[int, int]]:
    num_edges = len(kvedges)
    total = num_entities + num_edges
    if total == 0:
        return np.zeros((0, 0), dtype=np.float32), {}

    edge_to_idx: Dict[int, int] = {}
    mat = np.zeros((total, total), dtype=np.float32)

    for local_idx, eid in enumerate(sorted(kvedges.keys())):
        gi = num_entities + local_idx
        edge_to_idx[eid] = gi
        e = kvedges[eid]
        # entity <-> memory-node (passage/triple node)
        mat[e.src, gi] += 1.0
        mat[gi, e.src] += 1.0
        mat[e.dst, gi] += 1.0
        mat[gi, e.dst] += 1.0

    for i in range(total):
        row_sum = float(mat[i].sum())
        if row_sum > 0:
            mat[i] /= row_sum
        else:
            mat[i, i] = 1.0
    return mat, edge_to_idx


def run_hipporag_entity_ppr_hippo(
    seed_ids: List[int],
    seed_scores: np.ndarray,
    num_nodes: int,
    kvedges: Dict[int, KVEdge],
    alpha: float,
    max_iter: int,
) -> np.ndarray:
    trans = build_entity_transition_hippo(num_nodes, kvedges)
    tele = np.zeros((num_nodes,), dtype=np.float32)
    if seed_ids:
        vals = np.array([max(0.0, float(seed_scores[i])) + 1e-6 for i in seed_ids], dtype=np.float32)
        vals = vals / vals.sum()
        for sid, v in zip(seed_ids, vals):
            tele[sid] = float(v)
    elif num_nodes > 0:
        tele[:] = 1.0 / float(num_nodes)
    return personalized_pagerank_dense_hippo(trans, tele, alpha=alpha, max_iter=max_iter)


def run_hipporag2_bipartite_ppr_hippo(
    seed_ids: List[int],
    seed_scores: np.ndarray,
    num_nodes: int,
    kvedges: Dict[int, KVEdge],
    alpha: float,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    trans, edge_to_idx = build_bipartite_transition_hippo(num_nodes, kvedges)
    total = trans.shape[0]
    tele = np.zeros((total,), dtype=np.float32)
    if seed_ids:
        vals = np.array([max(0.0, float(seed_scores[i])) + 1e-6 for i in seed_ids], dtype=np.float32)
        vals = vals / vals.sum()
        for sid, v in zip(seed_ids, vals):
            tele[sid] = float(v)
    elif total > 0:
        tele[:num_nodes] = 1.0 / float(max(1, num_nodes))
    ppr = personalized_pagerank_dense_hippo(trans, tele, alpha=alpha, max_iter=max_iter)
    return ppr[:num_nodes], ppr[num_nodes:], edge_to_idx


def is_attribute_edge_hippo(e: KVEdge, node_names: List[str]) -> bool:
    if (e.triple_type or '').upper() == 'ATTRIBUTE':
        return True
    dst_name = node_names[e.dst] if (0 <= e.dst < len(node_names)) else ''
    if any(ch.isdigit() for ch in dst_name):
        return True
    if len(dst_name.split()) <= 2 and dst_name and dst_name[0].islower():
        return True
    return False


def rank_edges_hipporag_hippo(
    question: str,
    node_names: List[str],
    kvedges: Dict[int, KVEdge],
    entity_ppr: np.ndarray,
    memory_scores: Optional[np.ndarray],
    edge_to_idx: Optional[Dict[int, int]],
    edge_query_weight: float,
    leaf_bonus: float,
) -> List[int]:
    scored: List[Tuple[float, int]] = []
    q_norm = normalize_lex(question)

    for eid, e in kvedges.items():
        base = float(entity_ppr[e.src]) + float(entity_ppr[e.dst])
        if memory_scores is not None and edge_to_idx is not None:
            local_idx = edge_to_idx[eid] - len(entity_ppr)
            if 0 <= local_idx < len(memory_scores):
                base += float(memory_scores[local_idx])

        # optional lightweight passage readout, kept small for faithfulness
        qsig = lexical_overlap(q_norm, e.key) + lexical_overlap(q_norm, e.relation)
        score = base + edge_query_weight * qsig
        if is_attribute_edge_hippo(e, node_names):
            score += leaf_bonus
        scored.append((score, eid))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [eid for _, eid in scored]


def select_top_edges_with_connectivity_hippo(
    ranked_eids: List[int],
    seed_nodes: Set[int],
    kvedges: Dict[int, KVEdge],
    max_edges: int,
    max_nodes: int,
) -> Tuple[Set[int], Set[int]]:
    kept_edges: Set[int] = set()
    kept_nodes: Set[int] = set(seed_nodes)
    frontier: Set[int] = set(seed_nodes)

    # first pass: prefer edges touching current frontier
    for eid in ranked_eids:
        if len(kept_edges) >= max_edges or len(kept_nodes) >= max_nodes:
            break
        e = kvedges[eid]
        if e.src in frontier or e.dst in frontier or e.src in kept_nodes or e.dst in kept_nodes:
            kept_edges.add(eid)
            kept_nodes.add(e.src)
            kept_nodes.add(e.dst)
            frontier.add(e.src)
            frontier.add(e.dst)

    # second pass: fill budget if too sparse
    if len(kept_edges) < max_edges and len(kept_nodes) < max_nodes:
        for eid in ranked_eids:
            if len(kept_edges) >= max_edges or len(kept_nodes) >= max_nodes:
                break
            if eid in kept_edges:
                continue
            e = kvedges[eid]
            kept_edges.add(eid)
            kept_nodes.add(e.src)
            kept_nodes.add(e.dst)

    return kept_nodes, kept_edges


def topo_sort_hippo(nodes: Set[int], edges_kept: Set[int], kvedges: Dict[int, KVEdge]) -> Optional[List[int]]:
    indeg = {n: 0 for n in nodes}
    out = {n: [] for n in nodes}
    for eid in edges_kept:
        e = kvedges[eid]
        if e.src in nodes and e.dst in nodes:
            out[e.src].append(e.dst)
            indeg[e.dst] += 1
    q = [n for n in nodes if indeg[n] == 0]
    order: List[int] = []
    while q:
        u = q.pop()
        order.append(u)
        for v in out.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != len(nodes):
        return None
    return order


def find_cycle_edges_hippo(nodes: Set[int], edges_kept: Set[int], kvedges: Dict[int, KVEdge]) -> List[int]:
    out_adj: Dict[int, List[int]] = defaultdict(list)
    for eid in edges_kept:
        e = kvedges[eid]
        if e.src in nodes and e.dst in nodes:
            out_adj[e.src].append(eid)

    state = {n: 0 for n in nodes}
    parent_edge: Dict[int, int] = {}

    def dfs(u: int) -> Optional[Tuple[int, int, int]]:
        state[u] = 1
        for eid in out_adj.get(u, []):
            v = kvedges[eid].dst
            if v not in nodes:
                continue
            if state[v] == 0:
                parent_edge[v] = eid
                res = dfs(v)
                if res is not None:
                    return res
            elif state[v] == 1:
                return (u, v, eid)
        state[u] = 2
        return None

    back = None
    for n in list(nodes):
        if state[n] == 0:
            back = dfs(n)
            if back is not None:
                break
    if back is None:
        return []

    u, v, back_eid = back
    cyc: List[int] = [back_eid]
    cur = u
    guard = 0
    while cur != v and guard < 10000:
        pe = parent_edge.get(cur)
        if pe is None:
            break
        cyc.append(pe)
        cur = kvedges[pe].src
        guard += 1
    return list(set(cyc))


def break_cycles_to_dag_kv_hippo(nodes: Set[int], edges_kept: Set[int], kvedges: Dict[int, KVEdge]) -> Set[int]:
    while True:
        order = topo_sort_hippo(nodes, edges_kept, kvedges)
        if order is not None:
            return edges_kept
        cyc = find_cycle_edges_hippo(nodes, edges_kept, kvedges)
        if not cyc:
            worst = min(list(edges_kept), key=lambda eid: kvedges[eid].score)
            edges_kept.remove(worst)
            continue
        worst = min(cyc, key=lambda eid: kvedges[eid].score)
        if worst in edges_kept:
            edges_kept.remove(worst)


def enforce_max_sinks_entity_hippo(
    seeds: Set[int],
    max_sinks: int,
    kept_nodes: Set[int],
    kept_kvedges: Set[int],
    kvedges: Dict[int, KVEdge],
) -> Tuple[Set[int], Set[int]]:
    if max_sinks <= 0:
        return kept_nodes, kept_kvedges

    def reachable(nodes: Set[int], edges_set: Set[int]) -> Tuple[Set[int], Set[int]]:
        out_adj: Dict[int, List[int]] = defaultdict(list)
        for eid in edges_set:
            e = kvedges[eid]
            out_adj[e.src].append(eid)

        vis: Set[int] = set()
        stack = list(seeds)
        while stack:
            u = stack.pop()
            if u in vis:
                continue
            if u not in nodes:
                continue
            vis.add(u)
            for eid in out_adj.get(u, []):
                v = kvedges[eid].dst
                if v not in vis:
                    stack.append(v)

        vis_edges = {eid for eid in edges_set if kvedges[eid].src in vis and kvedges[eid].dst in vis}
        return vis, vis_edges

    kept_nodes, kept_kvedges = reachable(kept_nodes, kept_kvedges)

    while True:
        outdeg = {n: 0 for n in kept_nodes}
        for eid in kept_kvedges:
            e = kvedges[eid]
            if e.src in kept_nodes and e.dst in kept_nodes:
                outdeg[e.src] += 1
        sinks = [n for n in kept_nodes if outdeg.get(n, 0) == 0]
        if len(sinks) <= max_sinks:
            break

        order = topo_sort_hippo(kept_nodes, kept_kvedges, kvedges)
        if order is None:
            kept_kvedges = break_cycles_to_dag_kv_hippo(kept_nodes, kept_kvedges, kvedges)
            order = topo_sort_hippo(kept_nodes, kept_kvedges, kvedges) or list(kept_nodes)

        best = {n: -1e9 for n in kept_nodes}
        for s in seeds:
            if s in kept_nodes:
                best[s] = 0.0

        out_adj_e: Dict[int, List[int]] = defaultdict(list)
        indeg_edges: Dict[int, List[int]] = defaultdict(list)
        for eid in kept_kvedges:
            e = kvedges[eid]
            out_adj_e[e.src].append(eid)
            indeg_edges[e.dst].append(eid)

        for u in order:
            if best[u] <= -1e8:
                continue
            for eid in out_adj_e.get(u, []):
                v = kvedges[eid].dst
                cand = best[u] + kvedges[eid].score
                if cand > best.get(v, -1e9):
                    best[v] = cand

        sinks_sorted = sorted(sinks, key=lambda n: best.get(n, -1e9))
        target = sinks_sorted[0]
        ins = indeg_edges.get(target, [])
        if not ins:
            kept_nodes.remove(target)
        else:
            worst_in = min(ins, key=lambda eid: kvedges[eid].score)
            kept_kvedges.discard(worst_in)

        kept_nodes, kept_kvedges = reachable(kept_nodes, kept_kvedges)
        if not kept_kvedges:
            break

    return kept_nodes, kept_kvedges


def export_kv_nodes_and_adj_hippo(
    kept_kvedges: Set[int],
    kvedges: Dict[int, KVEdge],
    keep_score: bool,
) -> Tuple[List[Dict[str, Any]], List[List[int]]]:
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


def create_dag_hippo(args, samples: List[Dict[str, Any]], embedder: SentenceTransformer) -> List[Dict[str, Any]]:
    if args.limit is not None and args.limit > 0:
        samples = samples[:args.limit]

    out_samples: List[Dict[str, Any]] = []
    answer_recall = 0
    graph_recall = 0
    none_sink_recall = 0

    # --- new stats for sink-miss analysis ---
    miss_sink_but_in_graph = 0
    miss_due_to_answer_node_has_outgoing = 0
    miss_due_to_other_reason = 0

    for sample in tqdm(samples, desc=f'Create DAG ({args.variant})'):
        question = norm_text(sample.get('question', ''))
        answer = norm_text(sample.get('answer', ''))

        node_names, out_adj, kvedges = build_kvedge_graph(
            sample=sample,
            embedder=embedder,
            batch_size=args.batch_size,
            supporting_only=args.supporting_only,
            pred_weight=args.pred_weight,
        )

        if not kvedges:
            sample['dag'] = {'kv_nodes': [], 'adj': [], 'meta': {'reason': 'no_kv_edges'}}
            out_samples.append(sample)
            continue

        # full-graph recall before any retrieval pruning
        for e in kvedges.values():
            if answer and (answer in e.value or e.value in answer):
                graph_recall += 1
                break

        seed_ids, seed_scores, _ = get_seed_nodes(
            question=question,
            node_names=node_names,
            embedder=embedder,
            batch_size=args.batch_size,
            top_m=args.seed_top_m,
            mention_bonus=args.mention_bonus,
            lexical_weight=args.lexical_weight,
        )
        seed_nodes = set(seed_ids)

        if args.variant == 'hipporag2':
            entity_ppr, memory_scores, edge_to_idx = run_hipporag2_bipartite_ppr_hippo(
                seed_ids=seed_ids,
                seed_scores=seed_scores,
                num_nodes=len(node_names),
                kvedges=kvedges,
                alpha=args.ppr_alpha,
                max_iter=args.ppr_iters,
            )
        else:
            entity_ppr = run_hipporag_entity_ppr_hippo(
                seed_ids=seed_ids,
                seed_scores=seed_scores,
                num_nodes=len(node_names),
                kvedges=kvedges,
                alpha=args.ppr_alpha,
                max_iter=args.ppr_iters,
            )
            memory_scores, edge_to_idx = None, None

        ranked_eids = rank_edges_hipporag_hippo(
            question=question,
            node_names=node_names,
            kvedges=kvedges,
            entity_ppr=entity_ppr,
            memory_scores=memory_scores,
            edge_to_idx=edge_to_idx,
            edge_query_weight=args.edge_query_weight,
            leaf_bonus=args.leaf_bonus,
        )

        kept_nodes, kept_kvedges = select_top_edges_with_connectivity_hippo(
            ranked_eids=ranked_eids,
            seed_nodes=seed_nodes,
            kvedges=kvedges,
            max_edges=args.max_edges,
            max_nodes=args.max_nodes,
        )

        if not kept_kvedges:
            sample['dag'] = {'kv_nodes': [], 'adj': [], 'meta': {'reason': 'retrieval_empty'}}
            out_samples.append(sample)
            continue

        kept_kvedges = break_cycles_to_dag_kv_hippo(kept_nodes, kept_kvedges, kvedges)

        if args.max_sinks is not None and args.max_sinks > 0:
            kept_nodes, kept_kvedges = enforce_max_sinks_entity_hippo(
                seeds=seed_nodes,
                max_sinks=args.max_sinks,
                kept_nodes=kept_nodes,
                kept_kvedges=kept_kvedges,
                kvedges=kvedges,
            )

        if not kept_kvedges:
            sample['dag'] = {'kv_nodes': [], 'adj': [], 'meta': {'reason': 'empty_after_sink_prune'}}
            out_samples.append(sample)
            continue

        # ---------- answer recall on sinks ----------
        outdeg = defaultdict(int)
        for eid in kept_kvedges:
            e = kvedges[eid]
            outdeg[e.src] += 1
        sinks = [n for n in kept_nodes if outdeg.get(n, 0) == 0]

        answer_matched = False
        ansN = norm_match(answer)

        # all answer-bearing nodes in final graph:
        # if an edge value matches answer, regard its dst as an answer node
        answer_nodes_in_graph = set()

        if ansN:
            for eid in kept_kvedges:
                e = kvedges[eid]
                vN = norm_match(e.value)
                if vN and (vN == ansN or ansN in vN or vN in ansN):
                    answer_nodes_in_graph.add(e.dst)

            # sink-level answer match
            for sink in sinks:
                if sink in answer_nodes_in_graph:
                    answer_matched = True
                    break

        if answer_matched:
            answer_recall += 1

        # answer recall on path
        answer_in_graph = len(answer_nodes_in_graph) > 0
        if answer_in_graph:
            none_sink_recall += 1

        # --- new: analyze why answer in graph but not on sink ---
        if answer_in_graph and not answer_matched:
            miss_sink_but_in_graph += 1

            # if any answer node still has outgoing edges, then the miss is due to
            # "answer node exists, but it is not a sink"
            has_outgoing_answer_node = any(outdeg.get(n, 0) > 0 for n in answer_nodes_in_graph)

            if has_outgoing_answer_node:
                miss_due_to_answer_node_has_outgoing += 1
            else:
                miss_due_to_other_reason += 1
        
        # ----------------------------------------------------------

        kv_nodes, adj = export_kv_nodes_and_adj_hippo(kept_kvedges, kvedges, keep_score=args.keep_score)

        goal_ids: List[int] = []
        if ansN:
            for i, kv in enumerate(kv_nodes):
                vN = norm_match(kv.get('value', ''))
                if vN and (vN == ansN or ansN in vN):
                    goal_ids.append(i)

        meta = {
            'num_entity_nodes': int(len(kept_nodes)),
            'num_kv_edges': int(len(kept_kvedges)),
            'num_kv_nodes': int(len(kv_nodes)),
            'goal_ids': goal_ids,
            'seed_node_ids': sorted(list(seed_nodes)),
            'seed_entities': [node_names[i] for i in sorted(seed_nodes) if 0 <= i < len(node_names)],
            'variant': args.variant,
        }
        sample['dag'] = {'kv_nodes': kv_nodes, 'adj': adj, 'meta': meta}
        out_samples.append(sample)

    if len(samples) > 0:
        print(f"Answer recall: {answer_recall / len(samples):.4f}")
        print(f"Graph  recall: {graph_recall  / len(samples):.4f}")
        print(f"None-sink recall: {none_sink_recall / len(samples):.4f}")

        print(f"Miss sink but in graph: {miss_sink_but_in_graph}")
        if miss_sink_but_in_graph > 0:
            print(
                "Among sink-miss samples, due to answer node having outgoing edges: "
                f"{miss_due_to_answer_node_has_outgoing} / {miss_sink_but_in_graph} "
                f"= {miss_due_to_answer_node_has_outgoing / miss_sink_but_in_graph:.4f}"
            )
            print(
                "Among sink-miss samples, other reasons: "
                f"{miss_due_to_other_reason} / {miss_sink_but_in_graph} "
                f"= {miss_due_to_other_reason / miss_sink_but_in_graph:.4f}"
            )

    return out_samples


# ============================================================
# 9) Unified dispatcher / CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'infer'], default='infer')
    ap.add_argument('--variant', choices=['subgraphrag', 'hipporag', 'hipporag2'], default='subgraphrag',
                    help='retrieval/scoring pipeline variant')
    ap.add_argument('--input', required=True, help='all_triples dataset file (.json or .jsonl)')
    ap.add_argument('--output', default='', help='output dataset path (.jsonl or .json)')
    ap.add_argument('--model_ckpt', default='subgraphrag_mlp.pt', help='used by subgraphrag train/infer')
    ap.add_argument('--st_model', default='sentence-transformers/all-MiniLM-L6-v2')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--infer_batch_size', type=int, default=4096)
    ap.add_argument('--keep_score', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--supporting_only', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--seed', type=int, default=42)

    # -------------------------
    # SubgraphRAG-only knobs
    # -------------------------
    ap.add_argument('--topic_top_k', type=int, default=6)
    ap.add_argument('--dde_hops', type=int, default=3)

    # shared-ish mention signal
    ap.add_argument('--mention_bonus', type=float, default=0.20)

    # SubgraphRAG retrieval / dag size
    ap.add_argument('--seed_edge_topk', type=int, default=18)
    ap.add_argument('--expansion_hops', type=int, default=2)
    ap.add_argument('--per_src_cap', type=int, default=3)

    # shared graph budget
    ap.add_argument('--max_nodes', type=int, default=30)
    ap.add_argument('--max_edges', type=int, default=40)
    ap.add_argument('--max_sinks', type=int, default=8)

    # training (SubgraphRAG only)
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
    ap.add_argument("--train_cache_path", type=str, default="",
                    help="optional path to save/load cached training examples (X,Y,meta)")
    ap.add_argument("--rebuild_train_cache", action="store_true",
                    help="force rebuild training cache even if cache file exists")

    # -------------------------
    # HippoRAG-only knobs
    # -------------------------
    ap.add_argument('--seed_top_m', type=int, default=8, help='number of query-linked seed entities for hipporag')
    ap.add_argument('--ppr_alpha', type=float, default=0.15, help='PPR teleport probability for hipporag')
    ap.add_argument('--ppr_iters', type=int, default=50, help='max iterations for hipporag PPR')
    ap.add_argument('--lexical_weight', type=float, default=0.20, help='lexical overlap weight for hipporag seed linking')
    ap.add_argument('--edge_query_weight', type=float, default=0.10, help='question-edge lexical readout weight for hipporag')
    ap.add_argument('--leaf_bonus', type=float, default=0.08, help='bonus for value-like/attribute edges in hipporag')
    ap.add_argument('--pred_weight', type=float, default=0.0, help='optional relation similarity weight when building hipporag KV edges')

    ap.add_argument('--verbose', action='store_true')

    args = ap.parse_args()
    print(args)

    if args.mode == 'train' and args.variant != 'subgraphrag':
        raise ValueError("--mode train is only supported for --variant subgraphrag, because hipporag/hipporag2 are non-parametric in the original code paths.")

    embedder = SentenceTransformer(args.st_model)
    samples = read_json_or_jsonl(args.input)
    if args.limit:
        samples = samples[:args.limit]
    print(f'Load {len(samples)} samples from {args.input}')

    if args.mode == 'train':
        train_model(args, samples, embedder)
        return

    if args.variant == 'subgraphrag':
        if not args.model_ckpt or not os.path.exists(args.model_ckpt):
            raise FileNotFoundError(f'For --variant subgraphrag --mode infer, model checkpoint is required: {args.model_ckpt}')
        model, ckpt, device = load_model(args.model_ckpt, cpu=args.cpu)
        print(f'Loaded scorer from {args.model_ckpt}')
        out = create_dag_with_model(args, samples, embedder, model, ckpt, device, verbose=args.verbose)
    else:
        out = create_dag_hippo(args, samples, embedder)

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
