import argparse
import json
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


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
# 4) Triple iterator
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


# ============================================================
# 5) Graph build
# ============================================================
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
    sN = norm_match(s)
    oN = norm_match(o)
    vN = norm_match(value)
    kN = norm_match(key)
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
# 6) SubgraphRAG-style retrieval (unsupervised approximation)
# ============================================================
def identify_topic_entities(
    question: str,
    node_names: List[str],
    node_emb: np.ndarray,
    q_emb: np.ndarray,
    top_k: int,
    mention_bonus: float,
) -> List[int]:
    if len(node_names) == 0:
        return []
    sims = cosine_sim_matrix(q_emb[None, :], node_emb).reshape(-1)
    scores = sims.copy()
    for i, name in enumerate(node_names):
        if contains_mention(question, name):
            scores[i] += mention_bonus
        scores[i] += 0.15 * jaccard(question, name)
    order = np.argsort(-scores)
    out = []
    for idx in order[: max(1, min(top_k, len(order)))]:
        out.append(int(idx))
    return out


def compute_directional_distance_encoding(
    num_nodes: int,
    out_adj: Dict[int, List[int]],
    in_adj: Dict[int, List[int]],
    kvedges: Dict[int, KVEdge],
    topic_nodes: List[int],
    max_hops: int,
) -> np.ndarray:
    """
    Approximate SubgraphRAG's DDE for directed graphs.
    Returns [N, 1 + max_hops + max_hops] = [seed, forward_in, reverse_out].
    """
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
        indeg = np.zeros((num_nodes,), dtype=np.float32)
        for v, eids in in_adj.items():
            indeg[v] = max(1.0, float(len(eids)))
        nxt = nxt / indeg
        feats.append(nxt)
        cur = nxt

    cur = s0.copy()
    reverse_feats = []
    for _ in range(max_hops):
        nxt = np.zeros_like(cur)
        for e in kvedges.values():
            nxt[e.src] += cur[e.dst]
        outdeg = np.zeros((num_nodes,), dtype=np.float32)
        for u, eids in out_adj.items():
            outdeg[u] = max(1.0, float(len(eids)))
        nxt = nxt / outdeg
        reverse_feats.append(nxt)
        cur = nxt

    feats.extend(reverse_feats)
    return np.stack(feats, axis=1).astype(np.float32)


def edge_semantic_scores(
    question: str,
    q_emb: np.ndarray,
    node_names: List[str],
    node_emb: np.ndarray,
    kvedges: Dict[int, KVEdge],
    embedder: SentenceTransformer,
    batch_size: int,
) -> Dict[int, Dict[str, float]]:
    eids = sorted(kvedges.keys())
    rel_texts = [kvedges[eid].relation or kvedges[eid].key for eid in eids]
    key_texts = [kvedges[eid].key for eid in eids]
    val_texts = [kvedges[eid].value for eid in eids]

    rel_emb = embed_texts(embedder, rel_texts, batch_size=batch_size)
    key_emb = embed_texts(embedder, key_texts, batch_size=batch_size)
    val_emb = embed_texts(embedder, val_texts, batch_size=batch_size)

    rel_sim = cosine_sim_matrix(q_emb[None, :], rel_emb).reshape(-1)
    key_sim = cosine_sim_matrix(q_emb[None, :], key_emb).reshape(-1)
    val_sim = cosine_sim_matrix(q_emb[None, :], val_emb).reshape(-1)
    src_sim = np.array([float(np.dot(q_emb, node_emb[kvedges[eid].src])) for eid in eids], dtype=np.float32)
    dst_sim = np.array([float(np.dot(q_emb, node_emb[kvedges[eid].dst])) for eid in eids], dtype=np.float32)

    out: Dict[int, Dict[str, float]] = {}
    for i, eid in enumerate(eids):
        e = kvedges[eid]
        mention_src = 1.0 if contains_mention(question, e.src_name) else 0.0
        mention_dst = 1.0 if contains_mention(question, e.dst_name) else 0.0
        lex_rel = jaccard(question, e.relation)
        lex_key = jaccard(question, e.key)
        lex_val = jaccard(question, e.value)
        out[eid] = {
            'rel_sim': float(rel_sim[i]),
            'key_sim': float(key_sim[i]),
            'val_sim': float(val_sim[i]),
            'src_sim': float(src_sim[i]),
            'dst_sim': float(dst_sim[i]),
            'mention_src': mention_src,
            'mention_dst': mention_dst,
            'lex_rel': lex_rel,
            'lex_key': lex_key,
            'lex_val': lex_val,
            'is_attr': 1.0 if (e.triple_type or '').upper() == 'ATTRIBUTE' else 0.0,
            'dir_gain': float(dst_sim[i] - src_sim[i]),
        }
    return out


def score_edges_subgraphrag_style(
    question: str,
    node_names: List[str],
    node_emb: np.ndarray,
    kvedges: Dict[int, KVEdge],
    out_adj: Dict[int, List[int]],
    in_adj: Dict[int, List[int]],
    embedder: SentenceTransformer,
    batch_size: int,
    topic_top_k: int,
    dde_hops: int,
    mention_bonus: float,
    w_key: float,
    w_rel: float,
    w_val: float,
    w_src: float,
    w_dst: float,
    w_dir: float,
    w_dde: float,
    w_lex: float,
    topic_dst_bonus: float,
) -> Tuple[List[int], np.ndarray]:
    q_emb = embed_texts(embedder, [question], batch_size=batch_size)[0]
    topic_nodes = identify_topic_entities(question, node_names, node_emb, q_emb, top_k=topic_top_k, mention_bonus=mention_bonus)
    dde = compute_directional_distance_encoding(len(node_names), out_adj, in_adj, kvedges, topic_nodes, max_hops=dde_hops)
    sem = edge_semantic_scores(question, q_emb, node_names, node_emb, kvedges, embedder, batch_size)

    for eid, e in kvedges.items():
        f_src = dde[e.src]
        f_dst = dde[e.dst]
        # topic proximity: closer to topic at source, but slightly farther/expanded at dst to push answers outward
        topic_seed_src = float(f_src[0])
        topic_seed_dst = float(f_dst[0])
        forward_src = float(np.sum(f_src[1:1 + dde_hops]))
        forward_dst = float(np.sum(f_dst[1:1 + dde_hops]))
        reverse_src = float(np.sum(f_src[1 + dde_hops:]))
        reverse_dst = float(np.sum(f_dst[1 + dde_hops:]))
        dde_score = (
            0.65 * forward_src +
            1.10 * forward_dst +
            0.15 * reverse_src +
            0.20 * reverse_dst +
            0.35 * topic_seed_src +
            topic_dst_bonus * topic_seed_dst
        )

        s = sem[eid]
        lex_score = max(s['lex_rel'], s['lex_key']) + 0.35 * s['lex_val']
        score = (
            w_key * s['key_sim'] +
            w_rel * s['rel_sim'] +
            w_val * s['val_sim'] +
            w_src * s['src_sim'] +
            w_dst * s['dst_sim'] +
            w_dir * s['dir_gain'] +
            w_dde * dde_score +
            w_lex * lex_score +
            0.10 * s['mention_src'] +
            0.06 * s['mention_dst'] +
            0.04 * s['is_attr']
        )
        e.score = float(score)

    ranked = sorted(kvedges.keys(), key=lambda eid: kvedges[eid].score, reverse=True)
    return topic_nodes, dde


def dedup_edges(
    kvedges: Dict[int, KVEdge],
    per_src_cap: int,
    keep_best_per_signature: bool,
) -> Set[int]:
    keep: Set[int] = set()
    bucket: Dict[int, List[int]] = defaultdict(list)
    for eid, e in kvedges.items():
        bucket[e.src].append(eid)
    for src, eids in bucket.items():
        eids.sort(key=lambda eid: kvedges[eid].score, reverse=True)
        keep.update(eids[:max(1, per_src_cap)])

    if not keep_best_per_signature:
        return keep

    best: Dict[Tuple[int, int, str], int] = {}
    for eid in keep:
        e = kvedges[eid]
        sig = (e.src, e.dst, norm_match(e.relation))
        cur = best.get(sig)
        if cur is None or kvedges[eid].score > kvedges[cur].score:
            best[sig] = eid
    return set(best.values())


def select_subgraph_edges(
    topic_nodes: List[int],
    kvedges: Dict[int, KVEdge],
    out_adj: Dict[int, List[int]],
    max_edges: int,
    max_nodes: int,
    per_src_cap: int,
    expansion_hops: int,
    seed_edge_topk: int,
) -> Tuple[Set[int], Set[int]]:
    keep = dedup_edges(kvedges, per_src_cap=per_src_cap, keep_best_per_signature=True)
    ranked = [eid for eid in sorted(keep, key=lambda x: kvedges[x].score, reverse=True)]

    # Stage 1: global top edges, close to paper's parallel triple scoring
    selected_edges: Set[int] = set(ranked[:max(1, min(seed_edge_topk, len(ranked), max_edges))])
    selected_nodes: Set[int] = set(topic_nodes)
    for eid in list(selected_edges):
        selected_nodes.add(kvedges[eid].src)
        selected_nodes.add(kvedges[eid].dst)

    # Stage 2: topic-centered expansion to turn retrieved triples into a compact subgraph
    frontier = deque(topic_nodes)
    seen = set(topic_nodes)
    hop = 0
    while frontier and hop < expansion_hops and len(selected_edges) < max_edges and len(selected_nodes) < max_nodes:
        next_frontier = deque()
        while frontier and len(selected_edges) < max_edges and len(selected_nodes) < max_nodes:
            u = frontier.popleft()
            cand = [eid for eid in out_adj.get(u, []) if eid in keep]
            cand.sort(key=lambda eid: kvedges[eid].score, reverse=True)
            for eid in cand[:per_src_cap]:
                e = kvedges[eid]
                if len(selected_edges) >= max_edges or len(selected_nodes) >= max_nodes:
                    break
                selected_edges.add(eid)
                selected_nodes.add(e.src)
                selected_nodes.add(e.dst)
                if e.dst not in seen:
                    seen.add(e.dst)
                    next_frontier.append(e.dst)
        frontier = next_frontier
        hop += 1

    # Stage 3: fill remaining budget with best globally ranked edges touching selected nodes
    if len(selected_edges) < max_edges:
        for eid in ranked:
            e = kvedges[eid]
            if eid in selected_edges:
                continue
            if len(selected_edges) >= max_edges:
                break
            if (e.src in selected_nodes) or (e.dst in selected_nodes):
                if len(selected_nodes) + int(e.src not in selected_nodes) + int(e.dst not in selected_nodes) > max_nodes:
                    continue
                selected_edges.add(eid)
                selected_nodes.add(e.src)
                selected_nodes.add(e.dst)

    return selected_nodes, selected_edges


# ============================================================
# 7) DAG enforcement / sink pruning
# ============================================================
def topo_sort(nodes: Set[int], edges_kept: Set[int], kvedges: Dict[int, KVEdge]) -> Optional[List[int]]:
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
        for v in kept_out.get(u, ()):  # pragma: no branch
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

        in_edges = [eid for eid in kept_edges if kvedges[eid].dst == sinks[-1]]
        if not in_edges:
            kept_nodes.remove(sinks[-1])
        else:
            worst = min(in_edges, key=lambda eid: kvedges[eid].score)
            kept_edges.remove(worst)
        kept_nodes, kept_edges = prune_to_reachable(topic_nodes, kept_nodes, kept_edges, kvedges)
        if not kept_edges:
            return kept_nodes, kept_edges


# ============================================================
# 8) Export / metrics
# ============================================================
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


def value_matches_answer(value: str, answer: str) -> bool:
    v = norm_match(value)
    a = norm_match(answer)
    if not v or not a:
        return False
    return v == a or a in v or v in a


# ============================================================
# 9) Main pipeline
# ============================================================
def create_dag(args, samples: List[Dict[str, Any]], embedder: SentenceTransformer) -> List[Dict[str, Any]]:
    if args.limit is not None and args.limit > 0:
        samples = samples[:args.limit]

    out_samples: List[Dict[str, Any]] = []
    answer_recall = 0
    graph_recall = 0
    none_sink_recall = 0

    for sample in tqdm(samples, desc='Create DAG (SubgraphRAG-style)'):
        question = norm_text(sample.get('question', ''))
        answer = norm_text(sample.get('answer', ''))

        node_names, kvedges, out_adj, in_adj = build_kvedge_graph(sample, supporting_only=args.supporting_only)
        if not kvedges:
            sample['dag'] = {'kv_nodes': [], 'adj': [], 'meta': {'reason': 'no_kv_edges'}}
            out_samples.append(sample)
            continue

        node_emb = embed_texts(embedder, node_names, batch_size=args.batch_size)
        topic_nodes, _ = score_edges_subgraphrag_style(
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
            w_key=args.w_key,
            w_rel=args.w_rel,
            w_val=args.w_val,
            w_src=args.w_src,
            w_dst=args.w_dst,
            w_dir=args.w_dir,
            w_dde=args.w_dde,
            w_lex=args.w_lex,
            topic_dst_bonus=args.topic_dst_bonus,
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
        ansN = norm_match(answer)
        goal_ids: List[int] = []
        if ansN:
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
            },
        }
        out_samples.append(sample)

    if len(samples) > 0:
        print(f'Answer recall: {answer_recall / len(samples):.4f}')
        print(f'Graph  recall: {graph_recall / len(samples):.4f}')
        print(f'None-sink recall: {none_sink_recall / len(samples):.4f}')

    return out_samples


# ============================================================
# 10) CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--st_model', default='sentence-transformers/all-MiniLM-L6-v2')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--keep_score', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--supporting_only', action='store_true')

    # subgraphrag-style knobs
    ap.add_argument('--topic_top_k', type=int, default=6)
    ap.add_argument('--dde_hops', type=int, default=3)
    ap.add_argument('--mention_bonus', type=float, default=0.20)
    ap.add_argument('--seed_edge_topk', type=int, default=18)
    ap.add_argument('--expansion_hops', type=int, default=2)
    ap.add_argument('--per_src_cap', type=int, default=3)
    ap.add_argument('--max_nodes', type=int, default=30)
    ap.add_argument('--max_edges', type=int, default=40)
    ap.add_argument('--max_sinks', type=int, default=8)

    # scoring weights
    ap.add_argument('--w_key', type=float, default=0.34)
    ap.add_argument('--w_rel', type=float, default=0.18)
    ap.add_argument('--w_val', type=float, default=0.08)
    ap.add_argument('--w_src', type=float, default=0.10)
    ap.add_argument('--w_dst', type=float, default=0.16)
    ap.add_argument('--w_dir', type=float, default=0.12)
    ap.add_argument('--w_dde', type=float, default=0.22)
    ap.add_argument('--w_lex', type=float, default=0.12)
    ap.add_argument('--topic_dst_bonus', type=float, default=0.15)
    args = ap.parse_args()

    print(args)
    embedder = SentenceTransformer(args.st_model)
    samples = read_json_or_jsonl(args.input)
    print(f'Load {len(samples)} samples from {args.input}')

    out = create_dag(args, samples, embedder)
    if args.output.endswith('.jsonl'):
        write_jsonl(args.output, out)
    elif args.output.endswith('.json'):
        write_json(args.output, out)
    else:
        raise ValueError(f'Unknown file format: {args.output}')
    print(f'[DONE] input={len(samples)} output={len(out)} saved_to={args.output}')


if __name__ == '__main__':
    main()
