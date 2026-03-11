import argparse
import json
import os
import re
from collections import defaultdict
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
    if isinstance(obj, dict) and isinstance(obj.get('data'), list):
        return obj['data']
    raise ValueError('Unsupported JSON root format. Expect list or {data:[...]}')


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
# 1) Normalization / lexical helpers
# ============================================================
_SPACE_RE = re.compile(r"\s+")
_ZW_RE = re.compile(r"[\u200b\u200c\u200d\uFEFF]")
_PAREN_CONTENT_RE = re.compile(r"\([^)]*\)")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_STOPWORDS = {
    'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'from', 'by', 'with', 'and', 'or',
    'is', 'was', 'were', 'are', 'be', 'been', 'being', 'that', 'this', 'these', 'those', 'what',
    'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how', 'has', 'have', 'had', 'do',
    'does', 'did', 'as', 'it', 'its', 'into', 'than', 'then', 'their', 'his', 'her', 'about',
    'after', 'before', 'over', 'under', 'between', 'through', 'during', 'without', 'within'
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
    s = norm_text(s).lower()
    s = _PUNCT_RE.sub(' ', s)
    toks = [t for t in s.split() if t and t not in _STOPWORDS]
    return ' '.join(toks)


def token_set(s: str) -> Set[str]:
    s = normalize_lex(s)
    return set(s.split()) if s else set()


def lexical_overlap(a: str, b: str) -> float:
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
    out = []
    seen = set()
    for kk in keys:
        if kk not in seen:
            out.append(kk)
            seen.add(kk)
    return out


def contains_mention(q: str, name: str) -> bool:
    qn = normalize_lex(q)
    nn = normalize_lex(name)
    if not qn or not nn:
        return False
    if nn in qn:
        return True
    toks = nn.split()
    if len(toks) >= 2 and ' '.join(toks[:2]) in qn:
        return True
    if len(toks) == 1 and len(toks[0]) >= 5 and toks[0] in qn:
        return True
    return False


# ============================================================
# 2) Embeddings
# ============================================================
def embed_texts(embedder: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
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
# 3) KVEdge
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

    def print(self) -> None:
        print(
            f"【{self.kid}】 <{self.triple_s}, {self.relation}, {self.triple_o}> "
            f"({self.key} [{self.src}], {self.value} [{self.dst}]) {self.score:.4f}"
        )


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


def infer_kv_direction(s: str, o: str, key: str, value: str) -> Tuple[str, str]:
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


def build_kvedge_graph(
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
        direction, _ = infer_kv_direction(s, o, key, value)
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


def rebuild_out_adj(kvedges: Dict[int, KVEdge]) -> Dict[int, List[int]]:
    out_adj: Dict[int, List[int]] = defaultdict(list)
    for eid, e in kvedges.items():
        out_adj[e.src].append(eid)
    for u in out_adj:
        out_adj[u].sort(key=lambda x: kvedges[x].score, reverse=True)
    return dict(out_adj)


# ============================================================
# 5) HippoRAG seed selection
# ============================================================
def get_seed_nodes(
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


# ============================================================
# 6) HippoRAG / HippoRAG2 graph propagation
# ============================================================
def personalized_pagerank_dense(
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


def build_entity_transition(num_nodes: int, kvedges: Dict[int, KVEdge]) -> np.ndarray:
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


def build_bipartite_transition(num_entities: int, kvedges: Dict[int, KVEdge]) -> Tuple[np.ndarray, Dict[int, int]]:
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


def run_hipporag_entity_ppr(
    seed_ids: List[int],
    seed_scores: np.ndarray,
    num_nodes: int,
    kvedges: Dict[int, KVEdge],
    alpha: float,
    max_iter: int,
) -> np.ndarray:
    trans = build_entity_transition(num_nodes, kvedges)
    tele = np.zeros((num_nodes,), dtype=np.float32)
    if seed_ids:
        vals = np.array([max(0.0, float(seed_scores[i])) + 1e-6 for i in seed_ids], dtype=np.float32)
        vals = vals / vals.sum()
        for sid, v in zip(seed_ids, vals):
            tele[sid] = float(v)
    elif num_nodes > 0:
        tele[:] = 1.0 / float(num_nodes)
    return personalized_pagerank_dense(trans, tele, alpha=alpha, max_iter=max_iter)


def run_hipporag2_bipartite_ppr(
    seed_ids: List[int],
    seed_scores: np.ndarray,
    num_nodes: int,
    kvedges: Dict[int, KVEdge],
    alpha: float,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    trans, edge_to_idx = build_bipartite_transition(num_nodes, kvedges)
    total = trans.shape[0]
    tele = np.zeros((total,), dtype=np.float32)
    if seed_ids:
        vals = np.array([max(0.0, float(seed_scores[i])) + 1e-6 for i in seed_ids], dtype=np.float32)
        vals = vals / vals.sum()
        for sid, v in zip(seed_ids, vals):
            tele[sid] = float(v)
    elif total > 0:
        tele[:num_nodes] = 1.0 / float(max(1, num_nodes))
    ppr = personalized_pagerank_dense(trans, tele, alpha=alpha, max_iter=max_iter)
    return ppr[:num_nodes], ppr[num_nodes:], edge_to_idx


# ============================================================
# 7) Readout from PPR to KV edges
# ============================================================
def is_attribute_edge(e: KVEdge, node_names: List[str]) -> bool:
    if (e.triple_type or '').upper() == 'ATTRIBUTE':
        return True
    dst_name = node_names[e.dst] if (0 <= e.dst < len(node_names)) else ''
    if any(ch.isdigit() for ch in dst_name):
        return True
    if len(dst_name.split()) <= 2 and dst_name and dst_name[0].islower():
        return True
    return False


def rank_edges_hipporag(
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
        if is_attribute_edge(e, node_names):
            score += leaf_bonus
        scored.append((score, eid))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [eid for _, eid in scored]


def select_top_edges_with_connectivity(
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


# ============================================================
# 8) DAG enforcement / sink constraint / export
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


def find_cycle_edges(nodes: Set[int], edges_kept: Set[int], kvedges: Dict[int, KVEdge]) -> List[int]:
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


def break_cycles_to_dag_kv(nodes: Set[int], edges_kept: Set[int], kvedges: Dict[int, KVEdge]) -> Set[int]:
    while True:
        order = topo_sort(nodes, edges_kept, kvedges)
        if order is not None:
            return edges_kept
        cyc = find_cycle_edges(nodes, edges_kept, kvedges)
        if not cyc:
            worst = min(list(edges_kept), key=lambda eid: kvedges[eid].score)
            edges_kept.remove(worst)
            continue
        worst = min(cyc, key=lambda eid: kvedges[eid].score)
        if worst in edges_kept:
            edges_kept.remove(worst)


def enforce_max_sinks_entity(
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

        order = topo_sort(kept_nodes, kept_kvedges, kvedges)
        if order is None:
            kept_kvedges = break_cycles_to_dag_kv(kept_nodes, kept_kvedges, kvedges)
            order = topo_sort(kept_nodes, kept_kvedges, kvedges) or list(kept_nodes)

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


def export_kv_nodes_and_adj(
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

def _value_matches_answer(value: str, answer: str) -> bool:
    vN = norm_match(value)
    aN = norm_match(answer)
    if not vN or not aN:
        return False
    return vN == aN or aN in vN or vN in aN


def _prune_to_reachable_from_seeds(
    seeds: Set[int],
    kept_nodes: Set[int],
    kept_kvedges: Set[int],
    kvedges: Dict[int, KVEdge],
) -> Tuple[Set[int], Set[int]]:
    out_adj_e: Dict[int, List[int]] = defaultdict(list)
    for eid in kept_kvedges:
        e = kvedges[eid]
        if e.src in kept_nodes and e.dst in kept_nodes:
            out_adj_e[e.src].append(eid)

    vis: Set[int] = set()
    stack = list(seeds)
    while stack:
        u = stack.pop()
        if u in vis or u not in kept_nodes:
            continue
        vis.add(u)
        for eid in out_adj_e.get(u, []):
            v = kvedges[eid].dst
            if v not in vis:
                stack.append(v)

    vis_edges = {
        eid for eid in kept_kvedges
        if kvedges[eid].src in vis and kvedges[eid].dst in vis
    }
    return vis, vis_edges


def answer_terminalization_entity(
    seeds: Set[int],
    kept_nodes: Set[int],
    kept_kvedges: Set[int],
    kvedges: Dict[int, KVEdge],
    answer: str,
) -> Tuple[Set[int], Set[int]]:
    if not answer or not kept_kvedges:
        return kept_nodes, kept_kvedges

    in_map: Dict[int, List[int]] = defaultdict(list)
    out_map: Dict[int, List[int]] = defaultdict(list)
    answer_in_map: Dict[int, List[int]] = defaultdict(list)

    for eid in kept_kvedges:
        e = kvedges[eid]
        in_map[e.dst].append(eid)
        out_map[e.src].append(eid)
        if _value_matches_answer(e.value, answer):
            answer_in_map[e.dst].append(eid)

    cand_nodes = [n for n in kept_nodes if answer_in_map.get(n)]
    if not cand_nodes:
        return kept_nodes, kept_kvedges

    # 如果有非 seed 的答案候选，优先选非 seed
    non_seed_cands = [n for n in cand_nodes if n not in seeds]
    if non_seed_cands:
        cand_nodes = non_seed_cands

    def cand_key(n: int):
        ans_best = max(kvedges[eid].score for eid in answer_in_map[n])
        in_sum = sum(max(0.0, kvedges[eid].score) for eid in in_map.get(n, []))
        out_sum = sum(max(0.0, kvedges[eid].score) for eid in out_map.get(n, []))
        out_cnt = len(out_map.get(n, []))
        # 越大越好：答案入边强、总入边强、当前更接近叶子
        return (ans_best, in_sum, -out_cnt, -out_sum)

    target = max(cand_nodes, key=cand_key)

    # 核心：删掉答案节点的所有出边，让它变成 sink
    for eid in list(out_map.get(target, [])):
        kept_kvedges.discard(eid)

    kept_nodes, kept_kvedges = _prune_to_reachable_from_seeds(
        seeds=seeds,
        kept_nodes=kept_nodes,
        kept_kvedges=kept_kvedges,
        kvedges=kvedges,
    )
    return kept_nodes, kept_kvedges

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
    oracle_stop_answer_recall = 0
    
    # --- new stats for sink-miss analysis ---
    miss_sink_but_in_graph = 0
    miss_due_to_answer_node_has_outgoing = 0
    miss_due_to_other_reason = 0

    # new: outgoing-edge type distribution on answer nodes of failed samples
    fail_answer_out_attr = 0
    fail_answer_out_rel = 0
    fail_answer_out_other = 0

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
            entity_ppr, memory_scores, edge_to_idx = run_hipporag2_bipartite_ppr(
                seed_ids=seed_ids,
                seed_scores=seed_scores,
                num_nodes=len(node_names),
                kvedges=kvedges,
                alpha=args.ppr_alpha,
                max_iter=args.ppr_iters,
            )
        else:
            entity_ppr = run_hipporag_entity_ppr(
                seed_ids=seed_ids,
                seed_scores=seed_scores,
                num_nodes=len(node_names),
                kvedges=kvedges,
                alpha=args.ppr_alpha,
                max_iter=args.ppr_iters,
            )
            memory_scores, edge_to_idx = None, None

        ranked_eids = rank_edges_hipporag(
            question=question,
            node_names=node_names,
            kvedges=kvedges,
            entity_ppr=entity_ppr,
            memory_scores=memory_scores,
            edge_to_idx=edge_to_idx,
            edge_query_weight=args.edge_query_weight,
            leaf_bonus=args.leaf_bonus,
        )

        kept_nodes, kept_kvedges = select_top_edges_with_connectivity(
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

        kept_kvedges = break_cycles_to_dag_kv(kept_nodes, kept_kvedges, kvedges)

        if args.max_sinks is not None and args.max_sinks > 0:
            kept_nodes, kept_kvedges = enforce_max_sinks_entity(
                seeds=seed_nodes,
                max_sinks=args.max_sinks,
                kept_nodes=kept_nodes,
                kept_kvedges=kept_kvedges,
                kvedges=kvedges,
            )

        if args.answer_terminalization:
            kept_nodes, kept_kvedges = answer_terminalization_entity(
                seeds=seeds,
                kept_nodes=kept_nodes,
                kept_kvedges=kept_kvedges,
                kvedges=kvedges,
                answer=answer,
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

        # choose which edge set to evaluate sink-answer recall on
        eval_kept_kvedges = set(kept_kvedges)
        if args.oracle_stop and answer_nodes_in_graph:
            for eid in list(eval_kept_kvedges):
                e = kvedges[eid]
                if e.src in answer_nodes_in_graph:
                    eval_kept_kvedges.remove(eid)

        # recompute sinks on evaluation graph
        outdeg = defaultdict(int)
        for eid in eval_kept_kvedges:
            e = kvedges[eid]
            outdeg[e.src] += 1
        sinks = [n for n in kept_nodes if outdeg.get(n, 0) == 0]

        answer_matched = False
        if ansN:
            for sink in sinks:
                for eid in eval_kept_kvedges:
                    e = kvedges[eid]
                    if e.dst == sink:
                        vN = norm_match(e.value)
                        if vN and (vN == ansN or ansN in vN or vN in ansN):
                            answer_matched = True
                            break
                if answer_matched:
                    break

        if answer_matched:
            if args.oracle_stop:
                oracle_stop_answer_recall += 1
            else:
                answer_recall += 1

        # answer recall on path
        answer_in_graph = len(answer_nodes_in_graph) > 0
        if answer_in_graph:
            none_sink_recall += 1

        if answer_in_graph and not answer_matched:
            miss_sink_but_in_graph += 1
            has_outgoing_answer_node = any(outdeg.get(n, 0) > 0 for n in answer_nodes_in_graph)

            if has_outgoing_answer_node:
                miss_due_to_answer_node_has_outgoing += 1

                # count outgoing edge types from answer nodes in failed samples
                for eid in kept_kvedges:
                    e = kvedges[eid]
                    if e.src in answer_nodes_in_graph:
                        t = norm_text(getattr(e, "triple_type", ""))
                        if t == "ATTRIBUTE":
                            fail_answer_out_attr += 1
                        elif t == "RELATION":
                            fail_answer_out_rel += 1
                        else:
                            fail_answer_out_other += 1
            else:
                miss_due_to_other_reason += 1
        
        # ----------------------------------------------------------

        kv_nodes, adj = export_kv_nodes_and_adj(kept_kvedges, kvedges, keep_score=args.keep_score)

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
        if args.oracle_stop:
            print(f"Oracle-stop answer recall: {oracle_stop_answer_recall / len(samples):.4f}")
        else:
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

        total_fail_answer_out = fail_answer_out_attr + fail_answer_out_rel + fail_answer_out_other
        if total_fail_answer_out > 0:
            print(f"  ATTRIBUTE ratio: {fail_answer_out_attr / total_fail_answer_out:.4f}")
            print(f"  RELATION  ratio: {fail_answer_out_rel / total_fail_answer_out:.4f}")
            print(f"  OTHER     ratio: {fail_answer_out_other / total_fail_answer_out:.4f}")
    return out_samples


# ============================================================
# 10) CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='all_triples dataset file (.json or .jsonl)')
    ap.add_argument('--output', required=True, help='output dataset path (.jsonl or .json)')
    ap.add_argument('--st_model', default='sentence-transformers/all-MiniLM-L6-v2', help='SentenceTransformer model name or local path')
    ap.add_argument('--batch_size', type=int, default=256, help='embedding batch size')
    ap.add_argument('--keep_score', action='store_true', help='keep similarity scores in output')
    ap.add_argument('--limit', type=int, default=None, help='limit number of samples')
    ap.add_argument('--supporting_only', action='store_true', help='only use triples from supporting_facts titles')

    ap.add_argument('--variant', choices=['hipporag', 'hipporag2'], default='hipporag', help='HippoRAG retrieval variant')
    ap.add_argument('--seed_top_m', type=int, default=8, help='number of query-linked seed entities')
    ap.add_argument('--ppr_alpha', type=float, default=0.15, help='PPR teleport probability')
    ap.add_argument('--ppr_iters', type=int, default=50, help='max iterations for PPR')
    ap.add_argument('--max_nodes', type=int, default=30, help='max entity nodes in exported subgraph')
    ap.add_argument('--max_edges', type=int, default=40, help='max KVEdges in exported subgraph')
    ap.add_argument('--max_sinks', type=int, default=3, help='max sinks in induced DAG')
    ap.add_argument('--mention_bonus', type=float, default=0.25, help='bonus for explicit entity mentions in question')
    ap.add_argument('--lexical_weight', type=float, default=0.20, help='weight for lexical overlap in seed linking')
    ap.add_argument('--edge_query_weight', type=float, default=0.10, help='small readout weight for question/edge lexical match')
    ap.add_argument('--leaf_bonus', type=float, default=0.08, help='bonus for value-like/attribute edges so answers prefer sinks')
    ap.add_argument('--pred_weight', type=float, default=0.0, help='optional relation similarity weight when building directed KV edges')
    ap.add_argument('--oracle_stop', action='store_true', help='oracle analysis only: remove outgoing edges from answer nodes before sink evaluation')
    ap.add_argument("--answer_terminalization", action="store_true",
                    help="force one matched answer node to become sink by dropping its outgoing edges")


    args = ap.parse_args()
    print(args)

    embedder = SentenceTransformer(args.st_model)
    samples = read_json_or_jsonl(args.input)
    print(f'Load {len(samples)} samples from {args.input}')

    out = create_dag(args, samples, embedder)
    if not out:
        print('No valid output.')
        return

    if args.output.endswith('.jsonl'):
        write_jsonl(args.output, out)
    elif args.output.endswith('.json'):
        write_json(args.output, out)
    else:
        raise ValueError(f'Unknown file format: {args.output}')

    print(f'[DONE] input={len(samples)}  output={len(out)}  saved_to={args.output}')


if __name__ == '__main__':
    main()
