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
# 1) Text normalization
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
    'after', 'before', 'over', 'under', 'between', 'through', 'during', 'without', 'within',
    'told', 'tell', 'series', 'book', 'books', 'story', 'stories'
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


def lexical_overlap(a: str, b: str) -> float:
    A = token_set(a)
    B = token_set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A | B))


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


def cosine_sim_vec(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


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
# 5) Build graph
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

        for kv_idx, kv in enumerate(tri.get('kv_lists', []) or []):
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
# 6) Utilities for pruning / DAG
# ============================================================
def is_attribute_edge(e: KVEdge, node_names: List[str]) -> bool:
    if (e.triple_type or '').upper() == 'ATTRIBUTE':
        return True
    dst_name = node_names[e.dst] if 0 <= e.dst < len(node_names) else ''
    if any(ch.isdigit() for ch in dst_name):
        return True
    if len(dst_name.split()) <= 2 and dst_name and dst_name[0].islower():
        return True
    return False


def relation_signature(e: KVEdge) -> str:
    return norm_match(e.relation)


def prune_edges_before_search(
    out_adj: Dict[int, List[int]],
    kvedges: Dict[int, KVEdge],
    node_names: List[str],
    max_attr_out_per_entity: int = 2,
    keep_best_edge_per_pair: bool = True,
) -> Tuple[Dict[int, List[int]], Dict[int, KVEdge]]:
    attr_by_src = defaultdict(list)
    rel_eids = []
    for eid, e in kvedges.items():
        if is_attribute_edge(e, node_names):
            attr_by_src[e.src].append(eid)
        else:
            rel_eids.append(eid)

    keep: Set[int] = set()
    for src, eids in attr_by_src.items():
        eids.sort(key=lambda x: kvedges[x].score, reverse=True)
        keep.update(eids[:max_attr_out_per_entity])
    keep.update(rel_eids)

    if keep_best_edge_per_pair:
        best: Dict[Tuple[int, int, str], int] = {}
        for eid in list(keep):
            e = kvedges[eid]
            if is_attribute_edge(e, node_names):
                continue
            key = (e.src, e.dst, relation_signature(e))
            cur = best.get(key)
            if cur is None or kvedges[eid].score > kvedges[cur].score:
                best[key] = eid
        new_keep = set()
        for eid in keep:
            e = kvedges[eid]
            if is_attribute_edge(e, node_names):
                new_keep.add(eid)
            else:
                key = (e.src, e.dst, relation_signature(e))
                if best.get(key) == eid:
                    new_keep.add(eid)
        keep = new_keep

    new_kvedges = {eid: kvedges[eid] for eid in keep}
    return rebuild_out_adj(new_kvedges), new_kvedges


def keep_one_direction_for_attribute_pairs(
    question: str,
    kvedges: Dict[int, KVEdge],
    node_names: List[str],
    embedder: SentenceTransformer,
    batch_size: int,
    eps_dir: float = 0.05,
) -> Set[int]:
    pair_map = defaultdict(list)
    for eid, e in kvedges.items():
        if is_attribute_edge(e, node_names):
            u, v = e.src, e.dst
            if u != v:
                pair_map[(min(u, v), max(u, v))].append(eid)

    q_emb = embed_texts(embedder, [question], batch_size=batch_size)[0]
    node_emb_cache: Dict[int, np.ndarray] = {}

    def sim_to_q(node_id: int) -> float:
        if node_id not in node_emb_cache:
            node_emb_cache[node_id] = embed_texts(embedder, [node_names[node_id]], batch_size=batch_size)[0]
        return cosine_sim_vec(q_emb, node_emb_cache[node_id])

    keep = set(kvedges.keys())
    for (a, b), eids in pair_map.items():
        ab = [eid for eid in eids if kvedges[eid].src == a and kvedges[eid].dst == b]
        ba = [eid for eid in eids if kvedges[eid].src == b and kvedges[eid].dst == a]
        if not ab or not ba:
            continue

        name_a, name_b = node_names[a], node_names[b]
        a_in_q = contains_mention(question, name_a)
        b_in_q = contains_mention(question, name_b)
        if a_in_q and not b_in_q:
            for eid in ba:
                keep.discard(eid)
            continue
        if b_in_q and not a_in_q:
            for eid in ab:
                keep.discard(eid)
            continue

        dir_score = sim_to_q(a) - sim_to_q(b)
        if dir_score > eps_dir:
            for eid in ba:
                keep.discard(eid)
        elif dir_score < -eps_dir:
            for eid in ab:
                keep.discard(eid)
        else:
            best_ab = max(ab, key=lambda eid: kvedges[eid].score)
            best_ba = max(ba, key=lambda eid: kvedges[eid].score)
            if kvedges[best_ab].score >= kvedges[best_ba].score:
                for eid in ba:
                    keep.discard(eid)
            else:
                for eid in ab:
                    keep.discard(eid)
    return keep


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
    return order if len(order) == len(nodes) else None


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
            if u in vis or u not in nodes:
                continue
            vis.add(u)
            for eid in out_adj.get(u, []):
                stack.append(kvedges[eid].dst)
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

        target = sorted(sinks, key=lambda n: best.get(n, -1e9))[0]
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


# ============================================================
# 7) IRCoT-style iterative retrieval on graph
# ============================================================
class EdgeIndex:
    def __init__(self, embedder: SentenceTransformer, batch_size: int, kvedges: Dict[int, KVEdge], node_names: List[str]):
        self.embedder = embedder
        self.batch_size = batch_size
        self.kvedges = kvedges
        self.node_names = node_names
        self.eids = sorted(kvedges.keys())
        self.eid_to_pos = {eid: i for i, eid in enumerate(self.eids)}
        self.key_texts = [kvedges[eid].key for eid in self.eids]
        self.rel_texts = [kvedges[eid].relation or kvedges[eid].key for eid in self.eids]
        self.src_texts = [kvedges[eid].src_name for eid in self.eids]
        self.dst_texts = [kvedges[eid].dst_name for eid in self.eids]
        self.key_emb = embed_texts(embedder, self.key_texts, batch_size)
        self.rel_emb = embed_texts(embedder, self.rel_texts, batch_size)
        self.src_emb = embed_texts(embedder, self.src_texts, batch_size)
        self.dst_emb = embed_texts(embedder, self.dst_texts, batch_size)

    def query_scores(self, query_text: str) -> Dict[int, float]:
        q = embed_texts(self.embedder, [query_text], self.batch_size)
        key = cosine_sim_matrix(q, self.key_emb).reshape(-1)
        rel = cosine_sim_matrix(q, self.rel_emb).reshape(-1)
        src = cosine_sim_matrix(q, self.src_emb).reshape(-1)
        dst = cosine_sim_matrix(q, self.dst_emb).reshape(-1)
        scores = 0.55 * key + 0.20 * rel + 0.15 * src + 0.10 * dst
        return {eid: float(scores[i]) for i, eid in enumerate(self.eids)}


def build_structured_thought(
    question: str,
    selected_eids: List[int],
    frontier_nodes: List[int],
    covered_tokens: Set[str],
    kvedges: Dict[int, KVEdge],
    node_names: List[str],
    max_frontier: int = 3,
) -> str:
    q_tokens = [t for t in normalize_lex(question).split() if t not in covered_tokens]
    q_hint = ' '.join(q_tokens[:8])
    frontier_names = [node_names[n] for n in frontier_nodes[:max_frontier] if 0 <= n < len(node_names)]

    recent_rels = []
    recent_vals = []
    for eid in selected_eids[-3:]:
        e = kvedges[eid]
        if e.relation:
            recent_rels.append(e.relation)
        recent_vals.append(e.dst_name)

    parts = []
    if frontier_names:
        parts.append('follow entities: ' + ' ; '.join(frontier_names))
    if recent_rels:
        parts.append('focus relations: ' + ' ; '.join(recent_rels[:3]))
    if recent_vals:
        parts.append('recent clues: ' + ' ; '.join(recent_vals[:3]))
    if q_hint:
        parts.append('unresolved question focus: ' + q_hint)
    return ' | '.join(parts) if parts else question


def choose_seed_edges(
    question: str,
    edge_index: EdgeIndex,
    kvedges: Dict[int, KVEdge],
    node_names: List[str],
    top_k: int,
    seed_relation_bonus: float,
) -> List[int]:
    q_scores = edge_index.query_scores(question)
    scored = []
    for eid, e in kvedges.items():
        val_bonus = 0.08 if is_attribute_edge(e, node_names) else 0.0
        mention_bonus = 0.10 if contains_mention(question, e.src_name) or contains_mention(question, e.dst_name) else 0.0
        rel_bonus = seed_relation_bonus * lexical_overlap(question, e.relation)
        score = 0.65 * e.score + 0.35 * q_scores[eid] + val_bonus + mention_bonus + rel_bonus
        scored.append((score, eid))
    scored.sort(reverse=True)
    return [eid for _, eid in scored[:max(1, top_k)]]


def rank_candidates_ir_cot(
    question: str,
    thought_query: str,
    selected_edges: Set[int],
    current_nodes: Set[int],
    frontier_nodes: Set[int],
    edge_index: EdgeIndex,
    kvedges: Dict[int, KVEdge],
    node_names: List[str],
    args,
) -> List[Tuple[float, int]]:
    q_scores = edge_index.query_scores(question)
    t_scores = edge_index.query_scores(thought_query)
    ranked: List[Tuple[float, int]] = []

    q_tok = token_set(question)
    selected_rels = {norm_match(kvedges[eid].relation) for eid in selected_edges}

    for eid, e in kvedges.items():
        if eid in selected_edges:
            continue

        # IRCoT adaptation: use current thought state to re-rank retrieval.
        connect_bonus = 0.0
        if e.src in frontier_nodes:
            connect_bonus += args.frontier_bonus
        elif e.src in current_nodes:
            connect_bonus += 0.6 * args.frontier_bonus
        if e.dst not in current_nodes:
            connect_bonus += args.expand_bonus

        mention_bonus = 0.0
        if contains_mention(question, e.src_name):
            mention_bonus += 0.08
        if contains_mention(question, e.dst_name):
            mention_bonus += 0.04

        attr_bonus = args.leaf_bonus if is_attribute_edge(e, node_names) else 0.0
        novelty_bonus = 0.0
        rel_norm = norm_match(e.relation)
        if rel_norm and rel_norm not in selected_rels:
            novelty_bonus += 0.05
        val_toks = token_set(e.value)
        if q_tok and val_toks and len(q_tok & val_toks) == 0:
            novelty_bonus += 0.02

        redundancy_penalty = 0.0
        if e.dst in current_nodes:
            redundancy_penalty += 0.06
        if e.src in current_nodes and e.dst in current_nodes:
            redundancy_penalty += 0.08

        score = (
            args.base_question_weight * q_scores[eid]
            + args.thought_weight * t_scores[eid]
            + args.orig_edge_weight * e.score
            + connect_bonus
            + mention_bonus
            + attr_bonus
            + novelty_bonus
            - redundancy_penalty
        )
        ranked.append((float(score), eid))

    ranked.sort(reverse=True)
    return ranked


def retrieve_iteratively_ir_cot(
    question: str,
    edge_index: EdgeIndex,
    kvedges: Dict[int, KVEdge],
    node_names: List[str],
    args,
) -> Tuple[Set[int], Set[int], Dict[str, Any]]:
    selected_eids: List[int] = choose_seed_edges(
        question=question,
        edge_index=edge_index,
        kvedges=kvedges,
        node_names=node_names,
        top_k=args.seed_top_m,
        seed_relation_bonus=args.seed_relation_bonus,
    )
    selected_set: Set[int] = set(selected_eids)
    current_nodes: Set[int] = set()
    for eid in selected_eids:
        current_nodes.add(kvedges[eid].src)
        current_nodes.add(kvedges[eid].dst)

    frontier_nodes: Set[int] = {kvedges[eid].dst for eid in selected_eids}
    covered_tokens: Set[str] = set()
    for eid in selected_eids:
        covered_tokens |= token_set(kvedges[eid].relation)
        covered_tokens |= token_set(kvedges[eid].value)

    thoughts: List[str] = []
    for step in range(max(1, args.max_steps)):
        if len(selected_set) >= args.max_edges or len(current_nodes) >= args.max_nodes:
            break

        ordered_frontier = sorted(frontier_nodes, key=lambda n: node_names[n])
        thought = build_structured_thought(
            question=question,
            selected_eids=selected_eids,
            frontier_nodes=ordered_frontier,
            covered_tokens=covered_tokens,
            kvedges=kvedges,
            node_names=node_names,
        )
        thoughts.append(thought)

        ranked = rank_candidates_ir_cot(
            question=question,
            thought_query=thought,
            selected_edges=selected_set,
            current_nodes=current_nodes,
            frontier_nodes=frontier_nodes,
            edge_index=edge_index,
            kvedges=kvedges,
            node_names=node_names,
            args=args,
        )

        added_this_step = 0
        next_frontier: Set[int] = set()
        used_src: Dict[int, int] = defaultdict(int)
        for _, eid in ranked:
            if len(selected_set) >= args.max_edges or len(current_nodes) >= args.max_nodes:
                break
            e = kvedges[eid]
            if used_src[e.src] >= args.beam_width:
                continue
            if e.src not in frontier_nodes and e.src not in current_nodes:
                continue
            if e.dst in current_nodes and e.src in current_nodes and added_this_step >= args.min_add_per_step:
                continue

            selected_set.add(eid)
            selected_eids.append(eid)
            used_src[e.src] += 1
            before_size = len(current_nodes)
            current_nodes.add(e.src)
            current_nodes.add(e.dst)
            next_frontier.add(e.dst)
            covered_tokens |= token_set(e.relation)
            covered_tokens |= token_set(e.value)
            added_this_step += 1

            if added_this_step >= args.edges_per_step and len(current_nodes) == before_size:
                break
            if added_this_step >= args.edges_per_step and len(current_nodes) >= before_size:
                break

        if added_this_step == 0:
            break

        frontier_nodes = next_frontier if next_frontier else frontier_nodes

    meta = {
        'num_retrieval_steps': len(thoughts),
        'thoughts': thoughts if args.keep_thought_trace else [],
        'num_seed_edges': len(selected_eids[:args.seed_top_m]),
    }
    return current_nodes, selected_set, meta


# ============================================================
# 8) Export
# ============================================================
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


def print_current_graph_kv(
    seeds: Set[int],
    kept_nodes: Set[int],
    kept_kvedges: Set[int],
    kvedges: Dict[int, KVEdge],
    node_names: Optional[List[str]] = None,
):
    print('\n' + '=' * 110)
    print('📊 Current DAG (Entity nodes + KVEdges)')
    print('=' * 110)

    def node_label(nid: int) -> str:
        if node_names is None:
            return str(nid)
        return f'{nid}:{node_names[nid]}' if 0 <= nid < len(node_names) else str(nid)

    out_adj = defaultdict(list)
    outdeg = {n: 0 for n in kept_nodes}
    for eid in kept_kvedges:
        e = kvedges[eid]
        if e.src in kept_nodes and e.dst in kept_nodes:
            out_adj[e.src].append(eid)
            outdeg[e.src] += 1
    sinks = [n for n in kept_nodes if outdeg.get(n, 0) == 0]
    print(f'Seeds: {[node_label(s) for s in sorted(seeds)]}')
    print(f'Total Nodes: {len(kept_nodes)} | Total Edges: {len(kept_kvedges)} | Sinks({len(sinks)}): {[node_label(x) for x in sorted(sinks)]}')

    order = topo_sort(kept_nodes, kept_kvedges, kvedges) or list(sorted(kept_nodes))
    for n in order:
        tags = []
        if n in seeds:
            tags.append('SEED')
        if n in sinks:
            tags.append('SINK')
        tag_str = ('[' + ']['.join(tags) + '] ') if tags else ''
        print(f'{tag_str}{node_label(n)}')
        for eid in sorted(out_adj.get(n, []), key=lambda eid: kvedges[eid].score, reverse=True):
            e = kvedges[eid]
            print(f'   └── ({e.score:.4f}) ({e.key}, {e.value}) → {node_label(e.dst)}')
    print('=' * 110 + '\n')


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

    for sample in tqdm(samples, desc='Create DAG (IRCoT-style)'):
        question = norm_text(sample.get('question', ''))
        answer = norm_text(sample.get('answer', ''))
        ansN = norm_match(answer)

        node_names, out_adj, kvedges = build_kvedge_graph(
            sample=sample,
            embedder=embedder,
            batch_size=args.batch_size,
            supporting_only=args.supporting_only,
            pred_weight=args.pred_weight,
        )

        if args.max_attr_out_per_entity is not None and args.max_attr_out_per_entity > 0:
            out_adj, kvedges = prune_edges_before_search(
                out_adj=out_adj,
                kvedges=kvedges,
                node_names=node_names,
                max_attr_out_per_entity=args.max_attr_out_per_entity,
                keep_best_edge_per_pair=True,
            )

        if not kvedges:
            sample['dag'] = {'kv_nodes': [], 'adj': [], 'meta': {'reason': 'no_kv_edges'}}
            out_samples.append(sample)
            continue

        if args.keep_one_attr_direction:
            kept_eids = keep_one_direction_for_attribute_pairs(
                question=question,
                kvedges=kvedges,
                node_names=node_names,
                embedder=embedder,
                batch_size=args.batch_size,
                eps_dir=args.eps_dir,
            )
            kvedges = {eid: e for eid, e in kvedges.items() if eid in kept_eids}
            out_adj = rebuild_out_adj(kvedges)
            if not kvedges:
                sample['dag'] = {'kv_nodes': [], 'adj': [], 'meta': {'reason': 'empty_after_attr_direction'}}
                out_samples.append(sample)
                continue

        for e in kvedges.values():
            vN = norm_match(e.value)
            if ansN and vN and (vN == ansN or ansN in vN or vN in ansN):
                graph_recall += 1
                break

        edge_index = EdgeIndex(embedder=embedder, batch_size=args.batch_size, kvedges=kvedges, node_names=node_names)
        kept_nodes, kept_kvedges, iter_meta = retrieve_iteratively_ir_cot(
            question=question,
            edge_index=edge_index,
            kvedges=kvedges,
            node_names=node_names,
            args=args,
        )

        if not kept_kvedges:
            sample['dag'] = {'kv_nodes': [], 'adj': [], 'meta': {'reason': 'iterative_retrieval_empty'}}
            out_samples.append(sample)
            continue

        seeds = {kvedges[eid].src for eid in list(kept_kvedges)[:min(len(kept_kvedges), args.seed_top_m)]}
        seeds |= {kvedges[eid].dst for eid in list(kept_kvedges)[:min(len(kept_kvedges), args.seed_top_m)]}

        kept_kvedges = break_cycles_to_dag_kv(kept_nodes, set(kept_kvedges), kvedges)
        if args.max_sinks is not None and args.max_sinks > 0:
            kept_nodes, kept_kvedges = enforce_max_sinks_entity(
                seeds=seeds,
                max_sinks=args.max_sinks,
                kept_nodes=kept_nodes,
                kept_kvedges=kept_kvedges,
                kvedges=kvedges,
            )

        if not kept_kvedges:
            sample['dag'] = {'kv_nodes': [], 'adj': [], 'meta': {'reason': 'empty_after_sink_prune'}}
            out_samples.append(sample)
            continue

        if args.verbose:
            print(f'Q: {question}')
            print(f'A: {answer}')
            print_current_graph_kv(seeds, kept_nodes, kept_kvedges, kvedges, node_names=node_names)
            if iter_meta.get('thoughts'):
                print('Thought trace:')
                for i, t in enumerate(iter_meta['thoughts']):
                    print(f'  [{i}] {t}')

        outdeg = defaultdict(int)
        for eid in kept_kvedges:
            outdeg[kvedges[eid].src] += 1
        sinks = [n for n in kept_nodes if outdeg.get(n, 0) == 0]

        answer_matched = False
        if ansN:
            for sink in sinks:
                for eid in kept_kvedges:
                    e = kvedges[eid]
                    if e.dst == sink:
                        vN = norm_match(e.value)
                        if vN and (vN == ansN or ansN in vN or vN in ansN):
                            answer_matched = True
                            break
                if answer_matched:
                    break
        if answer_matched:
            answer_recall += 1

        for eid in kept_kvedges:
            e = kvedges[eid]
            vN = norm_match(e.value)
            if ansN and vN and (vN == ansN or ansN in vN or vN in ansN):
                none_sink_recall += 1
                break

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
            'retriever': 'IRCoT-style-graph',
            **iter_meta,
        }
        sample['dag'] = {'kv_nodes': kv_nodes, 'adj': adj, 'meta': meta}
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
    ap.add_argument('--input', required=True, help='all_triples dataset file (.json or .jsonl)')
    ap.add_argument('--output', required=True, help='output dataset path (.jsonl or .json)')
    ap.add_argument('--st_model', default='sentence-transformers/all-MiniLM-L6-v2', help='SentenceTransformer model name or local path')
    ap.add_argument('--batch_size', type=int, default=256, help='embedding batch size')
    ap.add_argument('--keep_score', action='store_true', help='keep similarity scores in output')
    ap.add_argument('--limit', type=int, default=None, help='limit number of samples')
    ap.add_argument('--supporting_only', action='store_true', help='only use triples from supporting_facts titles')
    ap.add_argument('--verbose', action='store_true', help='verbose debug')

    ap.add_argument('--pred_weight', type=float, default=0.0, help='weight for relation score sim(question, relation)')
    ap.add_argument('--keep_one_attr_direction', action='store_true', help='keep only one direction for ATTRIBUTE pairs')
    ap.add_argument('--eps_dir', type=float, default=0.05, help='direction confidence threshold')
    ap.add_argument('--max_attr_out_per_entity', type=int, default=None, help='cap attribute out edges per entity')

    # IRCoT-style knobs
    ap.add_argument('--seed_top_m', type=int, default=6, help='number of seed edges in step-0 retrieval')
    ap.add_argument('--max_steps', type=int, default=3, help='iterative retrieval steps')
    ap.add_argument('--edges_per_step', type=int, default=4, help='how many edges to add per iterative step')
    ap.add_argument('--min_add_per_step', type=int, default=1, help='minimum new edges before allowing purely internal edges')
    ap.add_argument('--beam_width', type=int, default=2, help='per-source expansion width in each step')
    ap.add_argument('--max_nodes', type=int, default=30, help='max entity nodes in final subgraph')
    ap.add_argument('--max_edges', type=int, default=40, help='max KVEdges in final subgraph')
    ap.add_argument('--max_sinks', type=int, default=3, help='max sinks (out-degree=0) on induced entity DAG')

    ap.add_argument('--base_question_weight', type=float, default=0.45, help='question-driven retrieval weight')
    ap.add_argument('--thought_weight', type=float, default=0.35, help='thought-guided retrieval weight')
    ap.add_argument('--orig_edge_weight', type=float, default=0.20, help='original edge score weight')
    ap.add_argument('--frontier_bonus', type=float, default=0.20, help='bonus for edges leaving the current frontier')
    ap.add_argument('--expand_bonus', type=float, default=0.12, help='bonus for discovering new destination nodes')
    ap.add_argument('--leaf_bonus', type=float, default=0.08, help='bonus for attribute/value-like edges to improve sink answer recall')
    ap.add_argument('--seed_relation_bonus', type=float, default=0.12, help='extra lexical relation bonus during seed retrieval')
    ap.add_argument('--keep_thought_trace', action='store_true', help='store structured thought trace in dag.meta')

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
