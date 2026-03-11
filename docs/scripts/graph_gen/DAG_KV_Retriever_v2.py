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


# --------------------------
# IO
# --------------------------
def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return obj["data"]
    raise ValueError("Unsupported JSON root format. Expect list or {data:[...]}.")


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------------
# Normalization
# --------------------------
_SPACE_RE = re.compile(r"\s+")
_ZW_RE = re.compile(r"[\u200b\u200c\u200d\uFEFF]")


def _fix_unbalanced_parentheses(s: str) -> str:
    if not s:
        return s
    l = s.count("(")
    r = s.count(")")
    if l > r:
        s = s + (")" * (l - r))
    elif r > l:
        extra = r - l
        while extra > 0 and s.endswith(")"):
            s = s[:-1]
            extra -= 1
    return s


def norm_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = _ZW_RE.sub("", s)
    s = s.strip()
    s = _SPACE_RE.sub(" ", s)
    s = _fix_unbalanced_parentheses(s)
    return s


def norm_match(x: Any) -> str:
    return norm_text(x).lower()

# entity normalization
_PAREN_CONTENT_RE = re.compile(r"\([^)]*\)")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")

def _lex_tokens(s: str) -> Set[str]:
    s = norm_text(s).lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    if not s:
        return set()
    return set(s.split())

def token_jaccard(a: str, b: str) -> float:
    A = _lex_tokens(a)
    B = _lex_tokens(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return float(inter) / float(union) if union > 0 else 0.0

def entity_key(name: str) -> str:
    """
    Aggressive but cheap canonical key for entity aliasing:
    - remove (...) content
    - lowercase
    - strip punctuation
    - collapse spaces
    """
    s = norm_text(name)
    s = _PAREN_CONTENT_RE.sub("", s)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s

def entity_alias_keys(name: str) -> List[str]:
    """
    Return candidate alias keys (ordered) for merge.
    Key idea: if a name has 3+ tokens, also try dropping the last token
    (e.g., "shirley temple black" -> "shirley temple").
    """
    k = entity_key(name)
    if not k:
        return [k]
    toks = k.split()
    keys = [k]

    # drop last token for 3+ tokens: "First Middle Last" -> "First Middle"
    if len(toks) >= 3:
        keys.append(" ".join(toks[:-1]))

    # (optional) if endswith common suffixes, drop them too
    # e.g., "jr", "sr", "iii" etc.
    suffixes = {"jr", "sr", "ii", "iii", "iv"}
    if toks and toks[-1] in suffixes and len(toks) >= 2:
        keys.append(" ".join(toks[:-1]))

    # dedup keep order
    out = []
    seen = set()
    for kk in keys:
        if kk not in seen:
            out.append(kk)
            seen.add(kk)
    return out

# --------------------------
# Embedding helpers
# --------------------------
def embed_texts(embedder: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    emb = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine -> dot
    )
    if isinstance(emb, list):
        emb = np.array(emb)
    return emb.astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (m,d), b:(n,d) both normalized
    return a @ b.T


# --------------------------
# KVEdge = Beam / DAG unit
# --------------------------
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
        print(f"【{self.kid}】 <{self.triple_s}, {self.relation}, {self.triple_o}> ({self.key} [{self.src}], {self.value} [{self.dst}]) {self.score:.4f}")
        # print(f"KVEdge({self.kid}) {self.src_name}({self.src}) -> {self.dst_name}({self.dst}) {self.key}={self.value} ({self.score:.4f})")


# --------------------------
# Triple iterator (optional supporting_only)
# --------------------------
def iter_triples(sample: Dict[str, Any], supporting_only: bool) -> Iterable[Tuple[str, Dict[str, Any]]]:
    ctx = sample.get("context", []) or []
    supporting_titles: Optional[Set[str]] = None
    if supporting_only:
        sf = sample.get("supporting_facts", []) or []
        supporting_titles = {t for t, _ in sf if isinstance(t, str)}
    for para in ctx:
        title = norm_text(para.get("title", ""))
        if supporting_titles is not None and title not in supporting_titles:
            continue
        for tri in (para.get("triple_list", []) or []):
            yield title, tri


# --------------------------
# Build entity nodes + KVEdge graph
# --------------------------
# def get_or_add_node(name: str, node_map: Dict[str, int], node_names: List[str]) -> int:
#     nn = norm_text(name)
#     key = norm_match(nn)
#     if key not in node_map:
#         node_map[key] = len(node_names)
#         node_names.append(nn)
#     return node_map[key]


def get_or_add_node(name: str, node_map: Dict[str, int], node_names: List[str]) -> int:
    nn = norm_text(name)

    # 1) try alias keys (canonicalization / merge)
    for k in entity_alias_keys(nn):
        if k in node_map:
            return node_map[k]

    # 2) create new node
    nid = len(node_names)
    node_names.append(nn)

    # 3) register all alias keys -> same id
    for k in entity_alias_keys(nn):
        if k:
            node_map[k] = nid
    return nid


# --------------------------
# Pre-Prun edges
# --------------------------

def is_attribute_edge(e: KVEdge, node_names: List[str]) -> bool:
    """
    Heuristic: if dst node looks like a literal/value (short, no title-case entity feel),
    or triple type is ATTRIBUTE if available.
    """
    t = getattr(e, "triple", None)
    if isinstance(t, dict) and t.get("type", "").upper() == "ATTRIBUTE":
        return True
    dst_name = node_names[e.dst] if (0 <= e.dst < len(node_names)) else ""
    # crude literal heuristic: contains digits or is very short or starts lowercase
    if any(ch.isdigit() for ch in dst_name):
        return True
    if len(dst_name.split()) <= 2 and dst_name and dst_name[0].islower():
        return True
    # you can tighten/loosen this based on your dataset
    return False


def relation_signature(e: KVEdge) -> str:
    """
    Try to build a stable relation signature for grouping the 4 edges of one relation triple.
    Prefer triple['description_type'] if present, else fall back to normalized key_string.
    """
    t = getattr(e, "triple", None)
    if isinstance(t, dict):
        dt = t.get("description_type", "") or t.get("relation", "")
        if dt:
            return norm_match(norm_text(dt))
    # fallback: key_string (remove entity names is better but keep minimal)
    return norm_match(norm_text(getattr(e, "key_string", "")))


def prune_edges_before_search(
    question: str,
    out_adj: Dict[int, List[int]],
    kvedges: Dict[int, KVEdge],
    node_names: List[str],
    max_attr_out_per_entity: int = 2,
    keep_best_edge_per_pair: bool = True,
) -> Tuple[Dict[int, List[int]], Dict[int, KVEdge]]:
    """
    Return pruned (out_adj, kvedges).
    - Keep at most K attribute edges per src
    - Keep only best relation edge per (src, dst, relation_signature)
    """
    # 1) attribute cap per src
    attr_by_src = defaultdict(list)
    rel_edges = []

    for eid, e in kvedges.items():
        if is_attribute_edge(e, node_names):
            attr_by_src[e.src].append(eid)
        else:
            rel_edges.append(eid)

    keep = set()

    for src, eids in attr_by_src.items():
        eids.sort(key=lambda x: kvedges[x].score, reverse=True)
        keep.update(eids[:max_attr_out_per_entity])

    # keep all relation edges for now
    keep.update(rel_edges)

    # 2) dedup relation edges
    if keep_best_edge_per_pair:
        best = {}
        for eid in list(keep):
            e = kvedges[eid]
            if is_attribute_edge(e, node_names):
                continue
            key = (e.src, e.dst, relation_signature(e))
            cur = best.get(key)
            if cur is None or kvedges[eid].score > kvedges[cur].score:
                best[key] = eid

        # remove other relation edges in same group
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

    # 3) rebuild kvedges/out_adj
    new_kvedges = {eid: kvedges[eid] for eid in keep}

    new_out_adj = defaultdict(list)
    for eid, e in new_kvedges.items():
        new_out_adj[e.src].append(eid)
    # keep deterministic ordering
    for src in new_out_adj:
        new_out_adj[src].sort(key=lambda x: new_kvedges[x].score, reverse=True)

    return dict(new_out_adj), new_kvedges



def infer_kv_direction(
    s: str, o: str, key: str, value: str
) -> Tuple[str, str]:
    """
    Return ("forward"|"backward"|"unknown", reason)
    forward: S -> O
    backward: O -> S
    """
    sN = norm_match(s)
    oN = norm_match(o)
    vN = norm_match(value)
    kN = norm_match(key)

    # strongest signal: value equals S or O
    if vN == oN:
        return "forward", "value==O"
    if vN == sN:
        return "backward", "value==S"

    # fallback: key mentions S/O
    s_in = sN and (sN in kN)
    o_in = oN and (oN in kN)
    if s_in and not o_in:
        return "forward", "key_has_S"
    if o_in and not s_in:
        return "backward", "key_has_O"

    # if both mentioned or neither, unknown; default to forward (safer for chain expansion)
    return "unknown", "ambiguous"


def build_kvedge_graph(
    sample: Dict[str, Any],
    embedder: SentenceTransformer,
    batch_size: int,
    supporting_only: bool,
    keep_score: bool,
    pred_weight: float = 0.0,
) -> Tuple[List[str], Dict[int, List[int]], Dict[int, KVEdge]]:
    """
    Returns:
      node_names: entity id -> display name
      out_adj: src_id -> [kvedge_ids...]
      kvedges: kvid -> KVEdge
    """
    question = norm_text(sample.get("question", ""))
    q_emb = embed_texts(embedder, [question], batch_size=batch_size)[0:1]

    node_map: Dict[str, int] = {}
    node_names: List[str] = []
    kvedges: Dict[int, KVEdge] = {}
    out_adj: Dict[int, List[int]] = {}

    # collect all kv keys first for embedding
    kv_records: List[Tuple[str, str, str, str, str, str, str, int]] = []
    score_texts: List[str] = []
    seen: Set[Tuple[str, str]] = set()
    # (title, triple_type, relation, S, O, key, value, kv_idx)

    for title, tri in iter_triples(sample, supporting_only=supporting_only):
        ttype = norm_text(tri.get("type", ""))
        rel = norm_text(tri.get("description_type", ""))
        s = norm_text(tri.get("name", ""))
        o = norm_text(tri.get("description", ""))
        if not s or not o:
            continue
        kvs = tri.get("kv_lists", []) or []
        for kv_idx, kv in enumerate(kvs):
            key = norm_text(kv.get("key_string", ""))
            value = norm_text(kv.get("value_string", ""))
            if not key or not value:
                continue
            sig = (key, value)
            if sig in seen:
                continue
            seen.add(sig)
            kv_records.append((title, ttype, rel, s, o, key, value, kv_idx))
            score_texts.append(f"{key} | {value}")


    if not kv_records:
        return node_names, out_adj, kvedges

    keys = [r[5] for r in kv_records]
    key_emb = embed_texts(embedder, keys, batch_size=batch_size)
    sims = cosine_sim(q_emb, key_emb).reshape(-1)  # question-key similarity

    rels = [r[2] for r in kv_records]
    rel_emb = embed_texts(embedder, rels, batch_size=batch_size)
    rel_sims = cosine_sim(q_emb, rel_emb).reshape(-1) 


    kid = 0
    for i, (title, ttype, rel, s, o, key, value, kv_idx) in enumerate(kv_records):
        direction, _reason = infer_kv_direction(s, o, key, value)
        if direction == "backward":
            src_name, dst_name = o, s
        else:
            # forward or unknown -> forward by default
            src_name, dst_name = s, o

        src = get_or_add_node(src_name, node_map, node_names)
        dst = get_or_add_node(dst_name, node_map, node_names)

        score = float(sims[i]) + pred_weight * float(rel_sims[i])
        e = KVEdge(
            kid=kid,
            src=src,
            dst=dst,
            src_name=node_names[src],
            dst_name=node_names[dst],
            key=key,
            value=value,
            score=score if keep_score else score,  # still need for pruning; keep_score only affects output fields
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

    return node_names, out_adj, kvedges


# --------------------------
# Beam prune on KVEdge graph
# --------------------------
def pick_seed_nodes_from_kvedges(kvedges: Dict[int, KVEdge], top_m: int) -> Set[int]:
    ranked = sorted(kvedges.values(), key=lambda e: e.score, reverse=True)[: max(1, top_m)]
    seeds: Set[int] = set()
    for e in ranked:
        seeds.add(e.src)
        seeds.add(e.dst)
    return seeds


def prune_with_beam_kv(
    seeds: Set[int],
    out_adj: Dict[int, List[int]],
    kvedges: Dict[int, KVEdge],
    max_hops: int,
    beam_width: int,
    max_nodes: int,
    max_kvedges: int,
) -> Tuple[Set[int], Set[int]]:
    kept_nodes: Set[int] = set(seeds)
    kept_edges: Set[int] = set()

    frontier: List[int] = list(seeds)
    seen_frontier: Set[int] = set(frontier)

    for _hop in range(max_hops):
        if len(kept_edges) >= max_kvedges or len(kept_nodes) >= max_nodes:
            break
        new_frontier: List[int] = []
        new_seen: Set[int] = set()

        for u in frontier:
            eids = out_adj.get(u, [])
            if not eids:
                continue
            # sort outgoing KVEdges by score desc and pick top beam_width
            eids_sorted = sorted(eids, key=lambda eid: kvedges[eid].score, reverse=True)
            for eid in eids_sorted[: max(1, beam_width)]:
                if len(kept_edges) >= max_kvedges:
                    break
                e = kvedges[eid]
                kept_edges.add(eid)
                kept_nodes.add(e.src)
                kept_nodes.add(e.dst)

                if e.dst not in seen_frontier and e.dst not in new_seen:
                    new_frontier.append(e.dst)
                    new_seen.add(e.dst)

                if len(kept_nodes) >= max_nodes:
                    break

            if len(kept_edges) >= max_kvedges or len(kept_nodes) >= max_nodes:
                break

        frontier = new_frontier
        seen_frontier |= new_seen
        if not frontier:
            break

    return kept_nodes, kept_edges


# --------------------------
# Seed-edge bidirectional beam prune (NEW)
# --------------------------
def build_in_adj_from_kvedges(kvedges: Dict[int, KVEdge]) -> Dict[int, List[int]]:
    """dst_id -> [edge_ids...]"""
    in_adj: Dict[int, List[int]] = {}
    for eid, e in kvedges.items():
        in_adj.setdefault(e.dst, []).append(eid)
    return in_adj


def _would_create_cycle(src: int, dst: int, kept_out: Dict[int, Set[int]]) -> bool:
    """
    If we add edge src->dst into current kept graph, it creates a cycle iff dst can reach src.
    kept_out: u -> {v1,v2,...} adjacency of kept edges (entity-level)
    """
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


def prune_with_seededge_bidirectional_beam(
    kvedges: Dict[int, KVEdge],
    out_adj: Dict[int, List[int]],
    max_hops: int,
    beam_width: int,
    max_nodes: int,
    max_kvedges: int,
    num_seed_edges: int,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """
    Returns: (seeds, kept_nodes, kept_edges)
      - seeds: entity ids touched by seed edges (for later enforce_max_sinks_entity)
      - kept_nodes/kept_edges: induced subgraph
    Strategy:
      1) rank KVEdges by score, take top num_seed_edges as seed edges
      2) for each seed edge e: do two directional beam searches
           - backward: expand incoming edges into e.src   (.. -> src)
           - forward : expand outgoing edges from e.dst   (dst -> ..)
         each step picks top beam_width edges by score among candidates at frontier,
         skip any edge that would create a cycle w.r.t. current kept graph
      3) keep adding edges until limits reached
    """
    if not kvedges:
        return set(), set(), set()

    ranked_edges = sorted(kvedges.values(), key=lambda e: e.score, reverse=True)
    seed_edges = ranked_edges[: max(1, min(num_seed_edges, len(ranked_edges)))]

    in_adj = build_in_adj_from_kvedges(kvedges)

    kept_edges: Set[int] = set()
    kept_nodes: Set[int] = set()

    # maintain entity-level adjacency of kept edges for fast cycle test
    kept_out: Dict[int, Set[int]] = {}

    def try_add_edge(eid: int) -> bool:
        if eid in kept_edges:
            return True
        e = kvedges[eid]
        if _would_create_cycle(e.src, e.dst, kept_out):
            return False
        kept_edges.add(eid)
        kept_nodes.add(e.src)
        kept_nodes.add(e.dst)
        kept_out.setdefault(e.src, set()).add(e.dst)
        return True

    # init with seed edges
    seeds: Set[int] = set()
    for e in seed_edges:
        if len(kept_edges) >= max_kvedges:
            break
        ok = try_add_edge(e.kid)
        if ok:
            seeds.add(e.src)
            seeds.add(e.dst)

    # directional beam expansion around each seed edge
    def expand_backward(start_node: int):
        # beam items: (cum_score, current_node)
        beam = [(0.0, start_node)]
        for _ in range(max_hops):
            if len(kept_edges) >= max_kvedges or len(kept_nodes) >= max_nodes:
                break
            cand: List[Tuple[float, int, int]] = []  # (new_cum, next_node, eid) where eid: next_node -> cur
            for cum, cur in beam:
                for eid in in_adj.get(cur, []):
                    e = kvedges[eid]
                    nxt = e.src
                    cand.append((cum + e.score, nxt, eid))
            if not cand:
                break
            cand.sort(key=lambda x: x[0], reverse=True)
            new_beam = []
            used = 0
            for new_cum, nxt, eid in cand:
                if used >= max(1, beam_width):
                    break
                if len(kept_edges) >= max_kvedges or len(kept_nodes) >= max_nodes:
                    break
                if try_add_edge(eid):
                    new_beam.append((new_cum, nxt))
                    used += 1
            beam = new_beam
            if not beam:
                break

    def expand_forward(start_node: int):
        beam = [(0.0, start_node)]
        for _ in range(max_hops):
            if len(kept_edges) >= max_kvedges or len(kept_nodes) >= max_nodes:
                break
            cand: List[Tuple[float, int, int]] = []  # (new_cum, next_node, eid) where eid: cur -> next_node
            for cum, cur in beam:
                for eid in out_adj.get(cur, []):
                    e = kvedges[eid]
                    nxt = e.dst
                    cand.append((cum + e.score, nxt, eid))
            if not cand:
                break
            cand.sort(key=lambda x: x[0], reverse=True)
            new_beam = []
            used = 0
            for new_cum, nxt, eid in cand:
                if used >= max(1, beam_width):
                    break
                if len(kept_edges) >= max_kvedges or len(kept_nodes) >= max_nodes:
                    break
                if try_add_edge(eid):
                    new_beam.append((new_cum, nxt))
                    used += 1
            beam = new_beam
            if not beam:
                break

    for e in seed_edges:
        if len(kept_edges) >= max_kvedges or len(kept_nodes) >= max_nodes:
            break
        # backward: into src
        expand_backward(e.src)
        if len(kept_edges) >= max_kvedges or len(kept_nodes) >= max_nodes:
            break
        # forward: out of dst
        expand_forward(e.dst)

    return seeds, kept_nodes, kept_edges

# --------------------------
# Break cycles on KVEdges (ensure DAG)
# --------------------------
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
    out_adj: Dict[int, List[int]] = {}
    for eid in edges_kept:
        e = kvedges[eid]
        if e.src in nodes and e.dst in nodes:
            out_adj.setdefault(e.src, []).append(eid)

    state = {n: 0 for n in nodes}  # 0 un, 1 vis, 2 done
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
                # back-edge u->v (cycle)
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
            # fallback: remove global worst
            worst = min(list(edges_kept), key=lambda eid: kvedges[eid].score)
            edges_kept.remove(worst)
            continue
        worst = min(cyc, key=lambda eid: kvedges[eid].score)
        if worst in edges_kept:
            edges_kept.remove(worst)


# --------------------------
# Enforce max sinks on entity graph induced by kept KVEdges
# (optional, default on)
# --------------------------
def enforce_max_sinks_entity(
    seeds: Set[int],
    max_sinks: int,
    kept_nodes: Set[int],
    kept_kvedges: Set[int],
    kvedges: Dict[int, KVEdge],
) -> Tuple[Set[int], Set[int]]:
    if max_sinks <= 0:
        return kept_nodes, kept_kvedges

    def compute_outdeg(nodes: Set[int], edges_set: Set[int]) -> Dict[int, int]:
        outdeg = {n: 0 for n in nodes}
        for eid in edges_set:
            e = kvedges[eid]
            if e.src in nodes and e.dst in nodes:
                outdeg[e.src] += 1
        return outdeg

    # keep only reachable from seeds
    def reachable(nodes: Set[int], edges_set: Set[int]) -> Tuple[Set[int], Set[int]]:
        out_adj: Dict[int, List[int]] = {}
        for eid in edges_set:
            e = kvedges[eid]
            out_adj.setdefault(e.src, []).append(eid)
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
        outdeg = compute_outdeg(kept_nodes, kept_kvedges)
        sinks = [n for n in kept_nodes if outdeg.get(n, 0) == 0]
        if len(sinks) <= max_sinks:
            break

        # simple pruning: remove the lowest-score incoming edge to the weakest sink (by best path score approximation)
        # We approximate "support" of sink by max incoming edge score along any path: DP on topo order.
        order = topo_sort(kept_nodes, kept_kvedges, kvedges)
        if order is None:
            kept_kvedges = break_cycles_to_dag_kv(kept_nodes, kept_kvedges, kvedges)
            order = topo_sort(kept_nodes, kept_kvedges, kvedges) or list(kept_nodes)

        best = {n: -1e9 for n in kept_nodes}
        for s in seeds:
            if s in kept_nodes:
                best[s] = 0.0
        # adjacency
        out_adj: Dict[int, List[int]] = {}
        indeg_edges: Dict[int, List[int]] = {n: [] for n in kept_nodes}
        for eid in kept_kvedges:
            e = kvedges[eid]
            out_adj.setdefault(e.src, []).append(eid)
            indeg_edges.setdefault(e.dst, []).append(eid)

        for u in order:
            if best[u] <= -1e8:
                continue
            for eid in out_adj.get(u, []):
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
            # remove weakest incoming edge
            worst_in = min(ins, key=lambda eid: kvedges[eid].score)
            if worst_in in kept_kvedges:
                kept_kvedges.remove(worst_in)

        kept_nodes, kept_kvedges = reachable(kept_nodes, kept_kvedges)
        if not kept_kvedges:
            break

    return kept_nodes, kept_kvedges


# --------------------------
# Export: KVEdges as nodes + adj between KVEdges
# --------------------------
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
            "key": e.key,
            "value": e.value,
            "src_entity": e.src_name,
            "dst_entity": e.dst_name,
            "title": e.title,
            "triple_type": e.triple_type,
            "relation": e.relation,
            "kv_idx": e.kv_idx,
        }
        if keep_score:
            obj["score"] = float(e.score)
        kv_nodes.append(obj)

    n = len(kv_nodes)
    adj = [[0] * n for _ in range(n)]
    # chain adjacency: edge_i.dst == edge_j.src
    for eid_i in kept_list:
        i = idx_map[eid_i]
        dst_i = kvedges[eid_i].dst
        for eid_j in kept_list:
            j = idx_map[eid_j]
            if i == j:
                continue
            if kvedges[eid_j].src == dst_i:
                adj[i][j] = 1
    return kv_nodes, adj

def _short(s: str, n: int = 80) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else (s[: n - 3] + "...")

def print_current_graph_kv(
    seeds: Set[int],
    kept_nodes: Set[int],
    kept_kvedges: Set[int],
    kvedges: Dict[int, "KVEdge"],
    node_names: Optional[List[str]] = None,   # 可选：传入的话会同时打印实体名
    max_out_edges_per_node: int = 20,         # 每个点最多打印多少条出边（按score降序）
    max_str_len: int = 80,                    # key/value截断长度
):
    print("\n" + "=" * 110)
    print("📊 Current DAG (Entity nodes + KVEdges)  -- show (key_string, value_string)")
    print("=" * 110)

    def node_label(nid: int) -> str:
        if node_names is None:
            return str(nid)
        if 0 <= nid < len(node_names):
            return f"{nid}:{_short(node_names[nid], 60)}"
        return str(nid)

    # build adjacency (entity -> outgoing edge ids)
    out_adj = defaultdict(list)
    outdeg = {n: 0 for n in kept_nodes}
    for eid in kept_kvedges:
        e = kvedges[eid]
        if e.src in kept_nodes and e.dst in kept_nodes:
            out_adj[e.src].append(eid)
            outdeg[e.src] += 1

    sinks = [n for n in kept_nodes if outdeg.get(n, 0) == 0]

    print(f"\nSeeds: {[node_label(s) for s in sorted(seeds)]}")
    print(f"Total Nodes: {len(kept_nodes)} | Total Edges: {len(kept_kvedges)} | Sinks({len(sinks)}): {[node_label(x) for x in sorted(sinks)]}\n")

    # topo order if possible
    try:
        order = topo_sort(kept_nodes, kept_kvedges, kvedges)  # 你文件里已有 topo_sort
        if order is None:
            order = list(sorted(kept_nodes))
    except Exception:
        order = list(sorted(kept_nodes))

    for n in order:
        tags = []
        if n in seeds:
            tags.append("SEED")
        if n in sinks:
            tags.append("SINK")
        tag_str = ("[" + "][".join(tags) + "] ") if tags else ""

        print(f"{tag_str}{node_label(n)}")

        eids = out_adj.get(n, [])
        if not eids:
            continue

        eids = sorted(eids, key=lambda eid: kvedges[eid].score, reverse=True)
        if len(eids) > max_out_edges_per_node:
            shown = eids[:max_out_edges_per_node]
            hidden = len(eids) - len(shown)
        else:
            shown = eids
            hidden = 0

        for eid in shown:
            e = kvedges[eid]
            k = _short(getattr(e, "key", ""), max_str_len)
            v = _short(getattr(e, "value", ""), max_str_len)
            print(f"   └── ({e.score:.4f}) ({k}, {v}) → {node_label(e.dst)}")

        if hidden > 0:
            print(f"   └── ... ({hidden} more outgoing edges hidden)")

    print("=" * 110 + "\n")



# --------------------------
# Main create_dag
# --------------------------
def create_dag(args, samples: List[Dict[str, Any]], embedder: SentenceTransformer) -> List[Dict[str, Any]]:
    if args.limit is not None and args.limit > 0:
        samples = samples[:args.limit]

    # debug
    if args.verbose:
        samples = [samples[2]]
    out_samples: List[Dict[str, Any]] = []
    answer_recall = 0
    graph_recall = 0

    for sample in tqdm(samples, desc="Create DAG"):
        question = norm_text(sample.get("question", ""))
        answer = norm_text(sample.get("answer", ""))

        node_names, out_adj, kvedges = build_kvedge_graph(
            sample=sample,
            embedder=embedder,
            batch_size=args.batch_size,
            supporting_only=args.supporting_only,
            keep_score=args.keep_score,
            pred_weight=args.pred_weight,
        )
        out_adj, kvedges = prune_edges_before_search(
            question=question,
            out_adj=out_adj,
            kvedges=kvedges,
            node_names=node_names,
            max_attr_out_per_entity=2,
            keep_best_edge_per_pair=True,
        )

        for e in kvedges.values():
            if answer in e.value or e.value in answer:
                graph_recall += 1
                break

        if args.verbose:
            print(f"Q: {question}")
            print(f"A: {answer}")
            # 按照score排序
            ranked = sorted(kvedges.values(), key=lambda e: e.score, reverse=True)
            for e in ranked:
                e.print()
            print("=================================")

        if not kvedges:
            sample["dag"] = {"kv_nodes": [], "adj": [], "meta": {"reason": "no_kv_edges"}}
            out_samples.append(sample)
            continue

        if args.use_seededge_beam:
            seeds, kept_nodes, kept_kvedges = prune_with_seededge_bidirectional_beam(
                kvedges=kvedges,
                out_adj=out_adj,
                max_hops=args.max_hops,
                beam_width=args.beam_width,
                max_nodes=args.max_nodes,
                max_kvedges=args.max_edges,
                num_seed_edges=args.seed_top_m,
            )
        else:
            seeds = pick_seed_nodes_from_kvedges(kvedges, top_m=args.seed_top_m)
            kept_nodes, kept_kvedges = prune_with_beam_kv(
                seeds=seeds,
                out_adj=out_adj,
                kvedges=kvedges,
                max_hops=args.max_hops,
                beam_width=args.beam_width,
                max_nodes=args.max_nodes,
                max_kvedges=args.max_edges,
            )

        if not kept_kvedges:
            sample["dag"] = {"kv_nodes": [], "adj": [], "meta": {"reason": "pruned_to_empty"}}
            out_samples.append(sample)
            continue

        # break cycles on KVEdges
        kept_kvedges = break_cycles_to_dag_kv(kept_nodes, kept_kvedges, kvedges)

        # optionally enforce sink constraint (entity-level, induced by KVEdges)
        kept_nodes, kept_kvedges = enforce_max_sinks_entity(
            seeds=seeds,
            max_sinks=args.max_sinks,
            kept_nodes=kept_nodes,
            kept_kvedges=kept_kvedges,
            kvedges=kvedges,
        )

        if args.verbose:
            print_current_graph_kv(seeds, kept_nodes, kept_kvedges, kvedges)

        if not kept_kvedges:
            sample["dag"] = {"kv_nodes": [], "adj": [], "meta": {"reason": "empty_after_sink_prune"}}
            out_samples.append(sample)
            continue

        # 统计answer recall：answer在其中任何一个sink的value中出现过多少次
        outdeg = defaultdict(int)
        for eid in kept_kvedges:
            e = kvedges[eid]
            outdeg[e.src] += 1

        sinks = [n for n in kept_nodes if outdeg.get(n, 0) == 0]

        answer_matched = False
        ansN = norm_match(answer)
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

        kv_nodes, adj = export_kv_nodes_and_adj(kept_kvedges, kvedges, keep_score=args.keep_score)

        # goal labels for supervision only (not used for pruning/retrieval)
        ansN = norm_match(answer)
        goal_ids: List[int] = []
        for i, kv in enumerate(kv_nodes):
            vN = norm_match(kv.get("value", ""))
            if ansN and (vN == ansN or ansN in vN):
                goal_ids.append(i)

        meta = {
            "num_entity_nodes": int(len(kept_nodes)),
            "num_kv_edges": int(len(kept_kvedges)),
            "num_kv_nodes": int(len(kv_nodes)),
            "goal_ids": goal_ids,
        }

        sample["dag"] = {"kv_nodes": kv_nodes, "adj": adj, "meta": meta}
        out_samples.append(sample)

    print(f"Answer recall: {answer_recall / len(samples)}")
    print(f"Graph recall: {graph_recall / len(samples)}")
    return out_samples


# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="all_triples dataset file (.json or .jsonl)")
    ap.add_argument("--output", required=True, help="output dataset path (.jsonl or .json)")
    ap.add_argument(
        "--st_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name or local path",
    )
    ap.add_argument("--batch_size", type=int, default=256, help="embedding batch size")
    ap.add_argument("--keep_score", action="store_true", help="keep similarity scores in output")
    ap.add_argument("--limit", type=int, default=None, help="limit number of samples (take last N)")
    ap.add_argument("--supporting_only", action="store_true", help="only use triples from supporting_facts titles")
    ap.add_argument("--verbose", action="store_true", help="verbose debug")

    # prune knobs
    ap.add_argument("--seed_top_m", type=int, default=30, help="top-M KVEdges to derive seed nodes")
    ap.add_argument("--max_hops", type=int, default=3, help="beam expansion max hops")
    ap.add_argument("--beam_width", type=int, default=3, help="beam width per node per hop")
    ap.add_argument("--max_nodes", type=int, default=30, help="max entity nodes in pruned subgraph")
    ap.add_argument("--max_edges", type=int, default=40, help="max KVEdges in pruned subgraph")
    ap.add_argument("--max_sinks", type=int, default=4, help="max sinks (out-degree=0) on induced entity DAG")
    ap.add_argument("--use_seededge_beam", action="store_true", help="use seed-edge bidirectional beam pruning")
    ap.add_argument("--pred_weight", type=float, default=0.0, help="weight for relation prediction score")

    args = ap.parse_args()
    embedder = SentenceTransformer(args.st_model)

    samples = read_json_or_jsonl(args.input)
    print(f"Load {len(samples)} samples from {args.input}")

    out = create_dag(args, samples, embedder)
    if out is None or len(out) == 0:
        print("No valid output.")
        return

    if args.output.endswith(".jsonl"):
        write_jsonl(args.output, out)
    elif args.output.endswith(".json"):
        write_json(args.output, out)
    else:
        raise ValueError(f"Unknown file format: {args.output}")

    print(f"[DONE] input={len(samples)}  output={len(out)}  saved_to={args.output}")


if __name__ == "__main__":
    main()