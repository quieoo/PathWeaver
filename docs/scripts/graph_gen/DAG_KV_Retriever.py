import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# --------------------------
# IO utils
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
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
        raise ValueError("Unsupported JSON root format. Expect list or {data:[...]}")


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
# normalization / heuristics
# --------------------------
_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\u200b\u200c\u200d\uFEFF]")  # zero-width


def _fix_unbalanced_parentheses(s: str) -> str:
    # Minimal safe fix: if more '(' than ')', append missing ')'
    # If more ')', strip extra ')' from the end.
    if not s:
        return s
    left = s.count("(")
    right = s.count(")")
    if left > right:
        s = s + (")" * (left - right))
    elif right > left:
        # remove extra ')' from end only (avoid messing inside)
        extra = right - left
        while extra > 0 and s.endswith(")"):
            s = s[:-1]
            extra -= 1
    return s


def norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = _PUNCT_RE.sub("", s)
    s = s.strip()
    s = _SPACE_RE.sub(" ", s)
    s = _fix_unbalanced_parentheses(s)
    return s


def norm_key_for_match(s: Any) -> str:
    # for hashing / matching (slightly more aggressive)
    s = norm_text(s).lower()
    return s


def looks_like_entity_value(v: str) -> bool:
    v = norm_text(v)
    if not v:
        return False
    # pure numbers / years are likely literals, not good for weak entity linking
    if re.fullmatch(r"[\d\.,]+", v):
        return False
    if re.fullmatch(r"\d{3,4}", v):
        return False
    # very short tokens often noisy
    if len(v) <= 2:
        return False
    return True


# --------------------------
# graph data structures
# --------------------------
@dataclass
class KVPair:
    key: str
    value: str
    key_norm: str
    value_norm: str
    # optional score later
    score: float = 0.0

    def print(self) -> None:
        print(f"    ({self.key_norm} , {self.value_norm}, {self.score:.4f})")


@dataclass
class Edge:
    eid: int
    src: int
    dst: int
    relation: str
    triple_type: str
    title: str
    kvs: List[KVPair]
    score: float = 0.0  # edge relevance wrt question

    def print(self) -> None:
        print(f"Edge({self.eid}) {self.src} -> {self.dst} {self.relation} {self.triple_type} {self.title}")
        for kv in self.kvs:
            kv.print()


# --------------------------
# embedding helpers
# --------------------------
def embed_texts(
    embedder: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> np.ndarray:
    # normalize_embeddings=True -> cosine sim = dot product
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
    # if both normalized: sim = dot
    return a @ b.T


# --------------------------
# core: build entity graph from triples
# --------------------------
def iter_triples(sample: Dict[str, Any], supporting_only: bool) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Yield (title, triple_dict)
    """
    ctx = sample.get("context", [])
    supporting_titles: Optional[Set[str]] = None
    if supporting_only:
        sf = sample.get("supporting_facts", [])
        supporting_titles = {t for t, _ in sf if isinstance(t, str)}
    for para in ctx:
        title = para.get("title", "")
        if supporting_titles is not None and title not in supporting_titles:
            continue
        for tri in para.get("triple_list", []) or []:
            yield title, tri


def build_entity_graph(
    sample: Dict[str, Any],
    supporting_only: bool,
    verbose: bool = False,
) -> Tuple[List[str], Dict[str, int], Dict[int, Edge], Dict[int, List[int]]]:
    """
    Returns:
      node_names: id->name
      node_id: name_norm -> id
      edges: eid->Edge
      out_adj: src_id -> [eid...]
    """
    node_names: List[str] = []
    node_id: Dict[str, int] = {}
    edges: Dict[int, Edge] = {}
    out_adj: Dict[int, List[int]] = {}

    def get_node(x: str) -> int:
        x0 = norm_text(x)
        xn = norm_key_for_match(x0)
        if xn not in node_id:
            nid = len(node_names)
            node_id[xn] = nid
            node_names.append(x0)
        return node_id[xn]

    eid = 0
    for title, tri in iter_triples(sample, supporting_only=supporting_only):
        ttype = norm_text(tri.get("type", ""))
        s = norm_text(tri.get("name", ""))
        o = norm_text(tri.get("description", ""))
        rel = norm_text(tri.get("description_type", ""))

        if not s or not o:
            continue

        src = get_node(s)
        dst = get_node(o)

        kv_lists = tri.get("kv_lists", []) or []
        kvs: List[KVPair] = []
        for kv in kv_lists:
            k = norm_text(kv.get("key_string", ""))
            v = norm_text(kv.get("value_string", ""))
            if not k or not v:
                continue
            kvs.append(KVPair(key=k, value=v, key_norm=norm_key_for_match(k), value_norm=norm_key_for_match(v)))

        # if kvs missing, still keep edge (but it will get low score and likely pruned)
        e = Edge(
            eid=eid,
            src=src,
            dst=dst,
            relation=rel,
            triple_type=ttype,
            title=norm_text(title),
            kvs=kvs,
            score=0.0,
        )
        edges[eid] = e
        out_adj.setdefault(src, []).append(eid)
        eid += 1

    if verbose:
        print(f"[build_entity_graph] nodes={len(node_names)} edges={len(edges)} supporting_only={supporting_only}")
        # for e in edges.values():
        #     e.print()

    return node_names, node_id, edges, out_adj


# --------------------------
# scoring edges with embeddings (question vs kv.key)
# --------------------------
def score_edges_with_question(
    question: str,
    edges: Dict[int, Edge],
    embedder: SentenceTransformer,
    batch_size: int,
    keep_score: bool,
) -> None:
    q = norm_text(question)
    q_emb = embed_texts(embedder, [q], batch_size=batch_size)[0:1]  # (1,d)

    # collect all unique key strings to embed once
    all_keys: List[str] = []
    key_to_index: Dict[str, int] = {}
    for e in edges.values():
        for kv in e.kvs:
            if kv.key_norm not in key_to_index:
                key_to_index[kv.key_norm] = len(all_keys)
                all_keys.append(kv.key)

    if len(all_keys) == 0:
        for e in edges.values():
            e.score = 0.0
        return

    key_emb = embed_texts(embedder, all_keys, batch_size=batch_size)  # (K,d)
    sims = cosine_sim_matrix(q_emb, key_emb).reshape(-1)  # (K,)

    # assign kv score + edge score
    for e in edges.values():
        best = 0.0
        for kv in e.kvs:
            idx = key_to_index.get(kv.key_norm, None)
            if idx is None:
                kv.score = 0.0
                continue
            kv.score = float(sims[idx]) if keep_score else 0.0
            if float(sims[idx]) > best:
                best = float(sims[idx])
        e.score = best


# --------------------------
# prune: beam + budgets
# --------------------------
def pick_seed_nodes(edges: Dict[int, Edge], top_m: int) -> Set[int]:
    ranked = sorted(edges.values(), key=lambda e: e.score, reverse=True)
    ranked = ranked[: max(1, top_m)]
    seeds: Set[int] = set()
    for e in ranked:
        seeds.add(e.src)
        seeds.add(e.dst)
    return seeds


def prune_with_beam(
    seeds: Set[int],
    out_adj: Dict[int, List[int]],
    edges: Dict[int, Edge],
    max_hops: int,
    beam_width: int,
    max_nodes: int,
    max_edges: int,
) -> Tuple[Set[int], Set[int]]:
    """
    Return (kept_nodes, kept_edge_ids)
    """
    kept_nodes: Set[int] = set(seeds)
    kept_edges: Set[int] = set()

    frontier: List[int] = list(seeds)
    visited_in_frontier: Set[int] = set(frontier)

    for _hop in range(max_hops):
        if len(kept_edges) >= max_edges or len(kept_nodes) >= max_nodes:
            break
        new_frontier: List[int] = []
        new_seen: Set[int] = set()
        for u in frontier:
            eids = out_adj.get(u, [])
            if not eids:
                continue
            # sort edges by relevance score
            eids_sorted = sorted(eids, key=lambda eid: edges[eid].score, reverse=True)
            eids_pick = eids_sorted[: max(1, beam_width)]
            for eid in eids_pick:
                if len(kept_edges) >= max_edges:
                    break
                e = edges[eid]
                kept_edges.add(eid)
                kept_nodes.add(e.src)
                kept_nodes.add(e.dst)
                if len(kept_nodes) >= max_nodes:
                    break
                if e.dst not in visited_in_frontier and e.dst not in new_seen:
                    new_frontier.append(e.dst)
                    new_seen.add(e.dst)
            if len(kept_edges) >= max_edges or len(kept_nodes) >= max_nodes:
                break
        frontier = new_frontier
        visited_in_frontier |= new_seen
        if not frontier:
            break

    return kept_nodes, kept_edges


# --------------------------
# enforce DAG: cycle breaking
# --------------------------
def subgraph_out_adj(kept_edges: Set[int], edges: Dict[int, Edge]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for eid in kept_edges:
        e = edges[eid]
        out.setdefault(e.src, []).append(eid)
    return out


def topo_sort_nodes(nodes: Set[int], kept_edges: Set[int], edges: Dict[int, Edge]) -> Optional[List[int]]:
    indeg = {n: 0 for n in nodes}
    out = {n: [] for n in nodes}
    for eid in kept_edges:
        e = edges[eid]
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


def find_cycle_edge_set(nodes: Set[int], kept_edges: Set[int], edges: Dict[int, Edge]) -> List[int]:
    """
    Return a list of edge ids participating in one detected cycle (approx).
    Uses DFS recursion stack to find a back edge, then reconstruct a cycle path.
    """
    out = subgraph_out_adj(kept_edges, edges)
    state = {n: 0 for n in nodes}  # 0=unseen,1=visiting,2=done
    parent_edge: Dict[int, int] = {}

    def dfs(u: int) -> Optional[Tuple[int, int]]:
        state[u] = 1
        for eid in out.get(u, []):
            v = edges[eid].dst
            if v not in nodes:
                continue
            if state[v] == 0:
                parent_edge[v] = eid
                res = dfs(v)
                if res is not None:
                    return res
            elif state[v] == 1:
                # found back edge u->v
                return (u, v)  # back edge
        state[u] = 2
        return None

    back: Optional[Tuple[int, int]] = None
    for n in list(nodes):
        if state[n] == 0:
            back = dfs(n)
            if back is not None:
                break

    if back is None:
        return []

    u, v = back
    # reconstruct node cycle: v ... u -> v
    cycle_edges: List[int] = []
    # include the back edge
    # find eid for u->v in kept_edges (could be multiple; choose one with same src/dst and highest score)
    cand = [eid for eid in out.get(u, []) if edges[eid].dst == v]
    if cand:
        # keep the one in kept_edges
        back_eid = sorted(cand, key=lambda eid: edges[eid].score, reverse=True)[0]
        cycle_edges.append(back_eid)

    # walk from u backwards to v using parent_edge
    cur = u
    guard = 0
    while cur != v and guard < 10000:
        pe = parent_edge.get(cur, None)
        if pe is None:
            break
        cycle_edges.append(pe)
        cur = edges[pe].src
        guard += 1

    return list(set(cycle_edges))


def break_cycles_to_dag(nodes: Set[int], kept_edges: Set[int], edges: Dict[int, Edge]) -> Set[int]:
    """
    Ensure DAG by iteratively removing the lowest-score edge from a detected cycle.
    """
    while True:
        order = topo_sort_nodes(nodes, kept_edges, edges)
        if order is not None:
            return kept_edges
        cyc = find_cycle_edge_set(nodes, kept_edges, edges)
        if not cyc:
            # fallback: remove a global lowest-score edge
            worst = min(list(kept_edges), key=lambda eid: edges[eid].score)
            kept_edges.remove(worst)
            continue
        # remove lowest-score edge in cycle
        worst = min(cyc, key=lambda eid: edges[eid].score)
        if worst in kept_edges:
            kept_edges.remove(worst)
        else:
            # just in case
            worst2 = min(list(kept_edges), key=lambda eid: edges[eid].score)
            kept_edges.remove(worst2)


# --------------------------
# enforce max sinks (out-degree == 0) using support score DP
# --------------------------
def induced_nodes_from_edges(seeds: Set[int], kept_edges: Set[int], edges: Dict[int, Edge]) -> Set[int]:
    nodes: Set[int] = set(seeds)
    for eid in kept_edges:
        e = edges[eid]
        nodes.add(e.src)
        nodes.add(e.dst)
    return nodes


def compute_reachable(seeds: Set[int], kept_edges: Set[int], edges: Dict[int, Edge]) -> Tuple[Set[int], Set[int]]:
    out = subgraph_out_adj(kept_edges, edges)
    vis_nodes: Set[int] = set()
    q = list(seeds)
    while q:
        u = q.pop()
        if u in vis_nodes:
            continue
        vis_nodes.add(u)
        for eid in out.get(u, []):
            v = edges[eid].dst
            if v not in vis_nodes:
                q.append(v)
    vis_edges = {eid for eid in kept_edges if edges[eid].src in vis_nodes and edges[eid].dst in vis_nodes}
    return vis_nodes, vis_edges


def enforce_max_sinks(
    seeds: Set[int],
    max_sinks: int,
    nodes: Set[int],
    kept_edges: Set[int],
    edges: Dict[int, Edge],
) -> Tuple[Set[int], Set[int]]:
    if max_sinks is None or max_sinks <= 0:
        return nodes, kept_edges

    while True:
        out = subgraph_out_adj(kept_edges, edges)
        outdeg = {n: 0 for n in nodes}
        indeg_edges: Dict[int, List[int]] = {n: [] for n in nodes}
        for eid in kept_edges:
            e = edges[eid]
            if e.src in nodes and e.dst in nodes:
                outdeg[e.src] += 1
                indeg_edges[e.dst].append(eid)

        sinks = [n for n in nodes if outdeg.get(n, 0) == 0]
        if len(sinks) <= max_sinks:
            break

        # DAG DP: max path score from any seed
        order = topo_sort_nodes(nodes, kept_edges, edges)
        if order is None:
            # should not happen after break_cycles_to_dag, but safe
            kept_edges = break_cycles_to_dag(nodes, kept_edges, edges)

        score = {n: -1e9 for n in nodes}
        prev_edge: Dict[int, Optional[int]] = {n: None for n in nodes}
        for s in seeds:
            if s in nodes:
                score[s] = 0.0

        out_e = out
        for u in order or []:
            if score[u] <= -1e8:
                continue
            for eid in out_e.get(u, []):
                v = edges[eid].dst
                if v not in nodes:
                    continue
                cand = score[u] + edges[eid].score
                if cand > score[v]:
                    score[v] = cand
                    prev_edge[v] = eid

        # choose sink to prune: lowest support score
        sinks_sorted = sorted(sinks, key=lambda n: score.get(n, -1e9))
        target_sink = sinks_sorted[0]

        # remove its best incoming edge (or any incoming if none)
        pe = prev_edge.get(target_sink, None)
        if pe is None:
            # remove any incoming edge
            ins = indeg_edges.get(target_sink, [])
            if not ins:
                # isolated node; drop it
                nodes.remove(target_sink)
            else:
                # remove weakest incoming (more aggressive)
                pe = min(ins, key=lambda eid: edges[eid].score)
                kept_edges.remove(pe)
        else:
            if pe in kept_edges:
                kept_edges.remove(pe)

        # after edge removal, keep only reachable portion
        nodes, kept_edges = compute_reachable(seeds, kept_edges, edges)

        if len(kept_edges) == 0:
            break

    return nodes, kept_edges


# --------------------------
# export KV graph + adjacency
# --------------------------
def export_kv_graph(
    question: str,
    nodes: Set[int],
    kept_edges: Set[int],
    edges: Dict[int, Edge],
    embedder: SentenceTransformer,
    batch_size: int,
    kv_per_edge: int,
    keep_score: bool,
) -> Tuple[List[Dict[str, Any]], List[List[int]]]:
    """
    Returns (kv_nodes, adj_matrix)
    """
    # Build kv_nodes by selecting top kv_per_edge kvs for each kept edge
    # Score kv by sim(question, kv.key)
    q = norm_text(question)
    q_emb = embed_texts(embedder, [q], batch_size=batch_size)[0:1]

    # gather candidate kv keys
    kv_key_texts: List[str] = []
    kv_key_owner: List[Tuple[int, int]] = []  # (eid, kv_idx)
    for eid in kept_edges:
        e = edges[eid]
        for i, kv in enumerate(e.kvs):
            kv_key_texts.append(kv.key)
            kv_key_owner.append((eid, i))

    kv_sims: Optional[np.ndarray] = None
    if kv_key_texts:
        kv_key_emb = embed_texts(embedder, kv_key_texts, batch_size=batch_size)
        kv_sims = cosine_sim_matrix(q_emb, kv_key_emb).reshape(-1)

    # pick kvs for each edge
    selected_kv_nodes: List[Dict[str, Any]] = []
    kv_node_edge: List[int] = []  # node -> eid
    kv_node_srcdst: List[Tuple[int, int]] = []  # node -> (src,dst)
    edge_to_kvnode_ids: Dict[int, List[int]] = {}

    # group per edge
    per_edge_candidates: Dict[int, List[Tuple[float, int]]] = {}  # eid -> [(sim, global_idx)...]
    if kv_sims is not None:
        for global_idx, (eid, kv_i) in enumerate(kv_key_owner):
            per_edge_candidates.setdefault(eid, []).append((float(kv_sims[global_idx]), global_idx))

    for eid in kept_edges:
        e = edges[eid]
        cands = per_edge_candidates.get(eid, [])
        if not cands:
            # fallback: create a single kv node from relation itself (no score)
            key = f"{edges[eid].relation} of {e.src}"
            val = str(e.dst)
            node_obj = {
                "key": key,
                "value": val,
                "src": e.src,
                "dst": e.dst,
                "edge_id": eid,
            }
            if keep_score:
                node_obj["score"] = 0.0
            nid = len(selected_kv_nodes)
            selected_kv_nodes.append(node_obj)
            kv_node_edge.append(eid)
            kv_node_srcdst.append((e.src, e.dst))
            edge_to_kvnode_ids.setdefault(eid, []).append(nid)
            continue

        # sort by similarity desc
        cands_sorted = sorted(cands, key=lambda x: x[0], reverse=True)
        take = cands_sorted[: max(1, kv_per_edge)]
        for sim, global_idx in take:
            oeid, kv_i = kv_key_owner[global_idx]
            kv = edges[oeid].kvs[kv_i]
            node_obj = {
                "key": kv.key,
                "value": kv.value,
                "src": edges[oeid].src,
                "dst": edges[oeid].dst,
                "edge_id": oeid,
            }
            if keep_score:
                node_obj["score"] = float(sim)
                node_obj["kv_variant"] = kv_i
            nid = len(selected_kv_nodes)
            selected_kv_nodes.append(node_obj)
            kv_node_edge.append(oeid)
            kv_node_srcdst.append((edges[oeid].src, edges[oeid].dst))
            edge_to_kvnode_ids.setdefault(oeid, []).append(nid)

    # Build adjacency on KV nodes: if edge1 dst == edge2 src then connect all kv nodes of edge1 -> all kv nodes of edge2
    n = len(selected_kv_nodes)
    adj = [[0] * n for _ in range(n)]

    # map from entity node -> outgoing kept edges
    ent_out_edges: Dict[int, List[int]] = {}
    for eid in kept_edges:
        e = edges[eid]
        ent_out_edges.setdefault(e.src, []).append(eid)

    # for each edge1, find edge2 starting from edge1.dst
    for eid1 in kept_edges:
        e1 = edges[eid1]
        next_edges = ent_out_edges.get(e1.dst, [])
        if not next_edges:
            continue
        for eid2 in next_edges:
            if eid2 not in kept_edges:
                continue
            # connect all kv nodes representing eid1 to all kv nodes representing eid2
            for i in edge_to_kvnode_ids.get(eid1, []):
                for j in edge_to_kvnode_ids.get(eid2, []):
                    adj[i][j] = 1

    return selected_kv_nodes, adj


# --------------------------
# main create_dag
# --------------------------
def create_dag(args, samples: List[Dict[str, Any]], embedder: SentenceTransformer) -> List[Dict[str, Any]]:
    out_samples: List[Dict[str, Any]] = []

    # apply limit (take last N, matching your previous convention)
    if args.limit is not None and args.limit > 0:
        samples = samples[:args.limit]
    
    # debug use
    sample_id=2
    samples=[samples[sample_id]]

    # for sample in tqdm(samples, desc="Create DAG"):
    for sample in samples:
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # 1) entity graph
        node_names, node_id_map, edges, out_adj = build_entity_graph(
            sample, supporting_only=args.supporting_only, verbose=args.verbose
        )

        if len(edges) == 0:
            sample["dag"] = {"kv_nodes": [], "adj": [], "meta": {"reason": "no_edges"}}
            out_samples.append(sample)
            continue

        # 2) score edges by question-key similarity (answer not used)
        score_edges_with_question(
            question=question,
            edges=edges,
            embedder=embedder,
            batch_size=args.batch_size,
            keep_score=args.keep_score,
        )

        if args.verbose:
            print(f"[create_dag] question={question} answer={answer}")
            for e in edges.values():
                e.print()
            # NOTE: 有些三元组（比如第二跳之后的）他们的边打分不准，可能错误方向的边反而分数更高

        # 3) seeds and prune
        seeds = pick_seed_nodes(edges, top_m=args.seed_top_m)
        if args.verbose:
            print(f"[create_dag] seeds={seeds}")

        kept_nodes, kept_edges = prune_with_beam(
            seeds=seeds,
            out_adj=out_adj,
            edges=edges,
            max_hops=args.max_hops,
            beam_width=args.beam_width,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
        )
        if args.verbose:
            print(f"[create_dag] kept_nodes={kept_nodes} kept_edges={kept_edges}")

        if len(kept_edges) == 0:
            sample["dag"] = {"kv_nodes": [], "adj": [], "meta": {"reason": "pruned_to_empty"}}
            out_samples.append(sample)
            continue

        # induce nodes from edges (ensure endpoints included)
        kept_nodes = induced_nodes_from_edges(seeds, kept_edges, edges)

        # 4) break cycles -> DAG
        kept_edges = break_cycles_to_dag(kept_nodes, kept_edges, edges)
        kept_nodes = induced_nodes_from_edges(seeds, kept_edges, edges)

        if args.verbose:
            print(f"[create_dag break cycles] kept_nodes={kept_nodes} kept_edges={kept_edges}")

        # keep only reachable from seeds (makes DAG more “chain-like”)
        kept_nodes, kept_edges = compute_reachable(seeds, kept_edges, edges)

        if len(kept_edges) == 0:
            sample["dag"] = {"kv_nodes": [], "adj": [], "meta": {"reason": "no_reachable_edges"}}
            out_samples.append(sample)
            continue
        
        if args.verbose:
            print(f"[create_dag reachable] kept_nodes={kept_nodes} kept_edges={kept_edges}")

        # 5) enforce max sinks
        # kept_nodes, kept_edges = enforce_max_sinks(
        #     seeds=seeds,
        #     max_sinks=args.max_sinks,
        #     nodes=kept_nodes,
        #     kept_edges=kept_edges,
        #     edges=edges,
        # )

        # if args.verbose:
        #     print(f"[create_dag max sinks] kept_nodes={kept_nodes} kept_edges={kept_edges}")

        # 6) export KV graph
        kv_nodes, adj = export_kv_graph(
            question=question,
            nodes=kept_nodes,
            kept_edges=kept_edges,
            edges=edges,
            embedder=embedder,
            batch_size=args.batch_size,
            kv_per_edge=args.kv_per_edge,
            keep_score=args.keep_score,
        )

        # 7) label goal ids (for supervision only; not used for pruning)
        ans_norm = norm_key_for_match(answer)
        goal_ids = []
        for i, kv in enumerate(kv_nodes):
            v_norm = norm_key_for_match(kv.get("value", ""))
            # exact match or substring (handles cases like "..., New York City")
            if ans_norm and (v_norm == ans_norm or ans_norm in v_norm):
                goal_ids.append(i)

        meta = {
            "num_entity_nodes": int(len(kept_nodes)),
            "num_entity_edges": int(len(kept_edges)),
            "num_kv_nodes": int(len(kv_nodes)),
            "goal_ids": goal_ids,
        }

        if args.keep_score:
            # edge score snapshot (optional)
            meta["edge_scores"] = {str(eid): float(edges[eid].score) for eid in kept_edges}

        sample["dag"] = {
            "kv_nodes": kv_nodes,
            "adj": adj,
            "meta": meta,
        }
        out_samples.append(sample)

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
    ap.add_argument("--keep_score", action="store_true", help="keep similarity scores & variant id")
    ap.add_argument("--limit", type=int, default=None, help="limit number of samples (take last N)")
    ap.add_argument("--k", type=int, default=1, help="reserved: top-k paths (currently used indirectly by budgets)")
    ap.add_argument("--supporting_only", action="store_true", help="only use triples from supporting_facts titles")
    ap.add_argument("--verbose", action="store_true", help="verbose debug")

    # DAG/prune knobs
    ap.add_argument("--seed_top_m", type=int, default=20, help="top-M edges to derive seed nodes")
    ap.add_argument("--max_hops", type=int, default=3, help="beam expansion max hops")
    ap.add_argument("--beam_width", type=int, default=3, help="beam width per node per hop")
    ap.add_argument("--max_nodes", type=int, default=25, help="max entity nodes in pruned subgraph")
    ap.add_argument("--max_edges", type=int, default=20, help="max entity edges in pruned subgraph")
    ap.add_argument("--max_sinks", type=int, default=3, help="max number of sinks(out-degree=0) in entity DAG")
    ap.add_argument("--kv_per_edge", type=int, default=1, help="how many KV variants kept per triple edge")

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