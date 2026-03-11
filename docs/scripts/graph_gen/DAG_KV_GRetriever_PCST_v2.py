import argparse
import math
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import DAG_KV_Retriever_v3 as base

try:
    from pcst_fast import pcst_fast  # type: ignore
    HAS_PCST_FAST = True
except Exception:
    pcst_fast = None
    HAS_PCST_FAST = False


# ============================================================
# 0) Small helpers
# ============================================================
def answer_matches_text(answer: str, text: str) -> bool:
    ansN = base.norm_match(answer)
    txtN = base.norm_match(text)
    return bool(ansN and txtN and (ansN == txtN or ansN in txtN or txtN in ansN))


def is_value_like_text(name: str) -> bool:
    n = base.norm_match(name)
    if not n:
        return False
    if any(ch.isdigit() for ch in n):
        return True
    toks = n.split()
    if len(toks) <= 3 and n[:1].islower():
        return True
    return False


def is_value_like_node(node_id: int, node_names: List[str], kvedges: Dict[int, base.KVEdge]) -> bool:
    name = node_names[node_id] if 0 <= node_id < len(node_names) else ""
    if is_value_like_text(name):
        return True
    in_attr = 0
    total_in = 0
    for e in kvedges.values():
        if e.dst == node_id:
            total_in += 1
            if (e.triple_type or "").upper() == "ATTRIBUTE":
                in_attr += 1
    return total_in > 0 and in_attr >= max(1, total_in // 2)


# ============================================================
# 1) G-Retriever-style node / edge textualization and prizes
# ============================================================
def build_edge_text(e: base.KVEdge, node_names: List[str]) -> str:
    src = node_names[e.src] if 0 <= e.src < len(node_names) else ""
    dst = node_names[e.dst] if 0 <= e.dst < len(node_names) else ""
    rel = base.norm_text(e.relation)
    key = base.norm_text(getattr(e, "key_string", "") or "")
    value = base.norm_text(e.value)
    parts = [p for p in [src, rel, key, dst, value] if p]
    return " ; ".join(parts)



def compute_question_sims(
    question: str,
    node_names: List[str],
    kvedges: Dict[int, base.KVEdge],
    embedder: SentenceTransformer,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    q_emb = base.embed_texts(embedder, [question], batch_size=batch_size)[0:1]

    if node_names:
        node_emb = base.embed_texts(embedder, node_names, batch_size=batch_size)
        n_sims = base.cosine_sim_matrix(q_emb, node_emb).reshape(-1).astype(np.float32)
    else:
        n_sims = np.zeros((0,), dtype=np.float32)

    edge_ids = sorted(kvedges.keys())
    edge_texts = [build_edge_text(kvedges[eid], node_names) for eid in edge_ids]
    if edge_texts:
        edge_emb = base.embed_texts(embedder, edge_texts, batch_size=batch_size)
        e_sims = base.cosine_sim_matrix(q_emb, edge_emb).reshape(-1).astype(np.float32)
    else:
        e_sims = np.zeros((0,), dtype=np.float32)

    return n_sims, e_sims, edge_texts



def rank_topk_prizes(values: np.ndarray, topk: int) -> np.ndarray:
    """
    Match retrieval.py's top-k ranking style more closely:
    keep only top-k unique/sorted candidates and assign descending rank prizes.
    """
    if values.size == 0 or topk <= 0:
        return np.zeros_like(values, dtype=np.float32)
    topk = min(int(topk), int(values.shape[0]))
    idx = np.argpartition(values, -topk)[-topk:]
    idx = idx[np.argsort(values[idx])[::-1]]
    prizes = np.zeros_like(values, dtype=np.float32)
    prizes[idx] = np.arange(topk, 0, -1, dtype=np.float32)
    return prizes



def rank_topk_edge_prizes(values: np.ndarray, topk_e: int, c: float = 0.01) -> np.ndarray:
    """
    Port of retrieval.py edge-prize logic:
    - take top-k unique similarity levels
    - zero out below threshold
    - assign decreasing prize masses with tie handling
    """
    if values.size == 0 or topk_e <= 0:
        return np.zeros_like(values, dtype=np.float32)
    uniq = np.unique(values)
    if uniq.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    topk_e = min(int(topk_e), int(uniq.size))
    top_vals = np.sort(uniq)[-topk_e:][::-1]
    prizes = values.copy().astype(np.float32)
    prizes[prizes < top_vals[-1]] = 0.0
    last_topk_val = float(topk_e)
    for k in range(topk_e):
        mask = prizes == top_vals[k]
        cnt = int(mask.sum())
        if cnt <= 0:
            continue
        value = min((topk_e - k) / cnt, last_topk_val)
        prizes[mask] = value
        last_topk_val = value * (1 - c)
    return prizes



def compute_anchor_set(question: str, node_names: List[str]) -> Set[int]:
    q_norm = base._norm_lex(question)
    anchors: Set[int] = set()
    for nid, name in enumerate(node_names):
        if base._contains_mention(q_norm, name):
            anchors.add(nid)
    return anchors


# ============================================================
# 2) retrieval.py-style PCST construction
# ============================================================
def build_pcst_problem_from_kv_graph(
    node_names: List[str],
    kvedges: Dict[int, base.KVEdge],
    node_prizes: np.ndarray,
    edge_prizes: np.ndarray,
    cost_e: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int], int]:
    """
    Reproduce retrieval.py's core transformation:
    - ordinary edge: cost = cost_e - prize_e
    - high-prize edge: introduce a virtual node with prize = prize_e - cost_e
      and replace edge by src->virtual, virtual->dst, both zero cost.

    Returns:
      edges_arr, costs_arr, mapping_e, mapping_virtual_node_to_eid, num_regular_edges
    where mapping_e maps pcst edge index -> original eid (for regular edges only).
    """
    edge_ids = sorted(kvedges.keys())
    costs: List[float] = []
    edges: List[Tuple[int, int]] = []
    virtual_node_prizes: List[float] = []
    virtual_edges: List[Tuple[int, int]] = []
    virtual_costs: List[float] = []
    mapping_e: Dict[int, int] = {}
    mapping_virtual_node_to_eid: Dict[int, int] = {}

    num_nodes = len(node_names)
    for i, eid in enumerate(edge_ids):
        e = kvedges[eid]
        if e.src == e.dst:
            continue
        prize_e = float(edge_prizes[i]) if i < len(edge_prizes) else 0.0
        if prize_e <= cost_e:
            mapping_e[len(edges)] = eid
            edges.append((e.src, e.dst))
            costs.append(float(cost_e - prize_e))
        else:
            virtual_node_id = num_nodes + len(virtual_node_prizes)
            mapping_virtual_node_to_eid[virtual_node_id] = eid
            virtual_edges.append((e.src, virtual_node_id))
            virtual_edges.append((virtual_node_id, e.dst))
            virtual_costs.append(0.0)
            virtual_costs.append(0.0)
            virtual_node_prizes.append(float(prize_e - cost_e))

    num_regular_edges = len(edges)
    if virtual_edges:
        edges_arr = np.array(edges + virtual_edges, dtype=np.int64)
        costs_arr = np.array(costs + virtual_costs, dtype=np.float32)
    else:
        edges_arr = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)
        costs_arr = np.array(costs, dtype=np.float32) if costs else np.zeros((0,), dtype=np.float32)

    all_node_prizes = np.concatenate([
        node_prizes.astype(np.float32),
        np.array(virtual_node_prizes, dtype=np.float32)
    ])
    return edges_arr, costs_arr, mapping_e, mapping_virtual_node_to_eid, num_regular_edges, all_node_prizes



def fallback_pcst_like_selection(
    node_names: List[str],
    kvedges: Dict[int, base.KVEdge],
    node_prizes: np.ndarray,
    edge_prizes: np.ndarray,
    anchors: Set[int],
    max_nodes: int,
    max_edges: int,
    cost_e: float,
) -> Tuple[Set[int], Set[int], int]:
    """
    If pcst_fast is unavailable, approximate retrieval.py with a node/edge prize graph,
    then connect top terminals with weighted Steiner tree.
    """
    G = nx.Graph()
    for nid, name in enumerate(node_names):
        G.add_node(nid, name=name)

    edge_ids = sorted(kvedges.keys())
    best_pair_cost: Dict[Tuple[int, int], float] = {}
    best_pair_eid: Dict[Tuple[int, int], int] = {}
    for i, eid in enumerate(edge_ids):
        e = kvedges[eid]
        if e.src == e.dst:
            continue
        pair = (min(e.src, e.dst), max(e.src, e.dst))
        c = float(max(1e-4, cost_e - float(edge_prizes[i])))
        if pair not in best_pair_cost or c < best_pair_cost[pair]:
            best_pair_cost[pair] = c
            best_pair_eid[pair] = eid

    for (u, v), w in best_pair_cost.items():
        G.add_edge(u, v, weight=w)

    if G.number_of_nodes() == 0:
        return set(), set(), -1

    root = max(anchors, key=lambda n: float(node_prizes[n])) if anchors else int(np.argmax(node_prizes))
    candidate_nodes = list(np.argsort(node_prizes)[::-1])
    terminals: List[int] = []
    for nid in candidate_nodes:
        if node_prizes[nid] <= 0:
            continue
        if not G.has_node(int(nid)):
            continue
        if terminals and len(terminals) >= min(max_nodes, 8):
            break
        if anchors and nid in anchors:
            terminals.append(int(nid))
        elif len(terminals) < min(max_nodes, 8):
            terminals.append(int(nid))
    if root not in terminals:
        terminals = [root] + terminals
    terminals = list(dict.fromkeys(terminals))
    if len(terminals) == 1:
        return {terminals[0]}, set(), root

    tree = nx.algorithms.approximation.steiner_tree(G, terminals, weight="weight")
    while tree.number_of_nodes() > max_nodes or tree.number_of_edges() > max_edges:
        leaves = [n for n in tree.nodes if tree.degree[n] <= 1 and n not in anchors and n != root]
        if not leaves:
            break
        worst = min(leaves, key=lambda n: float(node_prizes[n]))
        tree.remove_node(worst)

    selected_nodes = set(tree.nodes())
    selected_eids: Set[int] = set()
    for u, v in tree.edges():
        pair = (min(u, v), max(u, v))
        if pair in best_pair_eid:
            selected_eids.add(best_pair_eid[pair])
    return selected_nodes, selected_eids, root



def retrieval_via_pcst_for_dag_kv(
    question: str,
    node_names: List[str],
    kvedges: Dict[int, base.KVEdge],
    embedder: SentenceTransformer,
    batch_size: int,
    topk: int,
    topk_e: int,
    cost_e: float,
    max_nodes: int,
    max_edges: int,
    anchor_bonus: float = 1.0,
    node_sim_weight: float = 1.0,
    edge_sim_weight: float = 1.0,
) -> Tuple[Set[int], Set[int], Dict[int, float], Set[int], int]:
    """
    DAG-KV adaptation of retrieval.py's retrieval_via_pcst.

    Differences from the original:
    - input is KV graph rather than PyG Data
    - output is selected original node ids and original edge ids
    - still uses top-k node prizes, top-k edge prizes, and virtual-node PCST transform
    """
    if not node_names or not kvedges:
        return set(), set(), {}, set(), -1

    n_sims, e_sims, _ = compute_question_sims(
        question=question,
        node_names=node_names,
        kvedges=kvedges,
        embedder=embedder,
        batch_size=batch_size,
    )

    anchors = compute_anchor_set(question, node_names)

    n_rank = rank_topk_prizes(np.maximum(0.0, n_sims), topk=topk)
    if anchors:
        for nid in anchors:
            if 0 <= nid < len(n_rank):
                n_rank[nid] += float(anchor_bonus)
    node_prizes = node_sim_weight * n_rank

    e_rank = rank_topk_edge_prizes(np.maximum(0.0, e_sims), topk_e=topk_e)
    edge_prizes = edge_sim_weight * e_rank
    if edge_prizes.size > 0 and edge_prizes.max() > 0:
        cost_e = min(float(cost_e), float(edge_prizes.max()) * (1 - 0.01 / 2))

    prize_map = {nid: float(node_prizes[nid]) for nid in range(len(node_names))}

    if HAS_PCST_FAST:
        edges_arr, costs_arr, mapping_e, mapping_virtual_node_to_eid, num_regular_edges, all_node_prizes = build_pcst_problem_from_kv_graph(
            node_names=node_names,
            kvedges=kvedges,
            node_prizes=node_prizes,
            edge_prizes=edge_prizes,
            cost_e=cost_e,
        )

        if edges_arr.shape[0] == 0:
            # no usable edges, keep best prize nodes only
            kept_nodes = set(int(i) for i in np.argsort(node_prizes)[::-1][:max(1, min(max_nodes, topk))] if node_prizes[i] > 0)
            root = max(anchors, key=lambda n: prize_map.get(n, -1e9)) if anchors else (max(kept_nodes, key=lambda n: prize_map.get(n, -1e9)) if kept_nodes else -1)
            return kept_nodes, set(), prize_map, anchors, root

        root = -1
        num_clusters = 1
        pruning = 'gw'
        verbosity_level = 0
        vertices, edge_indices = pcst_fast(edges_arr, all_node_prizes, costs_arr, root, num_clusters, pruning, verbosity_level)

        vertices = np.array(vertices, dtype=np.int64)
        edge_indices = np.array(edge_indices, dtype=np.int64)

        selected_nodes = set(int(v) for v in vertices if v < len(node_names))
        selected_edges: Set[int] = set(int(mapping_e[eidx]) for eidx in edge_indices if int(eidx) < num_regular_edges and int(eidx) in mapping_e)
        virtual_vertices = [int(v) for v in vertices if v >= len(node_names)]
        for vv in virtual_vertices:
            if vv in mapping_virtual_node_to_eid:
                selected_edges.add(int(mapping_virtual_node_to_eid[vv]))

        # ensure endpoints are included
        for eid in list(selected_edges):
            e = kvedges[eid]
            selected_nodes.add(int(e.src))
            selected_nodes.add(int(e.dst))

        # budget trim: preferentially keep anchors + higher prize nodes
        if len(selected_nodes) > max_nodes:
            ordered = sorted(selected_nodes, key=lambda n: (n not in anchors, -prize_map.get(n, 0.0), n))
            keep = set(ordered[:max_nodes])
            selected_edges = set(eid for eid in selected_edges if kvedges[eid].src in keep and kvedges[eid].dst in keep)
            selected_nodes = keep

        if len(selected_edges) > max_edges:
            edge_ids_sorted = sorted(
                list(selected_edges),
                key=lambda eid: (
                    -(edge_prizes[sorted(kvedges.keys()).index(eid)] if eid in kvedges else 0.0),
                    -kvedges[eid].score,
                )
            )
            selected_edges = set(edge_ids_sorted[:max_edges])
            selected_nodes = set()
            for eid in selected_edges:
                e = kvedges[eid]
                selected_nodes.add(int(e.src))
                selected_nodes.add(int(e.dst))

        pcst_root = max(anchors & selected_nodes, key=lambda n: prize_map.get(n, -1e9)) if (anchors & selected_nodes) else (max(selected_nodes, key=lambda n: prize_map.get(n, -1e9)) if selected_nodes else -1)
        return selected_nodes, selected_edges, prize_map, anchors, pcst_root

    # fallback if pcst_fast is not installed
    selected_nodes, selected_edges, pcst_root = fallback_pcst_like_selection(
        node_names=node_names,
        kvedges=kvedges,
        node_prizes=node_prizes,
        edge_prizes=edge_prizes,
        anchors=anchors,
        max_nodes=max_nodes,
        max_edges=max_edges,
        cost_e=cost_e,
    )
    return selected_nodes, selected_edges, prize_map, anchors, pcst_root


# ============================================================
# 3) Directed DAG orientation after PCST retrieval
# ============================================================
def build_undirected_pairs(selected_edges: Set[int], kvedges: Dict[int, base.KVEdge]) -> Set[Tuple[int, int]]:
    pairs: Set[Tuple[int, int]] = set()
    for eid in selected_edges:
        e = kvedges[eid]
        if e.src == e.dst:
            continue
        pairs.add((min(e.src, e.dst), max(e.src, e.dst)))
    return pairs



def compute_root_distances(tree_nodes: Set[int], undirected_edges: Set[Tuple[int, int]], root: int) -> Dict[int, int]:
    adj = defaultdict(list)
    for u, v in undirected_edges:
        adj[u].append(v)
        adj[v].append(u)
    dist = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    for n in tree_nodes:
        dist.setdefault(n, 10 ** 9)
    return dist



def select_directed_edges_from_pcst_subgraph(
    selected_nodes: Set[int],
    selected_edges: Set[int],
    root: int,
    node_names: List[str],
    kvedges: Dict[int, base.KVEdge],
    prize_map: Dict[int, float],
    anchors: Set[int],
    leaf_bonus: float = 0.35,
    anchor_penalty: float = 0.25,
    internal_value_penalty: float = 0.10,
) -> Set[int]:
    if not selected_nodes:
        return set()
    if root < 0:
        root = max(selected_nodes, key=lambda n: prize_map.get(n, -1e9))

    undirected_edges = build_undirected_pairs(selected_edges, kvedges)
    dist = compute_root_distances(selected_nodes, undirected_edges, root=root)

    undir_to_all: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for eid in selected_edges:
        e = kvedges[eid]
        pair = (min(e.src, e.dst), max(e.src, e.dst))
        undir_to_all[pair].append(eid)

    kept: Set[int] = set()
    chosen_parent: Dict[int, Tuple[float, int]] = {}
    for u, v in undirected_edges:
        du, dv = dist.get(u, 10 ** 9), dist.get(v, 10 ** 9)
        if du <= dv:
            src_pref, dst_pref = u, v
        else:
            src_pref, dst_pref = v, u

        cand = undir_to_all[(u, v)]

        def orient_score(eid: int) -> float:
            e = kvedges[eid]
            src, dst = e.src, e.dst
            s = float(e.score)
            # prefer edges that follow root outward
            if src == src_pref and dst == dst_pref:
                s += 0.25
            elif src == dst_pref and dst == src_pref:
                s -= 0.10
            # prefer value-like destinations as leaves
            if is_value_like_node(dst, node_names, kvedges) and dst not in anchors:
                s += leaf_bonus
            if is_value_like_node(src, node_names, kvedges) and src != root:
                s -= internal_value_penalty
            if dst in anchors and dst != root:
                s -= anchor_penalty
            s += 0.05 * prize_map.get(dst, 0.0)
            return s

        best = max(cand, key=orient_score)
        e = kvedges[best]
        parent_score = orient_score(best)
        prev = chosen_parent.get(e.dst)
        if prev is None or parent_score > prev[0]:
            chosen_parent[e.dst] = (parent_score, best)

    for _, eid in chosen_parent.values():
        kept.add(eid)

    # connectivity repair: if root component misses some nodes, greedily add best outgoing edge from visited set
    visited = {root}
    changed = True
    while changed:
        changed = False
        for eid in list(kept):
            e = kvedges[eid]
            if e.src in visited and e.dst not in visited:
                visited.add(e.dst)
                changed = True

    missing = [n for n in selected_nodes if n not in visited]
    if missing:
        candidate_eids = [eid for eid in selected_edges if eid not in kept]
        candidate_eids = sorted(candidate_eids, key=lambda eid: kvedges[eid].score, reverse=True)
        while missing:
            added = False
            for eid in candidate_eids:
                e = kvedges[eid]
                if e.src in visited and e.dst in selected_nodes and e.dst not in visited:
                    kept.add(eid)
                    visited.add(e.dst)
                    added = True
                elif e.dst in visited and e.src in selected_nodes and e.src not in visited:
                    # reverse orientation if needed by selecting same undirected pair's reverse best if exists
                    pair = (min(e.src, e.dst), max(e.src, e.dst))
                    alternatives = undir_to_all.get(pair, [eid])
                    rev = None
                    for aeid in alternatives:
                        ae = kvedges[aeid]
                        if ae.src == e.dst and ae.dst == e.src:
                            rev = aeid
                            break
                    kept.add(rev if rev is not None else eid)
                    visited.add(e.src)
                    added = True
                if added:
                    break
            if not added:
                break
            missing = [n for n in selected_nodes if n not in visited]

    return kept


# ============================================================
# 4) Main pipeline
# ============================================================
def create_dag(args, samples: List[Dict[str, Any]], embedder: SentenceTransformer) -> List[Dict[str, Any]]:
    if args.limit is not None and args.limit > 0:
        samples = samples[:args.limit]

    out_samples: List[Dict[str, Any]] = []
    answer_recall = 0
    graph_recall = 0
    none_sink_recall = 0

    for sample in tqdm(samples, desc="Create DAG (G-Retriever PCST)"):
        question = base.norm_text(sample.get("question", ""))
        answer = base.norm_text(sample.get("answer", ""))

        node_names, out_adj, kvedges = base.build_kvedge_graph(
            sample=sample,
            embedder=embedder,
            batch_size=args.batch_size,
            supporting_only=args.supporting_only,
            pred_weight=args.pred_weight,
        )

        if args.max_attr_out_per_entity is not None and args.max_attr_out_per_entity > 0:
            out_adj, kvedges = base.prune_edges_before_search(
                out_adj=out_adj,
                kvedges=kvedges,
                node_names=node_names,
                max_attr_out_per_entity=args.max_attr_out_per_entity,
                keep_best_edge_per_pair=True,
            )

        if not kvedges:
            sample["dag"] = {"kv_nodes": [], "adj": [], "meta": {"reason": "no_kv_edges"}}
            out_samples.append(sample)
            continue

        if args.keep_one_attr_direction:
            q_emb = base.embed_texts(embedder, [question], batch_size=args.batch_size)[0]
            embed_text_fn = lambda t: base.embed_texts(embedder, [t], batch_size=args.batch_size)[0]
            kept_eids = base.keep_one_direction_for_attribute_pairs(
                question=question,
                kvedges=kvedges,
                node_names=node_names,
                embed_text_fn=embed_text_fn,
                q_emb=q_emb,
                eps_dir=args.eps_dir,
                keep_when_tie=True,
            )
            kvedges = {eid: e for eid, e in kvedges.items() if eid in kept_eids}
            out_adj = base.rebuild_out_adj(kvedges)

        # graph recall on full graph, same as v3
        for e in kvedges.values():
            if answer_matches_text(answer, e.value):
                graph_recall += 1
                break

        selected_nodes, selected_edges, prize_map, anchors, pcst_root = retrieval_via_pcst_for_dag_kv(
            question=question,
            node_names=node_names,
            kvedges=kvedges,
            embedder=embedder,
            batch_size=args.batch_size,
            topk=args.topk_node_prize,
            topk_e=args.topk_edge_prize,
            cost_e=args.pcst_cost_e,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            anchor_bonus=args.anchor_bonus,
            node_sim_weight=args.node_sim_weight,
            edge_sim_weight=args.edge_sim_weight,
        )

        if not selected_nodes:
            sample["dag"] = {"kv_nodes": [], "adj": [], "meta": {"reason": "pcst_empty", "pcst_fast": bool(HAS_PCST_FAST)}}
            out_samples.append(sample)
            continue

        if pcst_root < 0:
            pcst_root = max(selected_nodes, key=lambda n: prize_map.get(n, -1e9))

        kept_kvedges = select_directed_edges_from_pcst_subgraph(
            selected_nodes=selected_nodes,
            selected_edges=selected_edges,
            root=pcst_root,
            node_names=node_names,
            kvedges=kvedges,
            prize_map=prize_map,
            anchors=anchors,
            leaf_bonus=args.leaf_bonus,
            anchor_penalty=args.anchor_penalty,
            internal_value_penalty=args.internal_value_penalty,
        )

        if kept_kvedges:
            kept_nodes = set()
            for eid in kept_kvedges:
                e = kvedges[eid]
                kept_nodes.add(e.src)
                kept_nodes.add(e.dst)
        else:
            kept_nodes = set(selected_nodes)

        # safeguard DAG conversion; keeps behavior consistent with v3 export pipeline
        if kept_kvedges:
            kept_kvedges = base.break_cycles_to_dag_kv(kept_nodes, kept_kvedges, kvedges)

        if not kept_kvedges:
            sample["dag"] = {"kv_nodes": [], "adj": [], "meta": {"reason": "pcst_no_directed_edges", "pcst_fast": bool(HAS_PCST_FAST)}}
            out_samples.append(sample)
            continue

        outdeg = defaultdict(int)
        for eid in kept_kvedges:
            e = kvedges[eid]
            outdeg[e.src] += 1
        sinks = [n for n in kept_nodes if outdeg.get(n, 0) == 0]

        ansN = base.norm_match(answer)
        answer_matched = False
        if ansN:
            for sink in sinks:
                for eid in kept_kvedges:
                    e = kvedges[eid]
                    if e.dst == sink and answer_matches_text(answer, e.value):
                        answer_matched = True
                        break
                if answer_matched:
                    break
        if answer_matched:
            answer_recall += 1

        for eid in kept_kvedges:
            if answer_matches_text(answer, kvedges[eid].value):
                none_sink_recall += 1
                break

        kv_nodes, adj = base.export_kv_nodes_and_adj(kept_kvedges, kvedges, keep_score=args.keep_score)

        goal_ids: List[int] = []
        if ansN:
            for i, kv in enumerate(kv_nodes):
                if answer_matches_text(answer, kv.get("value", "")):
                    goal_ids.append(i)

        meta = {
            "num_entity_nodes": int(len(kept_nodes)),
            "num_kv_edges": int(len(kept_kvedges)),
            "num_kv_nodes": int(len(kv_nodes)),
            "goal_ids": goal_ids,
            "pcst_root": int(pcst_root),
            "pcst_root_name": node_names[pcst_root] if 0 <= pcst_root < len(node_names) else "",
            "num_sinks": int(len(sinks)),
            "num_anchors": int(len(anchors)),
            "pcst_fast": bool(HAS_PCST_FAST),
        }

        sample["dag"] = {"kv_nodes": kv_nodes, "adj": adj, "meta": meta}
        out_samples.append(sample)

        if args.verbose:
            base.print_current_graph_kv(anchors, kept_nodes, kept_kvedges, kvedges, node_names=node_names)

    if len(samples) > 0:
        print(f"Answer recall: {answer_recall / len(samples):.4f}")
        print(f"Graph  recall: {graph_recall  / len(samples):.4f}")
        print(f"None-sink recall: {none_sink_recall / len(samples):.4f}")

    return out_samples


# ============================================================
# 5) CLI
# ============================================================
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
    ap.add_argument("--limit", type=int, default=None, help="limit number of samples")
    ap.add_argument("--supporting_only", action="store_true", help="only use triples from supporting_facts titles")
    ap.add_argument("--verbose", action="store_true", help="verbose debug")

    # graph build / compatibility knobs
    ap.add_argument("--pred_weight", type=float, default=0.0, help="weight for relation score sim(question, relation)")
    ap.add_argument("--max_nodes", type=int, default=30, help="max entity/value nodes in retrieved subgraph")
    ap.add_argument("--max_edges", type=int, default=40, help="max KVEdges in retrieved DAG")
    ap.add_argument("--keep_one_attr_direction", action="store_true", help="keep only one direction for ATTRIBUTE pairs")
    ap.add_argument("--eps_dir", type=float, default=0.05, help="direction confidence threshold")
    ap.add_argument("--max_attr_out_per_entity", type=int, default=None, help="cap attribute out edges per entity")

    # closer-to-retrieval.py knobs
    ap.add_argument("--topk_node_prize", type=int, default=6, help="top-k nodes that receive node prizes, like retrieval.py")
    ap.add_argument("--topk_edge_prize", type=int, default=12, help="top-k unique edge sim levels that receive edge prizes, like retrieval.py")
    ap.add_argument("--pcst_cost_e", type=float, default=0.5, help="base edge cost in retrieval.py-style PCST")
    ap.add_argument("--anchor_bonus", type=float, default=1.0, help="extra node prize for question-mentioned anchors")
    ap.add_argument("--node_sim_weight", type=float, default=1.0, help="scale node prizes")
    ap.add_argument("--edge_sim_weight", type=float, default=1.0, help="scale edge prizes")

    # DAG orientation knobs
    ap.add_argument("--leaf_bonus", type=float, default=0.35, help="bonus for orienting edges toward value-like leaf nodes")
    ap.add_argument("--anchor_penalty", type=float, default=0.25, help="penalty for making mentioned anchor nodes into leaves")
    ap.add_argument("--internal_value_penalty", type=float, default=0.10, help="penalty for keeping value-like nodes as internal nodes")

    args = ap.parse_args()
    print(args)

    embedder = SentenceTransformer(args.st_model)
    samples = base.read_json_or_jsonl(args.input)
    print(f"Load {len(samples)} samples from {args.input}")
    if HAS_PCST_FAST:
        print("[INFO] pcst_fast is available: using retrieval.py-style PCST.")
    else:
        print("[INFO] pcst_fast is NOT available: using Steiner-style fallback.")

    out = create_dag(args, samples, embedder)
    if not out:
        print("No valid output.")
        return

    if args.output.endswith(".jsonl"):
        base.write_jsonl(args.output, out)
    elif args.output.endswith(".json"):
        base.write_json(args.output, out)
    else:
        raise ValueError(f"Unknown file format: {args.output}")

    print(f"[DONE] input={len(samples)}  output={len(out)}  saved_to={args.output}")


if __name__ == "__main__":
    main()
