import argparse
import math
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import DAG_KV_Retriever_v3 as base


# ============================================================
# 0) Small helpers
# ============================================================
def safe_cost_from_score(score: float) -> float:
    """Map retrieval score -> positive edge cost for PCST/Steiner."""
    # cosine-like scores may be negative; keep costs positive and smooth
    return float(max(1e-4, 1.25 - score))


def is_value_like_node(node_id: int, node_names: List[str], kvedges: Dict[int, base.KVEdge]) -> bool:
    name = node_names[node_id] if 0 <= node_id < len(node_names) else ""
    n = base.norm_match(name)
    if not n:
        return False
    if any(ch.isdigit() for ch in n):
        return True
    toks = n.split()
    if len(toks) <= 3 and n[0].islower():
        return True
    # destination of attribute-like edges is often value-like
    in_attr = 0
    total_in = 0
    for e in kvedges.values():
        if e.dst == node_id:
            total_in += 1
            if (e.triple_type or "").upper() == "ATTRIBUTE":
                in_attr += 1
    return total_in > 0 and in_attr >= max(1, total_in // 2)


def answer_matches_text(answer: str, text: str) -> bool:
    ansN = base.norm_match(answer)
    txtN = base.norm_match(text)
    return bool(ansN and txtN and (ansN == txtN or ansN in txtN or txtN in ansN))


# ============================================================
# 1) Question-aware node prizes (G-Retriever spirit)
# ============================================================
def compute_node_prizes(
    question: str,
    node_names: List[str],
    kvedges: Dict[int, base.KVEdge],
    embedder: SentenceTransformer,
    batch_size: int,
    prize_alpha: float = 1.0,
    mention_bonus: float = 1.2,
    edge_bonus: float = 0.5,
    bridge_bonus: float = 0.15,
    value_penalty: float = 0.15,
) -> Tuple[Dict[int, float], Dict[int, float], Set[int]]:
    if not node_names:
        return {}, {}, set()

    q_emb = base.embed_texts(embedder, [question], batch_size=batch_size)[0:1]
    node_emb = base.embed_texts(embedder, node_names, batch_size=batch_size)
    node_sims = base.cosine_sim_matrix(q_emb, node_emb).reshape(-1)
    q_norm = base._norm_lex(question)

    deg = defaultdict(int)
    best_edge = defaultdict(lambda: -1e9)
    anchors: Set[int] = set()
    for nid, name in enumerate(node_names):
        if base._contains_mention(q_norm, name):
            anchors.add(nid)
    for e in kvedges.values():
        deg[e.src] += 1
        deg[e.dst] += 1
        best_edge[e.src] = max(best_edge[e.src], e.score)
        best_edge[e.dst] = max(best_edge[e.dst], e.score)

    prizes: Dict[int, float] = {}
    sim_map: Dict[int, float] = {}
    for nid, sim in enumerate(node_sims):
        sim01 = float(max(0.0, sim))
        sim_map[nid] = sim01
        prize = prize_alpha * sim01
        if nid in anchors:
            prize += mention_bonus
        prize += edge_bonus * max(0.0, float(best_edge[nid]) if best_edge[nid] > -1e8 else 0.0)
        prize += bridge_bonus * min(1.0, math.log1p(deg[nid]) / math.log(6.0))
        if is_value_like_node(nid, node_names, kvedges) and nid not in anchors:
            prize -= value_penalty
        prizes[nid] = float(prize)

    return prizes, sim_map, anchors


# ============================================================
# 2) Undirected graph for PCST-style retrieval
# ============================================================
def build_pcst_graph(
    node_names: List[str],
    kvedges: Dict[int, base.KVEdge],
) -> Tuple[nx.Graph, Dict[Tuple[int, int], List[int]], Dict[Tuple[int, int], float]]:
    G = nx.Graph()
    for nid, name in enumerate(node_names):
        G.add_node(nid, name=name)

    pair_to_eids: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    pair_best_cost: Dict[Tuple[int, int], float] = {}
    for eid, e in kvedges.items():
        if e.src == e.dst:
            continue
        pair = (min(e.src, e.dst), max(e.src, e.dst))
        pair_to_eids[pair].append(eid)
        c = safe_cost_from_score(e.score)
        if pair not in pair_best_cost or c < pair_best_cost[pair]:
            pair_best_cost[pair] = c

    for (u, v), cost in pair_best_cost.items():
        G.add_edge(u, v, weight=float(cost))

    return G, pair_to_eids, pair_best_cost


# ============================================================
# 3) Greedy PCST-like terminal selection + Steiner tree
# ============================================================
def choose_pcst_terminals(
    G: nx.Graph,
    prizes: Dict[int, float],
    anchors: Set[int],
    max_nodes: int,
    terminal_budget: int,
    min_gain: float,
) -> Tuple[List[int], int]:
    if G.number_of_nodes() == 0:
        return [], -1

    if anchors:
        root = max(anchors, key=lambda n: prizes.get(n, -1e9))
        terminals: List[int] = sorted(anchors, key=lambda n: prizes.get(n, 0.0), reverse=True)
    else:
        root = max(prizes, key=prizes.get)
        terminals = [root]

    selected: Set[int] = set(terminals)
    frontier: Set[int] = set(terminals)
    if not frontier:
        frontier = {root}
        selected = {root}
        terminals = [root]

    # candidate order by node prize; non-anchor value-like nodes can still enter if connection is cheap
    candidates = [n for n in sorted(prizes, key=prizes.get, reverse=True) if n not in selected]
    while len(selected) < max(2, terminal_budget):
        best_gain = min_gain
        best_node = None
        for n in candidates:
            if n in selected:
                continue
            if not nx.has_path(G, root, n):
                continue
            try:
                dist = min(
                    nx.shortest_path_length(G, source=s, target=n, weight="weight")
                    for s in frontier
                    if nx.has_path(G, s, n)
                )
            except ValueError:
                continue
            gain = prizes.get(n, 0.0) - float(dist)
            if gain > best_gain:
                best_gain = gain
                best_node = n
        if best_node is None:
            break
        selected.add(best_node)
        frontier.add(best_node)
        terminals.append(best_node)
        if len(selected) >= max_nodes:
            break
    return terminals, root



def prune_tree_to_budget(
    tree: nx.Graph,
    prizes: Dict[int, float],
    anchors: Set[int],
    max_nodes: int,
) -> nx.Graph:
    tree = tree.copy()
    if tree.number_of_nodes() <= max_nodes:
        return tree
    while tree.number_of_nodes() > max_nodes:
        leaves = [n for n in tree.nodes if tree.degree[n] <= 1 and n not in anchors]
        if not leaves:
            break
        worst = min(leaves, key=lambda n: prizes.get(n, -1e9))
        tree.remove_node(worst)
    return tree



def retrieve_pcst_subgraph(
    node_names: List[str],
    kvedges: Dict[int, base.KVEdge],
    prizes: Dict[int, float],
    anchors: Set[int],
    max_nodes: int,
    max_edges: int,
    terminal_budget: int,
    min_gain: float,
) -> Tuple[Set[int], Set[Tuple[int, int]], int, List[int]]:
    G, _, _ = build_pcst_graph(node_names, kvedges)
    if G.number_of_edges() == 0:
        return set(), set(), -1, []

    terminals, root = choose_pcst_terminals(
        G=G,
        prizes=prizes,
        anchors=anchors,
        max_nodes=max_nodes,
        terminal_budget=terminal_budget,
        min_gain=min_gain,
    )
    if not terminals:
        return set(), set(), -1, []

    if len(terminals) == 1:
        nodes = {terminals[0]}
        edges: Set[Tuple[int, int]] = set()
        return nodes, edges, root, terminals

    # G-Retriever uses PCST; here we use a pragmatic PCST-like terminal growth and then weighted Steiner tree.
    tree = nx.algorithms.approximation.steiner_tree(G, terminals, weight="weight")
    tree = prune_tree_to_budget(tree, prizes, anchors, max_nodes=max_nodes)

    # further compress if the edge budget is exceeded: repeatedly drop the weakest non-anchor leaf
    while tree.number_of_edges() > max_edges:
        leaves = [n for n in tree.nodes if tree.degree[n] <= 1 and n not in anchors]
        if not leaves:
            break
        worst = min(leaves, key=lambda n: prizes.get(n, -1e9))
        tree.remove_node(worst)

    undirected_edges = {(min(u, v), max(u, v)) for u, v in tree.edges()}
    return set(tree.nodes()), undirected_edges, root, terminals


# ============================================================
# 4) Rooted orientation: tree -> DAG
# ============================================================
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
        dist.setdefault(n, 10**9)
    return dist



def select_directed_edges_from_tree(
    tree_nodes: Set[int],
    undirected_edges: Set[Tuple[int, int]],
    root: int,
    node_names: List[str],
    kvedges: Dict[int, base.KVEdge],
    sim_map: Dict[int, float],
    anchors: Set[int],
    leaf_bonus: float = 0.35,
    anchor_penalty: float = 0.25,
) -> Set[int]:
    if not tree_nodes:
        return set()

    dist = compute_root_distances(tree_nodes, undirected_edges, root=root)
    pair_to_dir: Dict[Tuple[int, int, int, int], List[int]] = defaultdict(list)
    undir_to_all: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for eid, e in kvedges.items():
        pair = (min(e.src, e.dst), max(e.src, e.dst))
        if pair in undirected_edges:
            undir_to_all[pair].append(eid)
            pair_to_dir[(pair[0], pair[1], e.src, e.dst)].append(eid)

    kept: Set[int] = set()
    for u, v in undirected_edges:
        du, dv = dist.get(u, 10**9), dist.get(v, 10**9)
        if du <= dv:
            src_pref, dst_pref = u, v
        else:
            src_pref, dst_pref = v, u

        preferred = [eid for eid in undir_to_all[(u, v)] if kvedges[eid].src == src_pref and kvedges[eid].dst == dst_pref]
        fallback = undir_to_all[(u, v)]
        cand = preferred if preferred else fallback

        def orient_score(eid: int) -> float:
            e = kvedges[eid]
            dst = e.dst
            val_like = is_value_like_node(dst, node_names, kvedges)
            bonus = 0.0
            if val_like and dst not in anchors:
                bonus += leaf_bonus
            if dst in anchors and dst != root:
                bonus -= anchor_penalty
            bonus += 0.10 * sim_map.get(dst, 0.0)
            return float(e.score + bonus)

        best_eid = max(cand, key=orient_score)
        kept.add(best_eid)
    return kept


# ============================================================
# 5) Main pipeline
# ============================================================
def create_dag(args, samples: List[Dict[str, Any]], embedder: SentenceTransformer) -> List[Dict[str, Any]]:
    if args.limit is not None and args.limit > 0:
        samples = samples[:args.limit]

    out_samples: List[Dict[str, Any]] = []
    answer_recall = 0
    graph_recall = 0
    none_sink_recall = 0

    for sample in tqdm(samples, desc="Create DAG (PCST)"):
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

        # full-graph recall before PCST retrieval
        for e in kvedges.values():
            if answer_matches_text(answer, e.value):
                graph_recall += 1
                break

        prizes, sim_map, anchors = compute_node_prizes(
            question=question,
            node_names=node_names,
            kvedges=kvedges,
            embedder=embedder,
            batch_size=args.batch_size,
            prize_alpha=args.node_prize_alpha,
            mention_bonus=args.anchor_mention_bonus,
            edge_bonus=args.node_edge_bonus,
            bridge_bonus=args.node_bridge_bonus,
            value_penalty=args.value_node_penalty,
        )

        kept_nodes, undirected_edges, root, terminals = retrieve_pcst_subgraph(
            node_names=node_names,
            kvedges=kvedges,
            prizes=prizes,
            anchors=anchors,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            terminal_budget=args.pcst_terminal_budget,
            min_gain=args.pcst_min_gain,
        )

        if not kept_nodes:
            sample["dag"] = {"kv_nodes": [], "adj": [], "meta": {"reason": "pcst_empty"}}
            out_samples.append(sample)
            continue

        if root < 0:
            root = max(kept_nodes, key=lambda n: prizes.get(n, -1e9))

        kept_kvedges = select_directed_edges_from_tree(
            tree_nodes=kept_nodes,
            undirected_edges=undirected_edges,
            root=root,
            node_names=node_names,
            kvedges=kvedges,
            sim_map=sim_map,
            anchors=anchors,
            leaf_bonus=args.leaf_bonus,
            anchor_penalty=args.anchor_penalty,
        )

        if not kept_kvedges:
            sample["dag"] = {"kv_nodes": [], "adj": [], "meta": {"reason": "pcst_no_directed_edges"}}
            out_samples.append(sample)
            continue

        # tree orientation is already acyclic; this is just a safeguard.
        kept_kvedges = base.break_cycles_to_dag_kv(kept_nodes, kept_kvedges, kvedges)
        if not kept_kvedges:
            sample["dag"] = {"kv_nodes": [], "adj": [], "meta": {"reason": "empty_after_cycle_break"}}
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
            "pcst_root": int(root),
            "pcst_root_name": node_names[root] if 0 <= root < len(node_names) else "",
            "pcst_terminal_nodes": sorted(int(x) for x in terminals),
            "num_sinks": int(len(sinks)),
        }

        sample["dag"] = {"kv_nodes": kv_nodes, "adj": adj, "meta": meta}
        out_samples.append(sample)

        if args.verbose:
            seed_like = set(terminals[: max(1, min(len(terminals), 5))])
            base.print_current_graph_kv(seed_like, kept_nodes, kept_kvedges, kvedges, node_names=node_names)

    if len(samples) > 0:
        print(f"Answer recall: {answer_recall / len(samples):.4f}")
        print(f"Graph  recall: {graph_recall  / len(samples):.4f}")
        print(f"None-sink recall: {none_sink_recall / len(samples):.4f}")

    return out_samples


# ============================================================
# 6) CLI
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
    ap.add_argument("--max_nodes", type=int, default=30, help="max entity/value nodes in retrieved PCST subgraph")
    ap.add_argument("--max_edges", type=int, default=40, help="max KVEdges in retrieved DAG")
    ap.add_argument("--keep_one_attr_direction", action="store_true", help="keep only one direction for ATTRIBUTE pairs")
    ap.add_argument("--eps_dir", type=float, default=0.05, help="direction confidence threshold")
    ap.add_argument("--max_attr_out_per_entity", type=int, default=None, help="cap attribute out edges per entity")

    # PCST-like retrieval knobs
    ap.add_argument("--pcst_terminal_budget", type=int, default=8, help="max number of prize terminals before Steiner connection")
    ap.add_argument("--pcst_min_gain", type=float, default=0.05, help="minimum prize-minus-cost gain to add a new terminal")
    ap.add_argument("--node_prize_alpha", type=float, default=1.0, help="weight for node-question similarity in node prize")
    ap.add_argument("--anchor_mention_bonus", type=float, default=1.2, help="bonus for nodes explicitly mentioned in the question")
    ap.add_argument("--node_edge_bonus", type=float, default=0.5, help="bonus from best incident edge relevance")
    ap.add_argument("--node_bridge_bonus", type=float, default=0.15, help="bonus for structurally central nodes")
    ap.add_argument("--value_node_penalty", type=float, default=0.15, help="small penalty to avoid over-selecting literal-like nodes as internal PCST nodes")
    ap.add_argument("--leaf_bonus", type=float, default=0.35, help="bonus for orienting edges toward value-like leaf nodes")
    ap.add_argument("--anchor_penalty", type=float, default=0.25, help="penalty for making mentioned anchor nodes into leaves")

    args = ap.parse_args()
    print(args)

    embedder = SentenceTransformer(args.st_model)
    samples = base.read_json_or_jsonl(args.input)
    print(f"Load {len(samples)} samples from {args.input}")

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
