#!/usr/bin/env python3
"""Standalone, answer-blind DAG-KV inference.

This module intentionally contains no training path and imports no earlier graph-gen
script. Gold answers are preserved as opaque output fields but are never accessed by
the graph construction, scoring, pruning, or reporting code.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

# -----------------------------------------------------------------------------
# IO and text normalization
# -----------------------------------------------------------------------------

def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        return obj["data"]
    raise ValueError("Expected JSON list, {data: [...]}, or JSONL input")


def write_rows(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    if path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(rows), f, ensure_ascii=False, indent=2)
        return
    raise ValueError("Output must end with .json or .jsonl")


_SPACE_RE = re.compile(r"\s+")
_ZW_RE = re.compile(r"[\u200b\u200c\u200d\uFEFF]")
_PAREN_RE = re.compile(r"\([^)]*\)")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "for", "to", "from", "by",
    "with", "is", "was", "were", "are", "be", "been", "being", "and", "or",
    "that", "this", "these", "those", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "as", "into", "about",
}


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = _SPACE_RE.sub(" ", _ZW_RE.sub("", str(value)).strip())
    left, right = text.count("("), text.count(")")
    if left > right:
        text += ")" * (left - right)
    elif right > left:
        extra = right - left
        while extra and text.endswith(")"):
            text, extra = text[:-1], extra - 1
    return text


def normalize_lex(text: str) -> str:
    toks = _PUNCT_RE.sub(" ", (text or "").lower()).split()
    return " ".join(t for t in toks if t and t not in _STOPWORDS)


def token_set(text: str) -> Set[str]:
    return set(normalize_lex(text).split())


def jaccard(left: str, right: str) -> float:
    a, b = token_set(left), token_set(right)
    return len(a & b) / float(len(a | b)) if a and b else 0.0


def entity_alias_keys(name: str) -> List[str]:
    key = _SPACE_RE.sub(
        " ", _PUNCT_RE.sub(" ", _PAREN_RE.sub("", norm_text(name)).lower())
    ).strip()
    if not key:
        return [""]
    toks = key.split()
    keys = [key]
    if len(toks) >= 3:
        keys.append(" ".join(toks[:-1]))
    if len(toks) >= 2 and toks[-1] in {"jr", "sr", "ii", "iii", "iv"}:
        keys.append(" ".join(toks[:-1]))
    return list(dict.fromkeys(keys))


def contains_mention(question: str, name: str) -> bool:
    q, n = normalize_lex(question), normalize_lex(name)
    if not n:
        return False
    toks = n.split()
    return (
        n in q
        or (len(toks) >= 2 and " ".join(toks[:2]) in q)
        or (len(toks) == 1 and len(toks[0]) >= 5 and toks[0] in q)
    )


# -----------------------------------------------------------------------------
# Graph and features
# -----------------------------------------------------------------------------

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
    kv_offset: Optional[int] = None


def iter_triples(sample: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    yielded = False
    for para in sample.get("context", []) or []:
        if isinstance(para, dict):
            title, triples = norm_text(para.get("title", "")), para.get("triple_list", []) or []
        elif isinstance(para, list):
            title = norm_text(para[0]) if para else ""
            triples = para[2] if len(para) >= 3 else []
        else:
            continue
        for triple in triples:
            yielded = True
            yield title, triple
    if not yielded:
        for triple in sample.get("triple_list", []) or []:
            title = norm_text(triple.get("title", "") if isinstance(triple, dict) else "")
            yield title, triple


def build_graph(sample: Dict[str, Any]):
    node_map: Dict[str, int] = {}
    names: List[str] = []
    edges: Dict[int, KVEdge] = {}
    out_adj: Dict[int, List[int]] = defaultdict(list)
    in_adj: Dict[int, List[int]] = defaultdict(list)
    seen: Set[Tuple[str, str]] = set()

    def node_id(name: str) -> int:
        clean = norm_text(name)
        for key in entity_alias_keys(clean):
            if key in node_map:
                return node_map[key]
        nid = len(names)
        names.append(clean)
        for key in entity_alias_keys(clean):
            if key:
                node_map[key] = nid
        return nid

    for title, triple in iter_triples(sample):
        if not isinstance(triple, dict):
            continue
        subject = norm_text(triple.get("name", ""))
        obj = norm_text(triple.get("description", ""))
        if not subject or not obj:
            continue
        for kv_idx, kv in enumerate(triple.get("kv_lists", []) or []):
            key = norm_text(kv.get("key_string", ""))
            value = norm_text(kv.get("value_string", ""))
            if not key or not value or (key, value) in seen:
                continue
            seen.add((key, value))
            src_name, dst_name = (subject, obj) if kv_idx % 2 == 0 else (obj, subject)
            src, dst = node_id(src_name), node_id(dst_name)
            eid = len(edges)
            edges[eid] = KVEdge(
                eid, src, dst, names[src], names[dst], key, value, 0.0, 0.0,
                norm_text(title), norm_text(triple.get("type", "")),
                norm_text(triple.get("description_type", "")), subject, obj, kv_idx,
                int(kv["kv_offset"]) if kv.get("kv_offset") is not None else None,
            )
            out_adj[src].append(eid)
            in_adj[dst].append(eid)
    return names, edges, dict(out_adj), dict(in_adj)


def encode_unique(
    embedder: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    prompt_name: Optional[str] = None,
):
    unique = list(dict.fromkeys(norm_text(x) for x in texts if norm_text(x)))
    if not unique:
        return {}
    encode_kwargs = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    if prompt_name:
        encode_kwargs["prompt_name"] = prompt_name
    vectors = embedder.encode(unique, **encode_kwargs)
    vectors = np.asarray(vectors, dtype=np.float32)
    return {text: vectors[i] for i, text in enumerate(unique)}


def encode_groups(
    embedder: SentenceTransformer,
    groups: List[List[str]],
    batch_size: int,
    prompt_name: Optional[str] = None,
):
    """Deduplicate across all groups, then encode once in a single large batch stream."""
    merged: List[str] = []
    seen: Set[str] = set()
    for group in groups:
        for text in group:
            clean = norm_text(text)
            if clean and clean not in seen:
                seen.add(clean)
                merged.append(clean)
    return encode_unique(embedder, merged, batch_size, prompt_name=prompt_name)


def load_text_embedder(
    model_path: str,
    profile: str,
    *,
    cpu: bool,
) -> tuple[SentenceTransformer, Optional[str]]:
    if profile == "sentence-transformer":
        device = "cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        return SentenceTransformer(model_path, device=device), None
    if profile != "qwen3-embedding-v2":
        raise ValueError(f"Unsupported --st_encoding_profile: {profile}")

    # Keep the DAG-side encode path aligned with Store KV base embedding generation.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    model = SentenceTransformer(
        model_path,
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    device = "cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    first_module = model._first_module() if hasattr(model, "_first_module") else None
    if first_module is not None and hasattr(first_module, "half"):
        first_module.half()
    return model, "query"


def identify_topics(question: str, names: List[str], node_emb: np.ndarray,
                    q_emb: np.ndarray, top_k: int, mention_bonus: float) -> List[int]:
    scores = (q_emb[None, :] @ node_emb.T).reshape(-1)
    for i, name in enumerate(names):
        scores[i] += mention_bonus * float(contains_mention(question, name))
        scores[i] += 0.15 * jaccard(question, name)
    order = np.argsort(-scores)
    return [int(i) for i in order[:max(1, min(top_k, len(order)))]]


def compute_dde(num_nodes: int, out_adj: Dict[int, List[int]], in_adj: Dict[int, List[int]],
                edges: Dict[int, KVEdge], topics: List[int], hops: int) -> np.ndarray:
    seed = np.zeros(num_nodes, dtype=np.float32)
    seed[topics] = 1.0
    feats, cur = [seed], seed.copy()
    for _ in range(hops):
        nxt = np.zeros_like(cur)
        for edge in edges.values():
            nxt[edge.dst] += cur[edge.src]
        for dst, eids in in_adj.items():
            nxt[dst] /= max(1, len(eids))
        feats.append(nxt)
        cur = nxt
    cur = seed.copy()
    for _ in range(hops):
        nxt = np.zeros_like(cur)
        for edge in edges.values():
            nxt[edge.src] += cur[edge.dst]
        for src, eids in out_adj.items():
            nxt[src] /= max(1, len(eids))
        feats.append(nxt)
        cur = nxt
    return np.stack(feats, axis=1).astype(np.float32)


def build_features(question: str, names: List[str], edges: Dict[int, KVEdge],
                   out_adj: Dict[int, List[int]], in_adj: Dict[int, List[int]],
                   cache: Dict[str, np.ndarray], topic_top_k: int, dde_hops: int,
                   mention_bonus: float):
    q = cache[norm_text(question)]
    node_emb = np.stack([cache[norm_text(x)] for x in names]).astype(np.float32)
    topics = identify_topics(question, names, node_emb, q, topic_top_k, mention_bonus)
    dde = compute_dde(len(names), out_adj, in_adj, edges, topics, dde_hops)
    topic_set = set(topics)
    edge_feats: Dict[int, Dict[str, np.ndarray]] = {}
    for eid in sorted(edges):
        edge = edges[eid]
        src, dst = node_emb[edge.src], node_emb[edge.dst]
        rel = cache[norm_text(edge.relation or edge.key)]
        key, value = cache[norm_text(edge.key)], cache[norm_text(edge.value)]
        scalar = np.array([
            q @ src, q @ dst, q @ rel, q @ key, q @ value, src @ dst,
            (q @ dst) - (q @ src), float(edge.src in topic_set),
            float(edge.dst in topic_set), float(contains_mention(question, edge.src_name)),
            float(contains_mention(question, edge.dst_name)), jaccard(question, edge.relation),
            jaccard(question, edge.key), jaccard(question, edge.value),
            float(edge.triple_type.upper() == "ATTRIBUTE"),
        ], dtype=np.float32)
        vector = np.concatenate([q, src, rel, dst, key, value, dde[edge.src], dde[edge.dst], scalar])
        edge_feats[eid] = {"vector": vector.astype(np.float32), "scalar": scalar, "q": q}

    node_feats: Dict[int, np.ndarray] = {}
    for nid, name in enumerate(names):
        incoming, outgoing = in_adj.get(nid, []), out_adj.get(nid, [])
        zero = np.zeros(15, dtype=np.float32)
        in_mean = np.mean([edge_feats[e]["scalar"] for e in incoming], axis=0) if incoming else zero
        out_mean = np.mean([edge_feats[e]["scalar"] for e in outgoing], axis=0) if outgoing else zero
        maxv = lambda ids, col: max((float(edge_feats[e]["scalar"][col]) for e in ids), default=0.0)
        scalar = np.array([
            q @ node_emb[nid], float(nid in topic_set), float(contains_mention(question, name)),
            jaccard(question, name), len(incoming), len(outgoing), maxv(incoming, 4),
            maxv(incoming, 1), maxv(outgoing, 4), maxv(outgoing, 1),
        ], dtype=np.float32)
        node_feats[nid] = np.concatenate([q, node_emb[nid], scalar, in_mean, out_mean]).astype(np.float32)
    return topics, edge_feats, node_feats


# -----------------------------------------------------------------------------
# Checkpoint-compatible scorers
# -----------------------------------------------------------------------------

class MLPScorer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim // 2, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


class NodeEndScorer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        hid2 = max(64, hidden_dim // 2)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, hid2), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hid2, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


class SharedEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout))
        self.output_dim = hidden_dim // 2
    def forward(self, x):
        return self.net(x)


class EdgeHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float):
        super().__init__()
        hid = max(64, in_dim // 2)
        self.head = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid, 1))
    def forward(self, x):
        return self.head(x).squeeze(-1)


class NodeHead(EdgeHead):
    pass


class JointScorer(nn.Module):
    def __init__(self, edge_in_dim: int, node_in_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.edge_input_proj = nn.Linear(edge_in_dim, hidden_dim)
        self.node_input_proj = nn.Linear(node_in_dim, hidden_dim)
        self.shared_encoder = SharedEncoder(hidden_dim, hidden_dim, dropout)
        self.edge_head = EdgeHead(self.shared_encoder.output_dim, dropout)
        self.node_head = NodeHead(self.shared_encoder.output_dim, dropout)
    def forward_edge(self, x):
        return self.edge_head(self.shared_encoder(self.edge_input_proj(x)))
    def forward_node(self, x):
        return self.node_head(self.shared_encoder(self.node_input_proj(x)))


def load_models(path: str, cpu: bool):
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    ckpt = torch.load(path, map_location="cpu")
    if ckpt.get("is_joint", False):
        model = JointScorer(ckpt["edge_input_dim"], ckpt["node_input_dim"],
                            ckpt.get("hidden_dim", 512), ckpt.get("dropout", 0.1))
        model.load_state_dict(ckpt["joint_state_dict"])
        model.to(device).eval()
        return model, model, ckpt, device
    edge = MLPScorer(ckpt["edge_input_dim"], ckpt.get("hidden_dim", 512), ckpt.get("dropout", 0.1))
    node = NodeEndScorer(ckpt["node_input_dim"], ckpt.get("end_hidden_dim", 256), ckpt.get("dropout", 0.1))
    edge.load_state_dict(ckpt["edge_state_dict"])
    node.load_state_dict(ckpt["node_state_dict"])
    edge.to(device).eval(); node.to(device).eval()
    return edge, node, ckpt, device


def sigmoid_scores(model: nn.Module, device: torch.device, x: np.ndarray, batch_size: int,
                   forward: Optional[Callable] = None) -> np.ndarray:
    out = np.empty(len(x), dtype=np.float32)
    fn = forward or model
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size])).to(device)
            out[start:start + len(batch)] = torch.sigmoid(fn(batch)).cpu().numpy().reshape(-1)
    return out


# -----------------------------------------------------------------------------
# Answer-free DAG selection
# -----------------------------------------------------------------------------

def apply_joint_scores(edges: Dict[int, KVEdge], edge_scores: Dict[int, float],
                       end_scores: Dict[int, float], alpha: float, beta: float, gamma: float):
    incoming_best: Dict[int, float] = defaultdict(float)
    for eid, edge in edges.items():
        incoming_best[edge.dst] = max(incoming_best[edge.dst], edge_scores.get(eid, 0.0))
    for eid, edge in edges.items():
        edge.edge_score = float(edge_scores.get(eid, 0.0))
        src_end, dst_end = end_scores.get(edge.src, 0.0), end_scores.get(edge.dst, 0.0)
        edge.score = float(edge.edge_score + alpha * dst_end - beta * src_end
                           - gamma * src_end * incoming_best.get(edge.src, 0.0))


def select_edges(topics: List[int], edges: Dict[int, KVEdge], out_adj: Dict[int, List[int]], args):
    selected, nodes = set(), set(topics)
    ranked = sorted(edges, key=lambda eid: edges[eid].score, reverse=True)
    per_src: Dict[int, int] = defaultdict(int)
    for eid in ranked:
        edge = edges[eid]
        if per_src[edge.src] >= args.per_src_cap:
            continue
        new = int(edge.src not in nodes) + int(edge.dst not in nodes)
        if len(nodes) + new > args.max_nodes:
            continue
        selected.add(eid); nodes.update((edge.src, edge.dst)); per_src[edge.src] += 1
        if len(selected) >= min(args.seed_edge_topk, args.max_edges):
            break
    frontier, seen, hop = deque(topics), set(topics), 0
    while frontier and hop < args.expansion_hops and len(selected) < args.max_edges:
        for _ in range(len(frontier)):
            src, local = frontier.popleft(), 0
            for eid in sorted(out_adj.get(src, []), key=lambda x: edges[x].score, reverse=True):
                if eid in selected:
                    continue
                if local >= args.per_src_cap:
                    break
                edge = edges[eid]
                new = int(edge.src not in nodes) + int(edge.dst not in nodes)
                if len(nodes) + new > args.max_nodes:
                    continue
                selected.add(eid); nodes.update((edge.src, edge.dst)); local += 1
                if edge.dst not in seen:
                    seen.add(edge.dst); frontier.append(edge.dst)
                # Preserve the checkpoint-era selection semantics: this breaks only
                # the current source's loop, so a frontier layer may slightly exceed
                # max_edges before the outer while condition is checked again.
                if len(selected) >= args.max_edges:
                    break
        hop += 1
    for eid in ranked:
        if len(selected) >= args.max_edges:
            break
        edge = edges[eid]
        new = int(edge.src not in nodes) + int(edge.dst not in nodes)
        if eid not in selected and (edge.src in nodes or edge.dst in nodes) and len(nodes) + new <= args.max_nodes:
            selected.add(eid); nodes.update((edge.src, edge.dst))
    return nodes, selected


def would_cycle(src: int, dst: int, kept_out: Dict[int, Set[int]]) -> bool:
    stack, seen = [dst], set()
    while stack:
        node = stack.pop()
        if node == src:
            return True
        if node not in seen:
            seen.add(node); stack.extend(kept_out.get(node, ()))
    return False


def break_cycles(nodes: Set[int], selected: Set[int], edges: Dict[int, KVEdge]) -> Set[int]:
    kept, kept_out = set(), defaultdict(set)
    for eid in sorted(selected, key=lambda x: edges[x].score, reverse=True):
        edge = edges[eid]
        if edge.src in nodes and edge.dst in nodes and not would_cycle(edge.src, edge.dst, kept_out):
            kept.add(eid); kept_out[edge.src].add(edge.dst)
    return kept


def reachable(topics: List[int], nodes: Set[int], selected: Set[int], edges: Dict[int, KVEdge]):
    out: Dict[int, List[int]] = defaultdict(list)
    for eid in selected:
        out[edges[eid].src].append(eid)
    seen, stack = set(), list(topics)
    while stack:
        node = stack.pop()
        if node in seen or node not in nodes:
            continue
        seen.add(node)
        stack.extend(edges[eid].dst for eid in out.get(node, []))
    return seen, {eid for eid in selected if edges[eid].src in seen and edges[eid].dst in seen}


def enforce_entity_sinks(topics, max_sinks, nodes, selected, edges):
    nodes, selected = reachable(topics, nodes, selected, edges)
    while selected and max_sinks > 0:
        outdeg = defaultdict(int)
        for eid in selected: outdeg[edges[eid].src] += 1
        sinks = [n for n in nodes if not outdeg[n]]
        if len(sinks) <= max_sinks: break
        weakest = min(sinks, key=lambda n: max(0.0, sum(edges[e].score for e in selected if edges[e].dst == n)))
        incoming = [e for e in selected if edges[e].dst == weakest]
        if incoming: selected.remove(min(incoming, key=lambda e: edges[e].score))
        else: nodes.remove(weakest)
        nodes, selected = reachable(topics, nodes, selected, edges)
    return nodes, selected


def terminal_feature_vector(eid, selected, edges, edge_feats, node_scores):
    """Answer-free terminal features used by the heuristic reranker."""
    edge = edges[eid]
    reverse_adj: Dict[int, List[int]] = defaultdict(list)
    for previous in selected:
        for current in selected:
            if previous != current and edges[previous].dst == edges[current].src:
                reverse_adj[current].append(previous)
    ancestors, stack = set(), [eid]
    while stack:
        current = stack.pop()
        if current in ancestors:
            continue
        ancestors.add(current)
        stack.extend(reverse_adj.get(current, []))
    key_sims = [float(edge_feats[x]["scalar"][3]) for x in ancestors]
    path_relevance = sum(key_sims) / max(1, len(key_sims))
    scalar = edge_feats[eid]["scalar"]
    return np.asarray([
        float(edge.score),
        float(edge.edge_score),
        float(node_scores.get(edge.dst, 0.0)),
        float(scalar[1]),
        float(scalar[3]),
        float(scalar[4]),
        float(scalar[12]),
        float(scalar[13]),
        path_relevance,
        max(key_sims, default=0.0),
        float(np.log1p(len(ancestors))),
    ], dtype=np.float32)


def terminal_rerank_score(eid, selected, edges, edge_feats, node_scores, args):
    """Question/path-aware terminal score that never accesses a gold answer."""
    features = terminal_feature_vector(eid, selected, edges, edge_feats, node_scores)
    return (
        float(features[0])
        + args.terminal_end_weight * float(features[2])
        + args.terminal_path_weight * float(features[8])
        + args.terminal_value_weight * float(features[5])
    )


def enforce_terminal_kv(topics, max_sinks, nodes, selected, edges,
                        edge_feats=None, node_scores=None, args=None):
    nodes, selected = reachable(topics, nodes, selected, edges)
    while selected and max_sinks > 0:
        outdeg = defaultdict(int)
        for eid in selected: outdeg[edges[eid].src] += 1
        terminals = [eid for eid in selected if not outdeg[edges[eid].dst]]
        if len(terminals) <= max_sinks: break
        if args is not None and args.terminal_reranker == "heuristic":
            bad_eid = min(
                terminals,
                key=lambda eid: terminal_rerank_score(
                    eid, selected, edges, edge_feats, node_scores, args
                ),
            )
        else:
            bad_eid = min(terminals, key=lambda eid: edges[eid].score)
        selected.remove(bad_eid)
        nodes, selected = reachable(topics, nodes, selected, edges)
    return nodes, selected


def reverse_expand(nodes, selected, edges, in_adj, args):
    if not selected or min(args.reverse_sink_edge_topk, args.reverse_sink_hops, args.reverse_sink_beam_width) <= 0:
        return nodes, selected
    outdeg = defaultdict(int)
    for eid in selected: outdeg[edges[eid].src] += 1
    sinks = sorted([e for e in selected if not outdeg[edges[e].dst]],
                   key=lambda e: (edges[e].score, e), reverse=True)[:args.reverse_sink_edge_topk]
    kept_out = defaultdict(set)
    for eid in selected: kept_out[edges[eid].src].add(edges[eid].dst)
    for sink in sinks:
        frontier, visited = [(sink, edges[sink].score)], {sink}
        for _ in range(args.reverse_sink_hops):
            candidates = []
            for current, path_score in frontier:
                for previous in in_adj.get(edges[current].src, []):
                    if previous not in visited:
                        visited.add(previous); candidates.append((path_score + edges[previous].score, previous))
            candidates.sort(key=lambda x: (x[0], edges[x[1]].score, -x[1]), reverse=True)
            frontier = []
            for score, eid in candidates[:args.reverse_sink_beam_width]:
                edge = edges[eid]
                new = int(edge.src not in nodes) + int(edge.dst not in nodes)
                if eid not in selected:
                    if len(selected) >= args.max_edges or len(nodes) + new > args.max_nodes or would_cycle(edge.src, edge.dst, kept_out):
                        continue
                    selected.add(eid); nodes.update((edge.src, edge.dst)); kept_out[edge.src].add(edge.dst)
                frontier.append((eid, score))
    return nodes, selected


def export_dag(selected: Set[int], edges: Dict[int, KVEdge], keep_score: bool):
    ordered = sorted(selected)
    index = {eid: i for i, eid in enumerate(ordered)}
    kv_nodes = []
    for eid in ordered:
        edge = edges[eid]
        row = {"key": edge.key, "value": edge.value, "src_entity": edge.src_name,
               "dst_entity": edge.dst_name, "title": edge.title, "triple_type": edge.triple_type,
               "relation": edge.relation, "kv_idx": edge.kv_idx}
        if edge.kv_offset is not None:
            row["kv_offset"] = edge.kv_offset
        if keep_score:
            row.update(edge_score=float(edge.edge_score), score=float(edge.score))
        kv_nodes.append(row)
    adj = [[0] * len(ordered) for _ in ordered]
    for left in ordered:
        for right in ordered:
            if left != right and edges[right].src == edges[left].dst:
                adj[index[left]][index[right]] = 1
    return kv_nodes, adj


# -----------------------------------------------------------------------------
# Batched inference
# -----------------------------------------------------------------------------

def score_and_export_contexts(args, contexts, cache, edge_model, node_model, ckpt,
                              device, batch_outputs, timing_stats=None):
    """Build/score only a bounded slice of samples to cap feature-block memory."""
    stage_started = time.perf_counter()
    prepared = []
    for context in contexts:
        batch_pos, sample, question, names, edges, out_adj, in_adj = context
        topics, edge_feats, node_feats = build_features(
            question, names, edges, out_adj, in_adj, cache,
            args.topic_top_k, args.dde_hops, args.mention_bonus,
        )
        prepared.append((context, topics, edge_feats, node_feats))
    if timing_stats is not None:
        timing_stats["feature_prepare"] += time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    edge_blocks, edge_meta, cursor = [], [], 0
    node_blocks, node_meta, node_cursor = [], [], 0
    for i, (_, _, edge_feats, node_feats) in enumerate(prepared):
        eids, nids = sorted(edge_feats), sorted(node_feats)
        ex = np.stack([edge_feats[e]["vector"] for e in eids]).astype(np.float32)
        nx = np.stack([node_feats[n] for n in nids]).astype(np.float32)
        edge_blocks.append(ex); edge_meta.append((i, eids, cursor, len(ex))); cursor += len(ex)
        node_blocks.append(nx); node_meta.append((i, nids, node_cursor, len(nx))); node_cursor += len(nx)
    edge_forward = edge_model.forward_edge if ckpt.get("is_joint", False) else None
    node_forward = node_model.forward_node if ckpt.get("is_joint", False) else None
    edge_all = sigmoid_scores(edge_model, device, np.concatenate(edge_blocks), args.infer_batch_size, edge_forward)
    node_all = sigmoid_scores(node_model, device, np.concatenate(node_blocks), args.infer_batch_size, node_forward)
    edge_scores, node_scores = [{} for _ in prepared], [{} for _ in prepared]
    for i, ids, pos, length in edge_meta:
        edge_scores[i] = {eid: float(edge_all[pos + j]) for j, eid in enumerate(ids)}
    for i, ids, pos, length in node_meta:
        node_scores[i] = {nid: float(node_all[pos + j]) for j, nid in enumerate(ids)}
    if timing_stats is not None:
        timing_stats["model_score"] += time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    for i, (context, topics, edge_feats, _) in enumerate(prepared):
        batch_pos, sample, question, names, edges, out_adj, in_adj = context
        apply_joint_scores(edges, edge_scores[i], node_scores[i], args.end_alpha, args.end_beta, args.end_gamma)
        nodes, selected = select_edges(topics, edges, out_adj, args)
        selected = break_cycles(nodes, selected, edges)
        nodes, selected = enforce_entity_sinks(topics, args.max_sinks, nodes, selected, edges)
        nodes, selected = enforce_terminal_kv(
            topics, args.max_sinks, nodes, selected, edges,
            edge_feats=edge_feats, node_scores=node_scores[i], args=args,
        )
        nodes, selected = reverse_expand(nodes, selected, edges, in_adj, args)
        kv_nodes, adj = export_dag(selected, edges, args.keep_score)
        row = dict(sample)
        row["dag"] = {"kv_nodes": kv_nodes, "adj": adj, "meta": {
            "num_entity_nodes": len(nodes), "num_kv_edges": len(selected),
            "num_kv_nodes": len(kv_nodes), "goal_ids": [],
            "topic_entity_ids": [int(x) for x in topics],
            "answer_free_inference": True,
            "scorer": "trainable_subgraphrag_mlp_v8_infer_only",
            "selection_mode": args.selection_mode,
            "terminal_reranker": args.terminal_reranker,
            "terminal_reranker_weights": {
                "end": args.terminal_end_weight,
                "path": args.terminal_path_weight,
                "value": args.terminal_value_weight,
            },
        }}
        batch_outputs[batch_pos] = row
    if timing_stats is not None:
        timing_stats["select_export"] += time.perf_counter() - stage_started


def infer(args, samples, embedder, edge_model, node_model, ckpt, device):
    outputs, _ = infer_profiled(args, samples, embedder, edge_model, node_model, ckpt, device)
    return outputs


def infer_profiled(args, samples, embedder, edge_model, node_model, ckpt, device):
    timing_stats = {
        "build_graph": 0.0,
        "encode": 0.0,
        "feature_prepare": 0.0,
        "model_score": 0.0,
        "select_export": 0.0,
        "total": 0.0,
    }
    total_started = time.perf_counter()
    outputs = []
    for start in tqdm(range(0, len(samples), args.infer_batch_size), desc="Create answer-blind DAG"):
        batch = samples[start:start + args.infer_batch_size]
        contexts, question_texts, doc_texts = [], [], []
        batch_outputs: List[Optional[Dict[str, Any]]] = [None] * len(batch)
        stage_started = time.perf_counter()
        for batch_pos, sample in enumerate(batch):
            question = norm_text(sample.get("question", ""))
            names, edges, out_adj, in_adj = build_graph(sample)
            if not edges:
                row = dict(sample)
                row["dag"] = {"kv_nodes": [], "adj": [], "meta": {"reason": "no_kv_edges", "answer_free_inference": True}}
                batch_outputs[batch_pos] = row
                continue
            current_docs = list(names)
            for edge in edges.values():
                current_docs.extend((norm_text(edge.relation or edge.key), norm_text(edge.key), norm_text(edge.value)))
            question_texts.append(question)
            doc_texts.extend(current_docs)
            contexts.append((batch_pos, sample, question, names, edges, out_adj, in_adj))
        timing_stats["build_graph"] += time.perf_counter() - stage_started
        if not contexts:
            outputs.extend(row for row in batch_outputs if row is not None)
            continue
        embedding_batch_size = args.embedding_batch_size or args.infer_batch_size
        stage_started = time.perf_counter()
        cache = encode_groups(
            embedder,
            [question_texts, doc_texts],
            embedding_batch_size,
            prompt_name=getattr(args, "st_prompt_name", None),
        )
        timing_stats["encode"] += time.perf_counter() - stage_started
        for feature_start in range(0, len(contexts), args.feature_batch_size):
            score_and_export_contexts(
                args, contexts[feature_start:feature_start + args.feature_batch_size],
                cache, edge_model, node_model, ckpt, device, batch_outputs, timing_stats,
            )
        if any(row is None for row in batch_outputs):
            raise RuntimeError("Internal error: an input row did not receive a DAG")
        outputs.extend(row for row in batch_outputs if row is not None)
    timing_stats["total"] = time.perf_counter() - total_started
    return outputs, timing_stats


def build_parser():
    ap = argparse.ArgumentParser(description="Standalone answer-blind DAG-KV inference")
    ap.add_argument("--mode", choices=["infer"], default="infer")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_ckpt", required=True)
    ap.add_argument("--st_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument(
        "--st_encoding_profile",
        choices=["sentence-transformer", "qwen3-embedding-v2"],
        default="sentence-transformer",
        help="How to load/encode --st_model. qwen3-embedding-v2 matches Store KV base encoding.",
    )
    ap.add_argument(
        "--st_prompt_name",
        default=None,
        help="Optional SentenceTransformer prompt_name. For qwen3-embedding-v2 this defaults to query.",
    )
    ap.add_argument("--infer_batch_size", type=int, default=4096)
    ap.add_argument("--embedding_batch_size", type=int, default=None,
                    help="Sentence-transformer encode batch size; defaults to infer_batch_size")
    ap.add_argument("--feature_batch_size", type=int, default=4096,
                    help="Maximum samples whose dense edge/node features coexist in memory")
    ap.add_argument("--keep_score", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true",
                    help="Append after validated existing JSONL rows and skip those input rows")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--topic_top_k", type=int, default=6)
    ap.add_argument("--dde_hops", type=int, default=3)
    ap.add_argument("--mention_bonus", type=float, default=0.2)
    ap.add_argument("--seed_edge_topk", type=int, default=18)
    ap.add_argument("--expansion_hops", type=int, default=2)
    ap.add_argument("--per_src_cap", type=int, default=3)
    ap.add_argument("--reverse_sink_edge_topk", type=int, default=3)
    ap.add_argument("--reverse_sink_hops", type=int, default=2)
    ap.add_argument("--reverse_sink_beam_width", type=int, default=4)
    ap.add_argument("--max_nodes", type=int, default=30)
    ap.add_argument("--max_edges", type=int, default=40)
    ap.add_argument("--max_sinks", type=int, default=8)
    ap.add_argument("--selection_mode", choices=["legacy"], default="legacy")
    ap.add_argument("--end_alpha", type=float, default=0.6)
    ap.add_argument("--end_beta", type=float, default=0.35)
    ap.add_argument("--end_gamma", type=float, default=0.25)
    ap.add_argument("--terminal_reranker", choices=["joint", "heuristic"], default="joint")
    ap.add_argument("--terminal_end_weight", type=float, default=0.35)
    ap.add_argument("--terminal_path_weight", type=float, default=0.25)
    ap.add_argument("--terminal_value_weight", type=float, default=0.20)
    return ap


def main():
    args = build_parser().parse_args()
    if not os.path.isfile(args.model_ckpt):
        raise FileNotFoundError(f"Existing --model_ckpt is required: {args.model_ckpt}")
    lock_path = args.output + ".lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    lock_file = open(lock_path, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another inference process already owns {lock_path}") from exc
    lock_file.write(str(os.getpid()) + "\n")
    lock_file.flush()
    embedder, default_prompt_name = load_text_embedder(
        args.st_model,
        args.st_encoding_profile,
        cpu=args.cpu,
    )
    if args.st_prompt_name is None:
        args.st_prompt_name = default_prompt_name
    edge_model, node_model, ckpt, device = load_models(args.model_ckpt, args.cpu)
    total_input = total_output = nonempty = total_nodes = 0

    # Stream JSONL in inference-sized chunks. The training-set source is 2.4 GiB;
    # retaining all source rows plus all generated DAGs until final serialization can
    # exceed a process/container memory limit even when the host has ample free RAM.
    if args.input.endswith(".jsonl") and args.output.endswith(".jsonl"):
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        resume_rows = 0
        if args.resume and os.path.exists(args.output):
            with open(args.output, "r", encoding="utf-8") as existing:
                for line_no, line in enumerate(existing, start=1):
                    if not line.strip():
                        raise ValueError(f"Blank line in resume output at line {line_no}")
                    row = json.loads(line)
                    kv_nodes = row.get("dag", {}).get("kv_nodes", [])
                    nonempty += bool(kv_nodes); total_nodes += len(kv_nodes)
                    resume_rows += 1
            total_input = total_output = resume_rows
            print(f"[RESUME] validated_existing_rows={resume_rows} output={args.output}")
        batch: List[Dict[str, Any]] = []
        output_mode = "a" if args.resume else "w"
        with open(args.input, "r", encoding="utf-8") as src, open(args.output, output_mode, encoding="utf-8") as dst:
            for input_index, line in enumerate(src):
                if not line.strip():
                    continue
                if input_index < resume_rows:
                    continue
                if args.limit is not None and args.limit > 0 and total_input >= args.limit:
                    break
                batch.append(json.loads(line))
                total_input += 1
                if len(batch) < args.infer_batch_size:
                    continue
                rows = infer(args, batch, embedder, edge_model, node_model, ckpt, device)
                for row in rows:
                    kv_nodes = row.get("dag", {}).get("kv_nodes", [])
                    nonempty += bool(kv_nodes); total_nodes += len(kv_nodes); total_output += 1
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                dst.flush()
                batch.clear()
            if batch:
                rows = infer(args, batch, embedder, edge_model, node_model, ckpt, device)
                for row in rows:
                    kv_nodes = row.get("dag", {}).get("kv_nodes", [])
                    nonempty += bool(kv_nodes); total_nodes += len(kv_nodes); total_output += 1
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        samples = read_json_or_jsonl(args.input)
        if args.limit is not None and args.limit > 0:
            samples = samples[:args.limit]
        total_input = len(samples)
        rows = infer(args, samples, embedder, edge_model, node_model, ckpt, device)
        write_rows(args.output, rows)
        total_output = len(rows)
        nonempty = sum(bool(row.get("dag", {}).get("kv_nodes")) for row in rows)
        total_nodes = sum(len(row.get("dag", {}).get("kv_nodes", [])) for row in rows)

    print(
        f"[DONE] input={total_input} output={total_output} nonempty={nonempty} "
        f"avg_kv_nodes={total_nodes / max(1, total_output):.2f} "
        f"device={device} saved_to={args.output}"
    )


if __name__ == "__main__":
    main()
