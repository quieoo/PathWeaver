import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Callable

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# ============================================================
# 0) IO
# ============================================================
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


# ============================================================
# 1) Normalization / cheap lexical signals
# ============================================================
_SPACE_RE = re.compile(r"\s+")
_ZW_RE = re.compile(r"[\u200b\u200c\u200d\uFEFF]")
_PAREN_CONTENT_RE = re.compile(r"\([^)]*\)")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")


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

_STOPWORDS = {
    "the", "of", "a", "an", "is", "are", "was", "were",
    "in", "on", "at", "to", "for", "by", "with", "and",
    "or", "as", "that", "this"
}

def normalize_lex(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()

    tokens = [t for t in s.split() if t not in _STOPWORDS]

    return " ".join(tokens)


def _token_set(s: str):
    s = normalize_lex(s)
    return set(s.split())


def _char_ngrams(s: str, n: int):
    s = normalize_lex(s).replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def token_jaccard(a: str, b: str) -> float:
    A = _token_set(a)
    B = _token_set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def char_jaccard(a: str, b: str, n: int = 3) -> float:
    A = _char_ngrams(a, n)
    B = _char_ngrams(b, n)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def lexical_similarity(
    text1: str,
    text2: str,
    token_weight: float = 1.0,
    char_weight: float = 0.0,
    char_ngram: int = 3,
) -> float:
    """
    Combined lexical similarity.

    score =
        token_weight * token_jaccard
      + char_weight  * char_jaccard

    Args:
        text1, text2: input strings
        token_weight: weight for token-level Jaccard
        char_weight: weight for char n-gram Jaccard
        char_ngram: n for char n-gram

    Returns:
        similarity score in [0,1]
    """

    t_sim = token_jaccard(text1, text2)
    c_sim = char_jaccard(text1, text2, char_ngram)

    return token_weight * t_sim + char_weight * c_sim

def norm_match(x: Any) -> str:
    return norm_text(x).lower()


def entity_key(name: str) -> str:
    """Aggressive canonical key for entity alias merging."""
    s = norm_text(name)
    s = _PAREN_CONTENT_RE.sub("", s)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def entity_alias_keys(name: str) -> List[str]:
    """Alias keys for merging, incl. dropping last token for long names."""
    k = entity_key(name)
    if not k:
        return [k]
    toks = k.split()
    keys = [k]
    if len(toks) >= 3:
        keys.append(" ".join(toks[:-1]))
    suffixes = {"jr", "sr", "ii", "iii", "iv"}
    if toks and toks[-1] in suffixes and len(toks) >= 2:
        keys.append(" ".join(toks[:-1]))

    out, seen = [], set()
    for kk in keys:
        if kk not in seen:
            out.append(kk)
            seen.add(kk)
    return out


def _norm_lex(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def _contains_mention(q_norm: str, name: str) -> bool:
    """
    Signal A (②.1): whether entity-like name is explicitly mentioned in question.
    """
    n = _norm_lex(name)
    if not n:
        return False
    if n in q_norm:
        return True
    toks = n.split()
    if len(toks) >= 2:
        if " ".join(toks[:2]) in q_norm:
            return True
    if len(toks) == 1 and len(toks[0]) >= 5 and toks[0] in q_norm:
        return True
    return False


# ============================================================
# 2) Embedding helpers
# ============================================================
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


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a:(m,d), b:(n,d), normalized -> cosine = dot."""
    return a @ b.T


def cosine_sim_vec(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


# ============================================================
# 3) KVEdge data model
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
    rel_score: float = 0.0
    rel_val_score: float = 0.0
    rel_sim_score: float = 0.0
    rel_val_sim_score: float = 0.0

    def print(self) -> None:
        print(
            f"【{self.kid}】 {self.triple_type} <{self.triple_s}, {self.relation}, {self.triple_o}> "
            f"({self.key} [{self.src}], {self.value} [{self.dst}]) {self.score:.4f} ({self.rel_score:.4f}, {self.rel_val_score:.4f})  ({self.rel_sim_score:.4f}, {self.rel_val_sim_score:.4f})"
        )


# ============================================================
# 4) Triple iterator
# ============================================================
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


# ============================================================
# 5) Graph build: entity nodes + KVEdges
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
    """
    Decide direction for relation edges (entity-entity).
    forward: S -> O
    backward: O -> S
    """
    sN = norm_match(s)
    oN = norm_match(o)
    vN = norm_match(value)
    kN = norm_match(key)

    if vN == oN:
        return "forward", "value==O"
    if vN == sN:
        return "backward", "value==S"

    s_in = sN and (sN in kN)
    o_in = oN and (oN in kN)
    if s_in and not o_in:
        return "forward", "key_has_S"
    if o_in and not s_in:
        return "backward", "key_has_O"
    return "unknown", "ambiguous"


def collect_kv_records(
    sample: Dict[str, Any],
    supporting_only: bool,
) -> List[Tuple[str, str, str, str, str, str, str, int]]:
    kv_records: List[Tuple[str, str, str, str, str, str, str, int]] = []
    seen: Set[Tuple[str, str]] = set()
    # record: (title, ttype, rel, S, O, key, value, kv_idx)

    for title, tri in iter_triples(sample, supporting_only=supporting_only):
        ttype = norm_text(tri.get("type", ""))  # ATTRIBUTE / RELATION
        rel = norm_text(tri.get("description_type", ""))  # relation name or attribute name
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

    return kv_records


def build_kvedge_graph_from_records(
    question: str,
    kv_records: List[Tuple[str, str, str, str, str, str, str, int]],
    key_sims: np.ndarray,
    rel_sims: np.ndarray,
    rel_val_sims: np.ndarray,
    pred_weight: float = 0.0,
) -> Tuple[List[str], Dict[int, List[int]], Dict[int, KVEdge]]:
    question = norm_text(question)

    node_map: Dict[str, int] = {}
    node_names: List[str] = []
    kvedges: Dict[int, KVEdge] = {}
    out_adj: Dict[int, List[int]] = {}

    rels = [r[2] for r in kv_records]
    values = [r[4] for r in kv_records]
    rel_vals = [rel + " " + val for rel, val in zip(rels, values)]

    rel_ler = np.array([lexical_similarity(rel, question) for rel in rels], dtype=np.float32)
    rel_val_ler = np.array([lexical_similarity(rel_val, question) for rel_val in rel_vals], dtype=np.float32)

    kid = 0
    for i, (title, ttype, rel, s, o, key, value, kv_idx) in enumerate(kv_records):
        direction, _ = infer_kv_direction(s, o, key, value)
        if direction == "backward":
            src_name, dst_name = o, s
        else:
            src_name, dst_name = s, o  # forward or unknown -> forward

        src = get_or_add_node(src_name, node_map, node_names)
        dst = get_or_add_node(dst_name, node_map, node_names)

        score = float(key_sims[i]) + float(pred_weight) * float(rel_sims[i])
        rel_score = float(rel_ler[i])
        rel_val_score = float(rel_val_ler[i])
        rel_sim_score = float(rel_sims[i])
        rel_val_sim_score = float(rel_val_sims[i])

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
            rel_score=rel_score,
            rel_val_score=rel_val_score,
            rel_sim_score=rel_sim_score,
            rel_val_sim_score=rel_val_sim_score,
        )
        kvedges[kid] = e
        out_adj.setdefault(src, []).append(kid)
        kid += 1

    for u in out_adj:
        out_adj[u].sort(key=lambda eid: kvedges[eid].score, reverse=True)

    return node_names, out_adj, kvedges


def build_kvedge_graph(
    sample: Dict[str, Any],
    embedder: SentenceTransformer,
    batch_size: int,
    supporting_only: bool,
    pred_weight: float = 0.0,
    is_keep_one_edge: bool = False,
) -> Tuple[List[str], Dict[int, List[int]], Dict[int, KVEdge]]:
    """
    Return:
      node_names: entity id -> name
      out_adj: src_id -> [edge_ids]
      kvedges: eid -> KVEdge
    """
    question = norm_text(sample.get("question", ""))
    kv_records = collect_kv_records(sample, supporting_only=supporting_only)

    if not kv_records:
        return [], {}, {}

    q_emb = embed_texts(embedder, [question], batch_size=batch_size)[0:1]

    keys = [r[5] for r in kv_records]
    key_emb = embed_texts(embedder, keys, batch_size=batch_size)
    key_sims = cosine_sim_matrix(q_emb, key_emb).reshape(-1)

    rels = [r[2] for r in kv_records]
    values = [r[4] for r in kv_records]
    rel_vals = [rel + " " + val for rel, val in zip(rels, values)]
    rel_embs = embed_texts(embedder, rels, batch_size=batch_size)
    rel_sims = cosine_sim_matrix(q_emb, rel_embs).reshape(-1)
    rel_val_embs = embed_texts(embedder, rel_vals, batch_size=batch_size)
    rel_val_sims = cosine_sim_matrix(q_emb, rel_val_embs).reshape(-1)

    return build_kvedge_graph_from_records(
        question=question,
        kv_records=kv_records,
        key_sims=key_sims,
        rel_sims=rel_sims,
        rel_val_sims=rel_val_sims,
        pred_weight=pred_weight,
    )


def rebuild_out_adj(kvedges: Dict[int, KVEdge]) -> Dict[int, List[int]]:
    out_adj: Dict[int, List[int]] = defaultdict(list)
    for eid, e in kvedges.items():
        out_adj[e.src].append(eid)
    for u in out_adj:
        out_adj[u].sort(key=lambda x: kvedges[x].score, reverse=True)
    return dict(out_adj)


# ============================================================
# 6) Pre-prune (before search)
# ============================================================
def is_attribute_edge(e: KVEdge, node_names: List[str]) -> bool:
    # Best signal: triple_type
    if (e.triple_type or "").upper() == "ATTRIBUTE":
        return True

    # fallback heuristic: literal-ish dst
    dst_name = node_names[e.dst] if (0 <= e.dst < len(node_names)) else ""
    if any(ch.isdigit() for ch in dst_name):
        return True
    if len(dst_name.split()) <= 2 and dst_name and dst_name[0].islower():
        return True
    return False


def relation_signature(e: KVEdge) -> str:
    # Use relation field (description_type) directly; stable enough.
    return norm_match(e.relation)


def prune_edges_before_search(
    out_adj: Dict[int, List[int]],
    kvedges: Dict[int, KVEdge],
    node_names: List[str],
    max_attr_out_per_entity: int = 2,
    keep_best_edge_per_pair: bool = True,
) -> Tuple[Dict[int, List[int]], Dict[int, KVEdge]]:
    """
    - Keep at most K attribute edges per src
    - Dedup relation edges: keep best per (src,dst,relation_signature)
    """
    attr_by_src = defaultdict(list)
    rel_eids = []
    for eid, e in kvedges.items():
        if is_attribute_edge(e, node_names):
            attr_by_src[e.src].append(eid)
        else:
            rel_eids.append(eid)

    keep: Set[int] = set()

    # cap attribute edges per src
    for src, eids in attr_by_src.items():
        eids.sort(key=lambda x: kvedges[x].score, reverse=True)
        keep.update(eids[:max_attr_out_per_entity])

    # keep all relation edges for now
    keep.update(rel_eids)

    # dedup relation edges
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
    new_out_adj = rebuild_out_adj(new_kvedges)
    return new_out_adj, new_kvedges


# ============================================================
# 7) Keep ONE direction for ATTRIBUTE bidirectional pairs
# ============================================================
def keep_one_direction_for_attribute_pairs(
    question: str,
    kvedges: Dict[int, KVEdge],
    node_names: List[str],
    embed_text_fn: Callable[[str], np.ndarray],
    q_emb: np.ndarray,
    eps_dir: float = 0.05,
    keep_when_tie: bool = True,
) -> Set[int]:
    """
    For each ATTRIBUTE bidirectional pair (u->v and v->u), keep one edge using:
      - Signal A: mention in question
      - Signal B (②.2): sim(q,src)-sim(q,dst)
    """
    q_norm = _norm_lex(question)

    # group by unordered endpoints
    pair_map = defaultdict(list)
    for eid, e in kvedges.items():
        if is_attribute_edge(e, node_names):
            u, v = e.src, e.dst
            if u != v:
                pair_map[(min(u, v), max(u, v))].append(eid)

    # cache node embeddings
    node_emb_cache: Dict[int, np.ndarray] = {}

    def sim_to_q(node_id: int) -> float:
        if node_id not in node_emb_cache:
            txt = node_names[node_id] if 0 <= node_id < len(node_names) else ""
            node_emb_cache[node_id] = embed_text_fn(txt)
        return cosine_sim_vec(q_emb, node_emb_cache[node_id])

    keep: Set[int] = set(kvedges.keys())

    for (a, b), eids in pair_map.items():
        ab = [eid for eid in eids if kvedges[eid].src == a and kvedges[eid].dst == b]
        ba = [eid for eid in eids if kvedges[eid].src == b and kvedges[eid].dst == a]
        if not ab or not ba:
            continue

        # choose best representative per direction
        best_ab = max(ab, key=lambda eid: kvedges[eid].score)
        best_ba = max(ba, key=lambda eid: kvedges[eid].score)

        name_a = node_names[a] if 0 <= a < len(node_names) else ""
        name_b = node_names[b] if 0 <= b < len(node_names) else ""

        # Signal A
        a_in_q = _contains_mention(q_norm, name_a)
        b_in_q = _contains_mention(q_norm, name_b)

        decided = False
        keep_dir: Optional[str] = None

        if a_in_q and not b_in_q:
            keep_dir = "ab"
            decided = True
        elif b_in_q and not a_in_q:
            keep_dir = "ba"
            decided = True

        # Signal B
        if not decided:
            dir_ab = sim_to_q(a) - sim_to_q(b)
            if dir_ab > eps_dir:
                keep_dir = "ab"
                decided = True
            elif dir_ab < -eps_dir:
                keep_dir = "ba"
                decided = True
            else:
                if keep_when_tie:
                    continue
                keep_dir = "ab" if kvedges[best_ab].score >= kvedges[best_ba].score else "ba"
                decided = True

        if decided and keep_dir == "ab":
            for eid in ba:
                keep.discard(eid)
        elif decided and keep_dir == "ba":
            for eid in ab:
                keep.discard(eid)

    return keep


# ============================================================
# 8) Beam prune (two options)
# ============================================================
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

    for _ in range(max_hops):
        if len(kept_edges) >= max_kvedges or len(kept_nodes) >= max_nodes:
            break
        new_frontier: List[int] = []
        new_seen: Set[int] = set()

        for u in frontier:
            eids = out_adj.get(u, [])
            if not eids:
                continue
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


def build_in_adj_from_kvedges(kvedges: Dict[int, KVEdge]) -> Dict[int, List[int]]:
    in_adj: Dict[int, List[int]] = defaultdict(list)
    for eid, e in kvedges.items():
        in_adj[e.dst].append(eid)
    # stable ordering
    for v in in_adj:
        in_adj[v].sort(key=lambda eid: kvedges[eid].score, reverse=True)
    return dict(in_adj)


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


def prune_with_seededge_bidirectional_beam(
    kvedges: Dict[int, KVEdge],
    out_adj: Dict[int, List[int]],
    max_hops: int,
    beam_width: int,
    max_nodes: int,
    max_kvedges: int,
    num_seed_edges: int,
) -> Tuple[Set[int], Set[int], Set[int]]:
    if not kvedges:
        return set(), set(), set()

    ranked_edges = sorted(kvedges.values(), key=lambda e: e.score, reverse=True)
    seed_edges = ranked_edges[: max(1, min(num_seed_edges, len(ranked_edges)))]

    in_adj = build_in_adj_from_kvedges(kvedges)

    kept_edges: Set[int] = set()
    kept_nodes: Set[int] = set()
    kept_out: Dict[int, Set[int]] = defaultdict(set)

    def try_add_edge(eid: int) -> bool:
        if eid in kept_edges:
            return True
        e = kvedges[eid]
        if _would_create_cycle(e.src, e.dst, kept_out):
            return False
        kept_edges.add(eid)
        kept_nodes.add(e.src)
        kept_nodes.add(e.dst)
        kept_out[e.src].add(e.dst)
        return True

    seeds: Set[int] = set()
    for e in seed_edges:
        if len(kept_edges) >= max_kvedges:
            break
        if try_add_edge(e.kid):
            seeds.add(e.src)
            seeds.add(e.dst)

    def expand_backward(start_node: int):
        beam = [(0.0, start_node)]
        for _ in range(max_hops):
            if len(kept_edges) >= max_kvedges or len(kept_nodes) >= max_nodes:
                break
            cand: List[Tuple[float, int, int]] = []
            for cum, cur in beam:
                for eid in in_adj.get(cur, []):
                    e = kvedges[eid]
                    cand.append((cum + e.score, e.src, eid))
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
            cand: List[Tuple[float, int, int]] = []
            for cum, cur in beam:
                for eid in out_adj.get(cur, []):
                    e = kvedges[eid]
                    cand.append((cum + e.score, e.dst, eid))
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
        expand_backward(e.src)
        if len(kept_edges) >= max_kvedges or len(kept_nodes) >= max_nodes:
            break
        expand_forward(e.dst)

    return seeds, kept_nodes, kept_edges


# ============================================================
# 9) DAG enforcement (break cycles)
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


# ============================================================
# 10) Sink constraint
# ============================================================
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


# ============================================================
# 11) Export: KV nodes + adjacency
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
    for eid_i in kept_list:
        i = idx_map[eid_i]
        dst_i = kvedges[eid_i].dst
        for eid_j in kept_list:
            j = idx_map[eid_j]
            if i != j and kvedges[eid_j].src == dst_i:
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

    non_seed_cands = [n for n in cand_nodes if n not in seeds]
    if non_seed_cands:
        cand_nodes = non_seed_cands

    def cand_key(n: int) -> Tuple[float, float, int, float]:
        ans_best = max(kvedges[eid].score for eid in answer_in_map[n])
        in_sum = sum(max(0.0, kvedges[eid].score) for eid in in_map.get(n, []))
        out_sum = sum(max(0.0, kvedges[eid].score) for eid in out_map.get(n, []))
        out_cnt = len(out_map.get(n, []))
        return (ans_best, in_sum, -out_cnt, -out_sum)

    target = max(cand_nodes, key=cand_key)

    for eid in list(out_map.get(target, [])):
        kept_kvedges.discard(eid)

    kept_nodes, kept_kvedges = _prune_to_reachable_from_seeds(
        seeds=seeds,
        kept_nodes=kept_nodes,
        kept_kvedges=kept_kvedges,
        kvedges=kvedges,
    )
    return kept_nodes, kept_kvedges


def _lex_tokens(s: str) -> set:
    s = norm_text(s).lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return set(s.split()) if s else set()

def _jaccard(a: str, b: str) -> float:
    A = _lex_tokens(a)
    B = _lex_tokens(b)
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A | B))

def cleanup_reachable_from_seeds(
    seeds: Set[int],
    kept_nodes: Set[int],
    kept_kvedges: Set[int],
    kvedges: Dict[int, "KVEdge"],
) -> Tuple[Set[int], Set[int]]:
    out_adj: Dict[int, List[int]] = defaultdict(list)
    for eid in kept_kvedges:
        e = kvedges[eid]
        if e.src in kept_nodes and e.dst in kept_nodes:
            out_adj[e.src].append(eid)

    vis: Set[int] = set()
    stack = list(seeds)
    while stack:
        u = stack.pop()
        if u in vis:
            continue
        if u not in kept_nodes:
            continue
        vis.add(u)
        for eid in out_adj.get(u, []):
            v = kvedges[eid].dst
            if v not in vis:
                stack.append(v)

    vis_edges = {eid for eid in kept_kvedges if kvedges[eid].src in vis and kvedges[eid].dst in vis}
    return vis, vis_edges


def iter_batches(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    step = max(1, batch_size)
    for i in range(0, len(items), step):
        yield items[i:i + step]



# ============================================================
# 12) Main pipeline: create_dag
# ============================================================
def create_dag(args, samples: List[Dict[str, Any]], embedder: SentenceTransformer) -> List[Dict[str, Any]]:
    if args.limit is not None and args.limit > 0:
        samples = samples[:args.limit]

    if args.verbose:
        # samples = [samples[5]]
        samples = samples[:10]

    out_samples: List[Dict[str, Any]] = []
    answer_recall = 0
    graph_recall = 0
    none_sink_recall = 0

    sample_batch_size = max(1, args.batch_size)

    total_batches = (len(samples) + sample_batch_size - 1) // sample_batch_size
    for sample_batch in tqdm(iter_batches(samples, sample_batch_size), total=total_batches, desc="Create DAG"):
        batch_infos: List[Dict[str, Any]] = []
        batch_questions: List[str] = []
        batch_keys: List[str] = []
        batch_rels: List[str] = []
        batch_rel_vals: List[str] = []

        for sample in sample_batch:
            question = norm_text(sample.get("question", ""))
            kv_records = collect_kv_records(sample, supporting_only=args.supporting_only)
            info = {
                "sample": sample,
                "question": question,
                "answer": norm_text(sample.get("answer", "")),
                "kv_records": kv_records,
                "key_offset": len(batch_keys),
                "num_records": len(kv_records),
            }
            batch_infos.append(info)
            batch_questions.append(question)

            if kv_records:
                rels = [r[2] for r in kv_records]
                values = [r[4] for r in kv_records]
                batch_keys.extend(r[5] for r in kv_records)
                batch_rels.extend(rels)
                batch_rel_vals.extend(rel + " " + val for rel, val in zip(rels, values))

        q_emb_batch = embed_texts(embedder, batch_questions, batch_size=args.batch_size)
        key_emb_batch = embed_texts(embedder, batch_keys, batch_size=args.batch_size) if batch_keys else None
        rel_emb_batch = embed_texts(embedder, batch_rels, batch_size=args.batch_size) if batch_rels else None
        rel_val_emb_batch = embed_texts(embedder, batch_rel_vals, batch_size=args.batch_size) if batch_rel_vals else None

        for batch_idx, info in enumerate(batch_infos):
            sample = info["sample"]
            question = info["question"]
            answer = info["answer"]
            kv_records = info["kv_records"]

            # sample, stats = filter_attribute_triples_by_relation_relevance(
            #     sample=sample,
            #     embedder=embedder,
            #     batch_size=args.batch_size,
            #     supporting_only=args.supporting_only,
            #     use_key_string=True,
            # )

            # ---------- A) build full KVEdge graph ----------
            if not kv_records:
                node_names, out_adj, kvedges = [], {}, {}
            else:
                start = info["key_offset"]
                end = start + info["num_records"]
                q_emb = q_emb_batch[batch_idx:batch_idx + 1]
                key_sims = cosine_sim_matrix(q_emb, key_emb_batch[start:end]).reshape(-1)
                rel_sims = cosine_sim_matrix(q_emb, rel_emb_batch[start:end]).reshape(-1)
                rel_val_sims = cosine_sim_matrix(q_emb, rel_val_emb_batch[start:end]).reshape(-1)
                node_names, out_adj, kvedges = build_kvedge_graph_from_records(
                    question=question,
                    kv_records=kv_records,
                    key_sims=key_sims,
                    rel_sims=rel_sims,
                    rel_val_sims=rel_val_sims,
                    pred_weight=args.pred_weight,
                )

            # ---------- B) pre-prune before search ----------
            if args.max_attr_out_per_entity is not None and args.max_attr_out_per_entity > 0:
                out_adj, kvedges = prune_edges_before_search(
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

            # ---------- D) quick graph-level recall (before pruning) ----------
            for e in kvedges.values():
                if answer and (answer in e.value or e.value in answer):
                    graph_recall += 1
                    break

            if args.verbose:
                print(f"Q: {question}")
                print(f"A: {answer}")
                for e in sorted(kvedges.values(), key=lambda x: x.score, reverse=True)[:50]:
                    e.print()
                print("=" * 60)

            # ---------- E) beam prune ----------
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

            # ---------- F) break cycles -> DAG ----------
            kept_kvedges = break_cycles_to_dag_kv(kept_nodes, kept_kvedges, kvedges)

            # ---------- G) enforce sinks ----------
            if args.max_sinks is not None and args.max_sinks > 0:
                kept_nodes, kept_kvedges = enforce_max_sinks_entity(
                    seeds=seeds,
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
                sample["dag"] = {"kv_nodes": [], "adj": [], "meta": {"reason": "empty_after_sink_prune"}}
                out_samples.append(sample)
                continue

            if args.verbose:
                print_current_graph_kv(seeds, kept_nodes, kept_kvedges, kvedges)

            # ---------- H) answer recall on sinks ----------
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

            # answer recall on path
            for eid in kept_kvedges:
                e = kvedges[eid]
                vN = norm_match(e.value)
                if vN and (vN == ansN or ansN in vN or vN in ansN):
                    none_sink_recall += 1
                    break

            # ---------- I) export ----------
            kv_nodes, adj = export_kv_nodes_and_adj(kept_kvedges, kvedges, keep_score=args.keep_score)

            # goal labels (optional supervision)
            goal_ids: List[int] = []
            if ansN:
                for i, kv in enumerate(kv_nodes):
                    vN = norm_match(kv.get("value", ""))
                    if vN and (vN == ansN or ansN in vN):
                        goal_ids.append(i)

            meta = {
                "num_entity_nodes": int(len(kept_nodes)),
                "num_kv_edges": int(len(kept_kvedges)),
                "num_kv_nodes": int(len(kv_nodes)),
                "goal_ids": goal_ids,
            }

            sample["dag"] = {"kv_nodes": kv_nodes, "adj": adj, "meta": meta}
            out_samples.append(sample)


    if len(samples) > 0:
        print(f"Answer recall: {answer_recall / len(samples):.4f}")
        print(f"Graph  recall: {graph_recall  / len(samples):.4f}")
        print(f"None-sink recall: {none_sink_recall / len(samples):.4f}")

    return out_samples


# ============================================================
# 13) CLI
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

    # prune knobs
    ap.add_argument("--seed_top_m", type=int, default=30, help="top-M KVEdges to derive seed nodes/seed edges")
    ap.add_argument("--max_hops", type=int, default=3, help="beam expansion max hops")
    ap.add_argument("--beam_width", type=int, default=3, help="beam width per node per hop")
    ap.add_argument("--max_nodes", type=int, default=30, help="max entity nodes in pruned subgraph")
    ap.add_argument("--max_edges", type=int, default=40, help="max KVEdges in pruned subgraph")
    ap.add_argument("--max_sinks", type=int, default=None, help="max sinks (out-degree=0) on induced entity DAG")
    ap.add_argument("--use_seededge_beam", action="store_true", help="use seed-edge bidirectional beam pruning")
    ap.add_argument("--pred_weight", type=float, default=0.0, help="weight for relation score sim(question, relation)")

    # new: attribute keep-one-direction
    ap.add_argument("--keep_one_attr_direction", action="store_true", help="keep only one direction for ATTRIBUTE pairs")
    ap.add_argument("--eps_dir", type=float, default=0.05, help="direction confidence threshold")
    ap.add_argument("--max_attr_out_per_entity", type=int, default=None, help="cap attribute out edges per entity")
    ap.add_argument(
        "--answer_terminalization",
        action="store_true",
        help="force one matched answer node to become sink by dropping its outgoing edges",
    )



    args = ap.parse_args()

    # 打印参数
    print(args)

    embedder = SentenceTransformer(args.st_model)

    samples = read_json_or_jsonl(args.input)
    print(f"Load {len(samples)} samples from {args.input}")

    out = create_dag(args, samples, embedder)
    if not out:
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
