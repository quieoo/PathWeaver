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
    q_emb = embed_texts(embedder, [question], batch_size=batch_size)[0:1]

    node_map: Dict[str, int] = {}
    node_names: List[str] = []
    kvedges: Dict[int, KVEdge] = {}
    out_adj: Dict[int, List[int]] = {}

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

    if not kv_records:
        return node_names, out_adj, kvedges

    keys = [r[5] for r in kv_records]
    key_emb = embed_texts(embedder, keys, batch_size=batch_size)
    key_sims = cosine_sim_matrix(q_emb, key_emb).reshape(-1)


    rels = [r[2] for r in kv_records]
    values = [r[4] for r in kv_records]
    rel_vals= [rel + " " + val for rel, val in zip(rels, values)]

    rel_ler = [lexical_similarity(rel, question) for rel in rels]
    val_ler = [lexical_similarity(val, question) for val in values]
    rel_val_ler = [lexical_similarity(rel_val, question) for rel_val in rel_vals]
    
    rel_embs = embed_texts(embedder, rels, batch_size=batch_size)
    rel_sims = cosine_sim_matrix(q_emb, rel_embs).reshape(-1)
    rel_val_embs = embed_texts(embedder, rel_vals, batch_size=batch_size)
    rel_val_sims = cosine_sim_matrix(q_emb, rel_val_embs).reshape(-1)


    kid = 0
    for i, (title, ttype, rel, s, o, key, value, kv_idx) in enumerate(kv_records):
        direction, _ = infer_kv_direction(s, o, key, value)
        if direction == "backward":
            src_name, dst_name = o, s
        else:
            src_name, dst_name = s, o  # forward or unknown -> forward

        # 不需要手动判方向 --------> 
        # is_backward = (kv_idx % 2 == 1)
        # if is_backward:
        #     src_name, dst_name = o, s
        # else:
        #     src_name, dst_name = s, o



        src = get_or_add_node(src_name, node_map, node_names)
        dst = get_or_add_node(dst_name, node_map, node_names)

        score = float(key_sims[i]) + float(pred_weight) * float(rel_sims[i])
        rel_score = float(rel_ler[i])
        rel_val_score = float(rel_val_ler[i])
        rel_sim_score = float(rel_sims[i])
        rel_val_sim_score = float(rel_val_sims[i])

        # if is_keep_one_edge and ttype == "ATTRIBUTE" and rel_val_score > rel_score and direction == "forward":
        #     continue
            
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

    # deterministic ordering by score
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

def filter_attribute_triples_by_relation_relevance(
    sample: Dict[str, Any],
    embedder,
    batch_size: int,
    supporting_only: bool = False,
    sim_th_rel: float = 0.3,      # sim(question, relation/description_type) 阈值
    sim_th_key: float = 0.4,      # sim(question, key_string) 阈值（更强）
    lex_th: float = 0.08,          # jaccard(question, relation/key) 阈值
    use_key_string: bool = True,   # 是否同时看 key_string（推荐 True）
    keep_if_empty_relation: bool = True,  # relation缺失时保守保留
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    在 build_kvedge_graph 之前调用：删除与问题无关的 ATTRIBUTE triples。

    保留条件（满足任一即可）：
      1) lexical overlap 足够大：Jaccard(question, relation/key) >= lex_th
      2) embedding 足够大：sim(question, relation) >= sim_th_rel 或 sim(question, key) >= sim_th_key

    Returns:
      new_sample: 深拷贝后（结构上）过滤了 triple_list（仅删 ATTRIBUTE）
      stats: 统计信息
    """
    question = norm_text(sample.get("question", ""))
    q_emb = embed_texts(embedder, [question], batch_size=batch_size)[0:1]  # (1,d)

    # supporting_only 时，只过滤 supporting titles 的 triples（和你 iter_triples 一致）
    supporting_titles = None
    if supporting_only:
        sf = sample.get("supporting_facts", []) or []
        supporting_titles = {t for t, _ in sf if isinstance(t, str)}

    # 为了不在原地乱改，构造浅拷贝结构（context/para/triple_list 都新建）
    new_sample = dict(sample)
    new_context: List[Dict[str, Any]] = []

    # 先收集所有 ATTRIBUTE triple 的 (relation_text, key_text) 以便批量 embedding
    # 同时保留索引映射，后续写回过滤结果
    attr_items = []  # (para_idx, tri_idx, relation_text, key_text_or_empty)
    para_kept_masks = []  # 每段落每个tri是否保留（先全True，后续对属性置False）
    for p_idx, para in enumerate(sample.get("context", []) or []):
        title = norm_text(para.get("title", ""))
        triple_list = para.get("triple_list", []) or []

        # 如果 supporting_only 且该 title 不在 supporting_titles，就不动它（保持原样）
        if supporting_titles is not None and title not in supporting_titles:
            para_kept_masks.append([True] * len(triple_list))
            continue

        kept_mask = [True] * len(triple_list)
        for t_idx, tri in enumerate(triple_list):
            ttype = norm_text(tri.get("type", "")).upper()
            if ttype != "ATTRIBUTE":
                continue

            rel = norm_text(tri.get("description_type", ""))  # 你的 relation（属性名）
            if not rel and keep_if_empty_relation:
                continue  # 保守：relation缺失直接保留

            key_txt = ""
            if use_key_string:
                # tri 中可能有 key_string；或者 tri["kv_lists"][0]["key_string"] 也行
                key_txt = norm_text(tri.get("key_string", ""))
                if not key_txt:
                    kvs = tri.get("kv_lists", []) or []
                    if kvs:
                        key_txt = norm_text(kvs[0].get("key_string", ""))

            attr_items.append((p_idx, t_idx, rel, key_txt))
        para_kept_masks.append(kept_mask)

    # 没有属性就直接返回
    if not attr_items:
        # rebuild new_context 但不改 triple_list
        for para in sample.get("context", []) or []:
            new_context.append(dict(para))
        new_sample["context"] = new_context
        return new_sample, {"attr_total": 0, "attr_removed": 0, "attr_kept": 0}

    # 批量 embed：relation / key
    rel_texts = [it[2] for it in attr_items]
    rel_emb = embed_texts(embedder, rel_texts, batch_size=batch_size)  # (N,d)
    rel_sims = cosine_sim_matrix(q_emb, rel_emb).reshape(-1)           # (N,)

    if use_key_string:
        key_texts = [it[3] if it[3] else it[2] for it in attr_items]   # key缺失就退化用rel
        key_emb = embed_texts(embedder, key_texts, batch_size=batch_size)
        key_sims = cosine_sim_matrix(q_emb, key_emb).reshape(-1)
    else:
        key_texts = [""] * len(attr_items)
        key_sims = np.zeros((len(attr_items),), dtype=np.float32)

    # 决策：不相关则删
    removed = 0
    q_lex = question
    for i, (p_idx, t_idx, rel, key_txt) in enumerate(attr_items):
        # lexical
        j_rel = _jaccard(q_lex, rel) if rel else 0.0
        j_key = _jaccard(q_lex, key_txt) if key_txt else 0.0
        lex_ok = (j_rel >= lex_th) or (j_key >= lex_th)

        # embedding
        emb_ok = (float(rel_sims[i]) >= sim_th_rel) or (float(key_sims[i]) >= sim_th_key)

        if not (lex_ok or emb_ok):
            # 删除该 ATTRIBUTE triple
            para_kept_masks[p_idx][t_idx] = False
            removed += 1

    # 写回 new_context
    for p_idx, para in enumerate(sample.get("context", []) or []):
        new_para = dict(para)
        triple_list = para.get("triple_list", []) or []
        mask = para_kept_masks[p_idx]

        if len(mask) == len(triple_list):
            new_para["triple_list"] = [tri for tri, keep in zip(triple_list, mask) if keep]
        else:
            # 极端情况下长度不一致：保守不删
            new_para["triple_list"] = list(triple_list)

        new_context.append(new_para)

    new_sample["context"] = new_context
    total = len(attr_items)
    return new_sample, {
        "attr_total": int(total),
        "attr_removed": int(removed),
        "attr_kept": int(total - removed),
    }

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


def prune_sinks_qrel_and_promote_no_rebuild(
    question: str,
    embedder: SentenceTransformer,
    batch_size: int,
    seeds: Set[int],
    kept_nodes: Set[int],
    kept_kvedges: Set[int],
    kvedges: Dict[int, "KVEdge"],
    node_names: List[str],
    sink_topk: int = 16,
    sink_rel_th: float = -1.0,        # <0 表示不用阈值，只用 topk
    w_node: float = 0.35,
    w_in_key: float = 0.20,
    w_in_val: float = 0.45,
    mention_bonus: float = 0.10,
    promote_margin: float = 0.03,     # 更容易触发
    promote_drop_th: float = 0.18,    # 更敢剪
) -> Tuple[Set[int], Set[int]]:
    if not kept_kvedges or not kept_nodes:
        return kept_nodes, kept_kvedges

    q = norm_text(question)
    q_norm = _norm_lex(q)
    q_emb = embed_texts(embedder, [q], batch_size=batch_size)[0]

    # build in/out adjacency on entity nodes (only within kept set)
    out_eids = defaultdict(list)
    in_eids = defaultdict(list)
    outdeg = defaultdict(int)

    for eid in kept_kvedges:
        e = kvedges[eid]
        if e.src in kept_nodes and e.dst in kept_nodes:
            out_eids[e.src].append(eid)
            in_eids[e.dst].append(eid)
            outdeg[e.src] += 1

    sinks = [n for n in kept_nodes if outdeg.get(n, 0) == 0]
    if not sinks:
        return kept_nodes, kept_kvedges

    # cosine sim(q, text)
    txt_cache: Dict[str, np.ndarray] = {}

    def sim_text(t: str) -> float:
        t = norm_text(t)
        if not t:
            return 0.0
        if t not in txt_cache:
            txt_cache[t] = embed_texts(embedder, [t], batch_size=batch_size)[0]
        return cosine_sim_vec(q_emb, txt_cache[t])

    # 1) score sinks by question relevance
    sink_scored = []
    for s in sinks:
        name = node_names[s] if 0 <= s < len(node_names) else ""
        s_node = sim_text(name)

        best_in_k = 0.0
        best_in_v = 0.0
        for eid in in_eids.get(s, []):
            e = kvedges[eid]
            best_in_k = max(best_in_k, sim_text(e.key))
            best_in_v = max(best_in_v, sim_text(e.value))

        bonus = mention_bonus if _contains_mention(q_norm, name) else 0.0
        score = w_node * s_node + w_in_key * best_in_k + w_in_val * best_in_v + bonus
        sink_scored.append((score, s))

    sink_scored.sort(key=lambda x: x[0], reverse=True)

    if sink_rel_th >= 0.0:
        sink_scored = [x for x in sink_scored if x[0] >= sink_rel_th]

    if sink_topk is not None and sink_topk > 0 and len(sink_scored) > sink_topk:
        sink_scored = sink_scored[:sink_topk]

    kept_sinks = {s for _, s in sink_scored}
    if not kept_sinks:
        kept_sinks = {sink_scored[0][1]} if sink_scored else set([sinks[0]])

    removed_sinks = set(sinks) - kept_sinks

    # 2) remove incoming edges to removed sinks (DO NOT rebuild subgraph!)
    new_nodes = set(kept_nodes)
    new_edges = set(kept_kvedges)

    for s in removed_sinks:
        for eid in in_eids.get(s, []):
            if eid in new_edges:
                new_edges.remove(eid)

    # cleanup unreachable (keep graph tight but avoid killing none-sink recall)
    new_nodes, new_edges = cleanup_reachable_from_seeds(seeds, new_nodes, new_edges, kvedges)
    if not new_edges:
        return new_nodes, new_edges

    # rebuild adjacency after sink incoming-edge removals
    out_eids = defaultdict(list)
    in_eids = defaultdict(list)
    for eid in new_edges:
        e = kvedges[eid]
        out_eids[e.src].append(eid)
        in_eids[e.dst].append(eid)

    # 3) sink promotion: cut low-relevance outgoing edges for “terminal-like” nodes
    best_in_val = defaultdict(float)
    best_out_rel = defaultdict(float)

    for v in new_nodes:
        for eid in in_eids.get(v, []):
            best_in_val[v] = max(best_in_val[v], sim_text(kvedges[eid].value))
        for eid in out_eids.get(v, []):
            e = kvedges[eid]
            # IMPORTANT: out relevance uses max(key,value) instead of only key
            r = max(sim_text(e.key), sim_text(e.value))
            best_out_rel[v] = max(best_out_rel[v], r)

    promoted_cut = 0
    for u in list(new_nodes):
        if u in kept_sinks:
            continue
        if not out_eids.get(u):
            continue

        if best_in_val[u] - best_out_rel[u] >= promote_margin:
            to_drop = []
            for eid in out_eids[u]:
                e = kvedges[eid]
                out_r = max(sim_text(e.key), sim_text(e.value))
                if out_r < promote_drop_th:
                    to_drop.append(eid)
            for eid in to_drop:
                if eid in new_edges:
                    new_edges.remove(eid)
                    promoted_cut += 1

    if promoted_cut > 0:
        new_nodes, new_edges = cleanup_reachable_from_seeds(seeds, new_nodes, new_edges, kvedges)

    return new_nodes, new_edges

def promote_answer_like_nodes_to_sinks(
    question: str,
    embedder: SentenceTransformer,
    batch_size: int,
    seeds: Set[int],
    kept_nodes: Set[int],
    kept_kvedges: Set[int],
    kvedges: Dict[int, "KVEdge"],
    node_names: List[str],
    promote_topk_nodes: int = 24,     # 选多少个“最像答案终点”的节点来做 sinkify
    promote_margin: float = 0.02,     # best_in_val - best_out_rel >= margin 才允许剪
    promote_drop_th: float = 0.18,    # out_rel < th 的出边会被剪掉
    protect_high_out: float = 0.45,   # out_rel >= 该值的出边永远不剪（避免剪掉强相关证据链）
) -> Tuple[Set[int], Set[int]]:
    """
    目标：不降低 none-sink recall（尽量不删入边/不删节点），
         通过“剪掉答案节点的低相关出边”把答案节点促成 sink，
         提升 answer recall（因为 answer recall 只看 sink 的入边 value）。

    注意：本函数不做 sink filtering，不删除任何 sink 的入边。
    """

    if not kept_kvedges or not kept_nodes:
        return kept_nodes, kept_kvedges

    q = norm_text(question)
    q_emb = embed_texts(embedder, [q], batch_size=batch_size)[0]

    # Build adjacency
    out_eids = defaultdict(list)
    in_eids = defaultdict(list)
    for eid in kept_kvedges:
        e = kvedges[eid]
        if e.src in kept_nodes and e.dst in kept_nodes:
            out_eids[e.src].append(eid)
            in_eids[e.dst].append(eid)

    # embedding cache + cosine sim
    txt_cache: Dict[str, np.ndarray] = {}

    def sim_text(t: str) -> float:
        t = norm_text(t)
        if not t:
            return 0.0
        if t not in txt_cache:
            txt_cache[t] = embed_texts(embedder, [t], batch_size=batch_size)[0]
        return cosine_sim_vec(q_emb, txt_cache[t])

    # Compute per-node:
    # best_in_val: 该节点作为“终点”时，入边 value 与问题相关度的最大值（更像答案/证据）
    # best_out_rel: 该节点继续往外扩展时，出边(key/value)与问题相关度的最大值（越大越不该剪）
    best_in_val = defaultdict(float)
    best_out_rel = defaultdict(float)

    for v in kept_nodes:
        for eid in in_eids.get(v, []):
            best_in_val[v] = max(best_in_val[v], sim_text(kvedges[eid].value))
        for eid in out_eids.get(v, []):
            e = kvedges[eid]
            r = max(sim_text(e.key), sim_text(e.value))
            best_out_rel[v] = max(best_out_rel[v], r)

    # Rank nodes by "answer-likeness":
    # 直觉：best_in_val 高且 best_out_rel 低 -> 更像应该成为终点
    node_scored = []
    for v in kept_nodes:
        if v in seeds:
            continue
        if not in_eids.get(v):
            continue
        score = best_in_val[v] - 0.7 * best_out_rel[v]
        node_scored.append((score, v))
    node_scored.sort(key=lambda x: x[0], reverse=True)

    if promote_topk_nodes is not None and promote_topk_nodes > 0:
        node_scored = node_scored[:promote_topk_nodes]

    new_edges = set(kept_kvedges)

    promoted_cut = 0
    promoted_nodes: Set[int] = set()

    for _, u in node_scored:
        if not out_eids.get(u):
            continue  # already sink
        if (best_in_val[u] - best_out_rel[u]) < promote_margin:
            continue

        # Cut low-relevance outgoing edges, but protect very relevant ones
        to_drop = []
        for eid in out_eids[u]:
            e = kvedges[eid]
            out_r = max(sim_text(e.key), sim_text(e.value))
            if out_r >= protect_high_out:
                continue
            if out_r < promote_drop_th:
                to_drop.append(eid)

        if to_drop:
            promoted_nodes.add(u)
        for eid in to_drop:
            if eid in new_edges:
                new_edges.remove(eid)
                promoted_cut += 1

    # 关键：不要做 cleanup_reachable_from_seeds（它会把答案所在的“非主链”可达结构扫掉，导致 none-sink 掉）
    # 如果你担心输出里出现孤点，可以仅在导出时过滤掉完全不相连的点，但不要在 recall 统计前删。

    return set(kept_nodes), new_edges, promoted_nodes

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


    for sample in tqdm(samples, desc="Create DAG"):
        question = norm_text(sample.get("question", ""))
        answer = norm_text(sample.get("answer", ""))

        # sample, stats = filter_attribute_triples_by_relation_relevance(
        #     sample=sample,
        #     embedder=embedder,
        #     batch_size=args.batch_size,
        #     supporting_only=args.supporting_only,
        #     use_key_string=True,
        # )

        # ---------- A) build full KVEdge graph ----------
        node_names, out_adj, kvedges = build_kvedge_graph(
            sample=sample,
            embedder=embedder,
            batch_size=args.batch_size,
            supporting_only=args.supporting_only,
            pred_weight=args.pred_weight,
            is_keep_one_edge=args.keep_one_attr_direction,
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

        # ---------- C) drop one direction for ATTRIBUTE bidirectional pairs ----------
        # if args.keep_one_attr_direction:
        #     q_emb = embed_texts(embedder, [question], batch_size=args.batch_size)[0]
        #     embed_text_fn = lambda t: embed_texts(embedder, [t], batch_size=args.batch_size)[0]

        #     kept_eids = keep_one_direction_for_attribute_pairs(
        #         question=question,
        #         kvedges=kvedges,
        #         node_names=node_names,
        #         embed_text_fn=embed_text_fn,
        #         q_emb=q_emb,
        #         eps_dir=args.eps_dir,
        #         keep_when_tie=True,
        #     )
        #     kvedges = {eid: e for eid, e in kvedges.items() if eid in kept_eids}
        #     out_adj = rebuild_out_adj(kvedges)

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