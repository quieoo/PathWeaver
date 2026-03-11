import argparse
import json
import os
import re
import math
import heapq
from collections import defaultdict, Counter
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
_SPACE_RE = re.compile(r'\s+')
_ZW_RE = re.compile(r'[\u200b\u200c\u200d\uFEFF]')
_PAREN_CONTENT_RE = re.compile(r'\([^)]*\)')
_PUNCT_RE = re.compile(r'[^a-z0-9\s]+')

STOPWORDS = {
    'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'by', 'with', 'from', 'and', 'or',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'as', 'that', 'this', 'these', 'those',
    'which', 'what', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how', 'many', 'much',
    'do', 'does', 'did', 'done', 'have', 'has', 'had', 'into', 'after', 'before', 'about',
    'during', 'through', 'over', 'under', 'between', 'among', 'than', 'then', 'it', 'its',
    'their', 'his', 'her', 'them', 'they', 'he', 'she', 'you', 'your', 'i', 'we', 'our',
    'me', 'my', 'mine', 'us', 'also', 'other', 'another', 'same', 'such', 'both', 'all',
    'any', 'some', 'more', 'most', 'less', 'least', 'each', 'per'
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


def _norm_lex(s: str) -> str:
    s = (s or '').lower()
    s = _PUNCT_RE.sub(' ', s)
    s = _SPACE_RE.sub(' ', s).strip()
    return s


def lexical_tokens(s: str) -> List[str]:
    s = _norm_lex(s)
    return [t for t in s.split() if t and t not in STOPWORDS]


def lexical_set(s: str) -> Set[str]:
    return set(lexical_tokens(s))


def jaccard_text(a: str, b: str) -> float:
    A = lexical_set(a)
    B = lexical_set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A | B))


def overlap_count(a: str, b: str) -> int:
    A = lexical_set(a)
    B = lexical_set(b)
    return len(A & B)


def contains_phrase(question_norm: str, phrase: str) -> bool:
    p = _norm_lex(phrase)
    if not p:
        return False
    return p in question_norm


# ============================================================
# 2) Embedding helpers
# ============================================================
def embed_texts(embedder: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
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
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


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

    def print(self) -> None:
        print(
            f'【{self.kid}】 <{self.triple_s}, {self.relation}, {self.triple_o}> '
            f'({self.key} [{self.src}], {self.value} [{self.dst}]) {self.score:.4f}'
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
    pred_weight: float = 0.25,
    title_weight: float = 0.05,
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
    rels = [r[2] for r in kv_records]
    titles = [r[0] for r in kv_records]

    key_emb = embed_texts(embedder, keys, batch_size=batch_size)
    key_sims = cosine_sim_matrix(q_emb, key_emb).reshape(-1)
    rel_emb = embed_texts(embedder, rels, batch_size=batch_size)
    rel_sims = cosine_sim_matrix(q_emb, rel_emb).reshape(-1)
    title_emb = embed_texts(embedder, titles, batch_size=batch_size)
    title_sims = cosine_sim_matrix(q_emb, title_emb).reshape(-1)

    kid = 0
    for i, (title, ttype, rel, s, o, key, value, kv_idx) in enumerate(kv_records):
        direction, _ = infer_kv_direction(s, o, key, value)
        if direction == 'backward':
            src_name, dst_name = o, s
        else:
            src_name, dst_name = s, o

        src = get_or_add_node(src_name, node_map, node_names)
        dst = get_or_add_node(dst_name, node_map, node_names)

        score = float(key_sims[i]) + float(pred_weight) * float(rel_sims[i]) + float(title_weight) * float(title_sims[i])

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
# 6) Utilities for UNIQORN-style context graph / GST
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


def build_node_embeddings(node_names: List[str], embedder: SentenceTransformer, batch_size: int) -> np.ndarray:
    if not node_names:
        return np.zeros((0, 1), dtype=np.float32)
    return embed_texts(embedder, node_names, batch_size=batch_size)


def build_relation_inventory(kvedges: Dict[int, KVEdge]) -> Tuple[List[str], Dict[str, List[int]]]:
    rel_to_eids: Dict[str, List[int]] = defaultdict(list)
    for eid, e in kvedges.items():
        rel = norm_text(e.relation)
        if rel:
            rel_to_eids[rel].append(eid)
    rel_texts = list(rel_to_eids.keys())
    return rel_texts, rel_to_eids


def detect_anchor_groups(
    question: str,
    node_names: List[str],
    node_emb: np.ndarray,
    kvedges: Dict[int, KVEdge],
    embedder: SentenceTransformer,
    batch_size: int,
    max_groups: int = 6,
    max_nodes_per_group: int = 8,
    entity_sim_th: float = 0.32,
    rel_sim_th: float = 0.30,
    min_token_overlap: int = 1,
) -> Tuple[List[Set[int]], Dict[int, str], Set[int]]:
    """
    Approximate UNIQORN anchor extraction.
    Each group corresponds to one question cue and contains candidate anchor nodes.
    """
    q_norm = _norm_lex(question)
    q_tokens = lexical_tokens(question)
    q_emb = embed_texts(embedder, [question], batch_size=batch_size)[0]

    groups: List[Tuple[float, str, Set[int]]] = []
    anchor_reason: Dict[int, str] = {}
    hard_mentioned_nodes: Set[int] = set()

    # 1) entity/node anchors
    if len(node_names) > 0:
        sims = (node_emb @ q_emb.reshape(-1, 1)).reshape(-1)
        node_candidates = []
        for nid, name in enumerate(node_names):
            if not name:
                continue
            ov = overlap_count(question, name)
            phrase = contains_phrase(q_norm, name)
            sim = float(sims[nid])
            score = sim + 0.18 * ov + (0.20 if phrase else 0.0)
            if phrase or ov >= min_token_overlap or sim >= entity_sim_th:
                node_candidates.append((score, nid, name, phrase, ov, sim))
                if phrase or ov >= max(1, min_token_overlap):
                    hard_mentioned_nodes.add(nid)

        node_candidates.sort(key=lambda x: x[0], reverse=True)
        used = set()
        for score, nid, name, phrase, ov, sim in node_candidates[: max_groups * max_nodes_per_group]:
            key = entity_key(name)
            if not key or key in used:
                continue
            used.add(key)
            groups.append((score, f'entity::{name}', {nid}))
            anchor_reason[nid] = f'entity:{name}'

    # 2) relation anchors -> convert to endpoint groups
    rel_texts, rel_to_eids = build_relation_inventory(kvedges)
    if rel_texts:
        rel_emb = embed_texts(embedder, rel_texts, batch_size=batch_size)
        rel_sims = cosine_sim_matrix(q_emb.reshape(1, -1), rel_emb).reshape(-1)
        rel_candidates = []
        for i, rel in enumerate(rel_texts):
            ov = overlap_count(question, rel)
            phrase = contains_phrase(q_norm, rel)
            sim = float(rel_sims[i])
            score = sim + 0.18 * ov + (0.20 if phrase else 0.0)
            if phrase or ov >= min_token_overlap or sim >= rel_sim_th:
                rel_candidates.append((score, rel))
        rel_candidates.sort(reverse=True)
        for score, rel in rel_candidates[:max_groups]:
            nodes = set()
            for eid in rel_to_eids[rel]:
                e = kvedges[eid]
                nodes.add(e.src)
                nodes.add(e.dst)
                if len(nodes) >= max_nodes_per_group:
                    break
            if nodes:
                groups.append((score, f'relation::{rel}', nodes))
                for nid in list(nodes)[:2]:
                    anchor_reason.setdefault(nid, f'relation:{rel}')

    # 3) fallback: top nodes by q similarity
    if not groups and len(node_names) > 0:
        sims = (node_emb @ q_emb.reshape(-1, 1)).reshape(-1)
        top = np.argsort(-sims)[: min(3, len(node_names))]
        for nid in top:
            groups.append((float(sims[nid]), f'fallback::{node_names[nid]}', {int(nid)}))
            anchor_reason[int(nid)] = f'fallback:{node_names[int(nid)]}'

    # dedup and keep strongest groups
    cleaned: List[Set[int]] = []
    seen_sig: Set[Tuple[int, ...]] = set()
    for _, _, nodes in sorted(groups, key=lambda x: x[0], reverse=True):
        sig = tuple(sorted(nodes))
        if not nodes or sig in seen_sig:
            continue
        seen_sig.add(sig)
        cleaned.append(set(sorted(list(nodes))[:max_nodes_per_group]))
        if len(cleaned) >= max_groups:
            break

    return cleaned, anchor_reason, hard_mentioned_nodes


def build_undirected_edge_index(kvedges: Dict[int, KVEdge]) -> Dict[Tuple[int, int], List[int]]:
    pair_to_eids: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for eid, e in kvedges.items():
        pair = (min(e.src, e.dst), max(e.src, e.dst))
        pair_to_eids[pair].append(eid)
    return pair_to_eids


def edge_cost_from_score(score: float, attr_penalty: float = 0.10, is_attr: bool = False) -> float:
    # higher similarity => lower cost
    base = 1.25 - score
    if is_attr:
        base += attr_penalty
    return max(0.02, float(base))


def build_undirected_graph(
    kvedges: Dict[int, KVEdge],
    node_names: List[str],
) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[Tuple[int, int], float], Dict[Tuple[int, int], List[int]]]:
    pair_to_eids = build_undirected_edge_index(kvedges)
    ug: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    pair_cost: Dict[Tuple[int, int], float] = {}

    for pair, eids in pair_to_eids.items():
        best = min(
            edge_cost_from_score(kvedges[eid].score, is_attr=is_attribute_edge(kvedges[eid], node_names))
            for eid in eids
        )
        u, v = pair
        ug[u].append((v, best))
        ug[v].append((u, best))
        pair_cost[pair] = best

    return dict(ug), pair_cost, pair_to_eids


def dijkstra_paths(ug: Dict[int, List[Tuple[int, float]]], start: int) -> Tuple[Dict[int, float], Dict[int, int]]:
    dist: Dict[int, float] = {start: 0.0}
    parent: Dict[int, int] = {start: -1}
    pq: List[Tuple[float, int]] = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 1e18) + 1e-12:
            continue
        for v, w in ug.get(u, []):
            nd = d + w
            if nd + 1e-12 < dist.get(v, 1e18):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, parent


def restore_path(parent: Dict[int, int], start: int, end: int) -> List[int]:
    if end not in parent:
        return []
    cur = end
    path = [cur]
    guard = 0
    while cur != start and guard < 100000:
        cur = parent.get(cur, -1)
        if cur == -1:
            return []
        path.append(cur)
        guard += 1
    path.reverse()
    return path


def path_cost(path: List[int], pair_cost: Dict[Tuple[int, int], float]) -> float:
    if len(path) <= 1:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        total += pair_cost.get((min(path[i], path[i + 1]), max(path[i], path[i + 1])), 1.0)
    return total


def approximate_gst(
    anchor_groups: List[Set[int]],
    ug: Dict[int, List[Tuple[int, float]]],
    pair_cost: Dict[Tuple[int, int], float],
    node_names: List[str],
    anchor_prize: Optional[Dict[int, float]] = None,
    k_best: int = 5,
    max_roots: int = 24,
    per_group_top_t: int = 4,
) -> List[Dict[str, Any]]:
    """
    Approximate Group Steiner Trees by rooting at strong nodes and linking to one terminal per group.
    """
    if not anchor_groups:
        return []

    candidate_roots: Set[int] = set()
    for g in anchor_groups:
        candidate_roots.update(list(g)[:per_group_top_t])
    if not candidate_roots:
        return []

    candidate_roots = set(list(candidate_roots)[:max_roots])

    results: List[Dict[str, Any]] = []
    root_list = sorted(candidate_roots)
    for root in root_list:
        dist, parent = dijkstra_paths(ug, root)
        if any(all(t not in dist for t in g) for g in anchor_groups):
            continue

        chosen_terminals: List[int] = []
        tree_nodes: Set[int] = set([root])
        tree_pairs: Set[Tuple[int, int]] = set()
        total_cost = 0.0

        for gi, g in enumerate(anchor_groups):
            ranked = sorted(
                [(dist[t], t) for t in g if t in dist],
                key=lambda x: (x[0], -(anchor_prize or {}).get(x[1], 0.0), x[1])
            )[:per_group_top_t]
            if not ranked:
                break
            # pick one terminal with best local objective
            best_obj = None
            best_t = None
            best_path = None
            for d, t in ranked:
                p = restore_path(parent, root, t)
                if not p:
                    continue
                local_cost = path_cost(p, pair_cost)
                reuse_bonus = 0.0
                for i in range(len(p) - 1):
                    pair = (min(p[i], p[i + 1]), max(p[i], p[i + 1]))
                    if pair in tree_pairs:
                        reuse_bonus += pair_cost.get(pair, 0.0)
                obj = local_cost - reuse_bonus - 0.05 * (anchor_prize or {}).get(t, 0.0)
                if best_obj is None or obj < best_obj:
                    best_obj = obj
                    best_t = t
                    best_path = p
            if best_t is None or best_path is None:
                break
            chosen_terminals.append(best_t)
            total_cost += max(0.0, best_obj or 0.0)
            tree_nodes.update(best_path)
            for i in range(len(best_path) - 1):
                tree_pairs.add((min(best_path[i], best_path[i + 1]), max(best_path[i], best_path[i + 1])))
        else:
            internal_bonus = sum((anchor_prize or {}).get(n, 0.0) for n in tree_nodes if n != root) * 0.03
            results.append({
                'root': root,
                'terminals': chosen_terminals,
                'nodes': tree_nodes,
                'pairs': tree_pairs,
                'cost': float(total_cost - internal_bonus),
            })

    results.sort(key=lambda x: (x['cost'], len(x['pairs']), len(x['nodes'])))

    # deduplicate by pair set
    uniq = []
    seen = set()
    for r in results:
        sig = tuple(sorted(r['pairs']))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(r)
        if len(uniq) >= k_best:
            break
    return uniq


def rank_answer_nodes_from_gsts(
    gst_list: List[Dict[str, Any]],
    anchor_nodes: Set[int],
    hard_mentioned_nodes: Set[int],
    node_names: List[str],
    node_emb: np.ndarray,
    question: str,
    embedder: SentenceTransformer,
    batch_size: int,
) -> List[Tuple[float, int]]:
    if not gst_list:
        return []
    q_emb = embed_texts(embedder, [question], batch_size=batch_size)[0]
    q_norm = _norm_lex(question)
    cnt = Counter()
    cost_acc = defaultdict(float)
    depth_bonus = defaultdict(float)

    for rank, gst in enumerate(gst_list):
        weight = 1.0 / (1.0 + rank)
        for n in gst['nodes']:
            if n in anchor_nodes:
                continue
            cnt[n] += weight
            cost_acc[n] += gst['cost']
            depth_bonus[n] += len(gst['pairs']) * 0.02

    ranked = []
    for n, freq in cnt.items():
        name = node_names[n]
        mention_penalty = 0.35 if n in hard_mentioned_nodes or contains_phrase(q_norm, name) else 0.0
        attr_like_bonus = 0.10 if len(name.split()) <= 4 else 0.0
        sim = float(cosine_sim_vec(q_emb, node_emb[n])) if len(node_emb) > n else 0.0
        score = 1.2 * freq - 0.10 * cost_acc[n] + 0.25 * sim + depth_bonus[n] + attr_like_bonus - mention_penalty
        ranked.append((score, n))

    ranked.sort(reverse=True)
    return ranked


def choose_oriented_edges_toward_answers(
    gst_list: List[Dict[str, Any]],
    answer_nodes_ranked: List[Tuple[float, int]],
    kvedges: Dict[int, KVEdge],
    pair_to_eids: Dict[Tuple[int, int], List[int]],
    max_edges: int,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """
    Convert undirected GST(s) into a directed acyclic evidence graph.
    We orient edges toward the chosen answer node so that answers tend to be sinks.
    """
    if not gst_list:
        return set(), set(), set()

    best_tree = gst_list[0]
    tree_nodes: Set[int] = set(best_tree['nodes'])
    tree_pairs: Set[Tuple[int, int]] = set(best_tree['pairs'])

    answer_target = answer_nodes_ranked[0][1] if answer_nodes_ranked else best_tree['root']
    if answer_target not in tree_nodes:
        answer_target = best_tree['root']

    # Build adjacency over tree pairs for BFS distances to answer_target.
    tree_adj: Dict[int, List[int]] = defaultdict(list)
    for u, v in tree_pairs:
        tree_adj[u].append(v)
        tree_adj[v].append(u)

    dist = {answer_target: 0}
    q = [answer_target]
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in tree_adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)

    kept_eids: Set[int] = set()
    sink_nodes: Set[int] = {answer_target}

    for u, v in sorted(tree_pairs):
        if u not in dist or v not in dist:
            continue
        if dist[u] > dist[v]:
            src, dst = u, v
        elif dist[v] > dist[u]:
            src, dst = v, u
        else:
            # tie: orient toward lexically more answer-like side by favoring shorter text as sink-side
            if len((kvedges[pair_to_eids[(u, v)][0]].src_name if pair_to_eids[(u, v)] else '')) <= len((kvedges[pair_to_eids[(u, v)][0]].dst_name if pair_to_eids[(u, v)] else '')):
                src, dst = u, v
            else:
                src, dst = v, u

        chosen = None
        reverse = None
        for eid in pair_to_eids.get((min(u, v), max(u, v)), []):
            e = kvedges[eid]
            if e.src == src and e.dst == dst:
                if chosen is None or e.score > kvedges[chosen].score:
                    chosen = eid
            elif e.src == dst and e.dst == src:
                if reverse is None or e.score > kvedges[reverse].score:
                    reverse = eid

        if chosen is None:
            chosen = reverse
        if chosen is not None:
            kept_eids.add(chosen)
        if len(kept_eids) >= max_edges:
            break

    kept_nodes = set()
    for eid in kept_eids:
        kept_nodes.add(kvedges[eid].src)
        kept_nodes.add(kvedges[eid].dst)
    if answer_target in kept_nodes:
        sink_nodes.add(answer_target)
    return kept_nodes, kept_eids, sink_nodes


def topo_sort(nodes: Set[int], edges_kept: Set[int], kvedges: Dict[int, KVEdge]) -> Optional[List[int]]:
    indeg = {n: 0 for n in nodes}
    out = {n: [] for n in nodes}
    for eid in edges_kept:
        e = kvedges[eid]
        if e.src in nodes and e.dst in nodes:
            out[e.src].append(e.dst)
            indeg[e.dst] += 1
    stack = [n for n in nodes if indeg[n] == 0]
    order: List[int] = []
    while stack:
        u = stack.pop()
        order.append(u)
        for v in out.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
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
            if state.get(v, 0) == 0:
                parent_edge[v] = eid
                res = dfs(v)
                if res is not None:
                    return res
            elif state.get(v, 0) == 1:
                return (u, v, eid)
        state[u] = 2
        return None

    for n in list(nodes):
        if state[n] == 0:
            back = dfs(n)
            if back is not None:
                u, v, back_eid = back
                cyc = [back_eid]
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
    return []


def break_cycles_to_dag_kv(nodes: Set[int], edges_kept: Set[int], kvedges: Dict[int, KVEdge]) -> Set[int]:
    edges_kept = set(edges_kept)
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


# ============================================================
# 7) Export
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


def print_debug_graph(question: str, answer: str, node_names: List[str], kvedges: Dict[int, KVEdge], gst_list: List[Dict[str, Any]], ranked_answers: List[Tuple[float, int]]) -> None:
    print(f'Q: {question}')
    print(f'A: {answer}')
    print('Top evidence edges:')
    for e in sorted(kvedges.values(), key=lambda x: x.score, reverse=True)[:20]:
        e.print()
    if gst_list:
        print('Top GSTs:')
        for i, gst in enumerate(gst_list[:3]):
            print(f'  GST-{i+1}: cost={gst["cost"]:.4f} root={gst["root"]}:{node_names[gst["root"]]}')
            print('   nodes:', [f'{n}:{node_names[n]}' for n in sorted(gst['nodes'])])
    if ranked_answers:
        print('Top answer nodes:')
        for score, nid in ranked_answers[:10]:
            print(f'  {score:.4f} -> {nid}:{node_names[nid]}')
    print('=' * 80)


# ============================================================
# 8) Main pipeline
# ============================================================
def create_dag(args, samples: List[Dict[str, Any]], embedder: SentenceTransformer) -> List[Dict[str, Any]]:
    if args.limit is not None and args.limit > 0:
        samples = samples[:args.limit]

    out_samples: List[Dict[str, Any]] = []
    answer_recall = 0
    graph_recall = 0
    none_sink_recall = 0

    for sample in tqdm(samples, desc='Create UNIQORN-style DAG'):
        question = norm_text(sample.get('question', ''))
        answer = norm_text(sample.get('answer', ''))
        ansN = norm_match(answer)

        node_names, out_adj, kvedges = build_kvedge_graph(
            sample=sample,
            embedder=embedder,
            batch_size=args.batch_size,
            supporting_only=args.supporting_only,
            pred_weight=args.pred_weight,
            title_weight=args.title_weight,
        )

        if not kvedges:
            sample['dag'] = {'kv_nodes': [], 'adj': [], 'meta': {'reason': 'no_kv_edges'}}
            out_samples.append(sample)
            continue

        # graph recall before pruning
        for e in kvedges.values():
            vN = norm_match(e.value)
            if ansN and vN and (vN == ansN or ansN in vN or vN in ansN):
                graph_recall += 1
                break

        node_emb = build_node_embeddings(node_names, embedder, args.batch_size)
        anchor_groups, anchor_reason, hard_mentioned_nodes = detect_anchor_groups(
            question=question,
            node_names=node_names,
            node_emb=node_emb,
            kvedges=kvedges,
            embedder=embedder,
            batch_size=args.batch_size,
            max_groups=args.max_anchor_groups,
            max_nodes_per_group=args.max_nodes_per_group,
            entity_sim_th=args.entity_sim_th,
            rel_sim_th=args.rel_sim_th,
            min_token_overlap=args.min_token_overlap,
        )

        ug, pair_cost, pair_to_eids = build_undirected_graph(kvedges, node_names)
        anchor_nodes = set().union(*anchor_groups) if anchor_groups else set()
        anchor_prize = {}
        for nid in anchor_nodes:
            prize = 0.0
            prize += 1.0 if nid in hard_mentioned_nodes else 0.0
            prize += 0.2 if nid in anchor_reason else 0.0
            anchor_prize[nid] = prize

        gst_list = approximate_gst(
            anchor_groups=anchor_groups,
            ug=ug,
            pair_cost=pair_cost,
            node_names=node_names,
            anchor_prize=anchor_prize,
            k_best=args.top_k_gst,
            max_roots=args.max_roots,
            per_group_top_t=args.per_group_top_t,
        )

        # fallback to top scored local evidence if GST failed
        if not gst_list:
            ranked_edges = sorted(kvedges.values(), key=lambda e: e.score, reverse=True)[:max(1, args.max_edges)]
            kept_kvedges = {e.kid for e in ranked_edges}
            kept_nodes = set()
            for e in ranked_edges:
                kept_nodes.add(e.src)
                kept_nodes.add(e.dst)
            kept_kvedges = break_cycles_to_dag_kv(kept_nodes, kept_kvedges, kvedges)
            ranked_answers = []
            sink_nodes = set()
        else:
            ranked_answers = rank_answer_nodes_from_gsts(
                gst_list=gst_list,
                anchor_nodes=anchor_nodes,
                hard_mentioned_nodes=hard_mentioned_nodes,
                node_names=node_names,
                node_emb=node_emb,
                question=question,
                embedder=embedder,
                batch_size=args.batch_size,
            )
            kept_nodes, kept_kvedges, sink_nodes = choose_oriented_edges_toward_answers(
                gst_list=gst_list,
                answer_nodes_ranked=ranked_answers,
                kvedges=kvedges,
                pair_to_eids=pair_to_eids,
                max_edges=args.max_edges,
            )
            kept_kvedges = break_cycles_to_dag_kv(kept_nodes, kept_kvedges, kvedges)

        if not kept_kvedges:
            sample['dag'] = {'kv_nodes': [], 'adj': [], 'meta': {'reason': 'empty_after_gst'}}
            out_samples.append(sample)
            continue

        # cap node size conservatively by keeping highest-score edges if needed
        if len(kept_nodes) > args.max_nodes:
            ranked = sorted(list(kept_kvedges), key=lambda eid: kvedges[eid].score, reverse=True)
            new_edges: Set[int] = set()
            new_nodes: Set[int] = set()
            for eid in ranked:
                e = kvedges[eid]
                cand_nodes = set(new_nodes)
                cand_nodes.add(e.src)
                cand_nodes.add(e.dst)
                if len(cand_nodes) <= args.max_nodes:
                    new_edges.add(eid)
                    new_nodes = cand_nodes
            if new_edges:
                kept_kvedges = break_cycles_to_dag_kv(new_nodes, new_edges, kvedges)
                kept_nodes = set()
                for eid in kept_kvedges:
                    kept_nodes.add(kvedges[eid].src)
                    kept_nodes.add(kvedges[eid].dst)

        outdeg = defaultdict(int)
        for eid in kept_kvedges:
            outdeg[kvedges[eid].src] += 1
        sinks = [n for n in kept_nodes if outdeg.get(n, 0) == 0]
        if sink_nodes:
            preferred = [n for n in sinks if n in sink_nodes]
            if preferred:
                sinks = preferred + [n for n in sinks if n not in sink_nodes]

        # answer recall on sinks
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
            vN = norm_match(kvedges[eid].value)
            if ansN and vN and (vN == ansN or ansN in vN or vN in ansN):
                none_sink_recall += 1
                break

        kv_nodes, adj = export_kv_nodes_and_adj(kept_kvedges, kvedges, keep_score=args.keep_score)

        goal_ids: List[int] = []
        if ansN:
            for i, kv in enumerate(kv_nodes):
                vN = norm_match(kv.get('value', ''))
                if vN and (vN == ansN or ansN in vN or vN in ansN):
                    goal_ids.append(i)

        meta = {
            'num_entity_nodes': int(len(kept_nodes)),
            'num_kv_edges': int(len(kept_kvedges)),
            'num_kv_nodes': int(len(kv_nodes)),
            'goal_ids': goal_ids,
            'method': 'UNIQORN-style',
            'num_anchor_groups': len(anchor_groups),
            'top_answer_nodes': [int(nid) for _, nid in ranked_answers[:5]] if 'ranked_answers' in locals() else [],
        }
        sample['dag'] = {'kv_nodes': kv_nodes, 'adj': adj, 'meta': meta}
        out_samples.append(sample)

        if args.verbose:
            print_debug_graph(question, answer, node_names, kvedges, gst_list, ranked_answers if 'ranked_answers' in locals() else [])

    if len(samples) > 0:
        print(f'Answer recall: {answer_recall / len(samples):.4f}')
        print(f'Graph  recall: {graph_recall / len(samples):.4f}')
        print(f'None-sink recall: {none_sink_recall / len(samples):.4f}')

    return out_samples


# ============================================================
# 9) CLI
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

    # UNIQORN-style knobs
    ap.add_argument('--pred_weight', type=float, default=0.25, help='weight for relation relevance in edge score')
    ap.add_argument('--title_weight', type=float, default=0.05, help='weight for title relevance in edge score')
    ap.add_argument('--max_anchor_groups', type=int, default=5, help='maximum number of anchor groups')
    ap.add_argument('--max_nodes_per_group', type=int, default=8, help='maximum candidate nodes per anchor group')
    ap.add_argument('--entity_sim_th', type=float, default=0.32, help='entity anchor embedding threshold')
    ap.add_argument('--rel_sim_th', type=float, default=0.30, help='relation anchor embedding threshold')
    ap.add_argument('--min_token_overlap', type=int, default=1, help='minimum lexical token overlap for anchor detection')
    ap.add_argument('--top_k_gst', type=int, default=5, help='number of approximate GSTs to compute/rank')
    ap.add_argument('--max_roots', type=int, default=24, help='number of candidate roots for GST approximation')
    ap.add_argument('--per_group_top_t', type=int, default=4, help='top terminal candidates used per anchor group')
    ap.add_argument('--max_nodes', type=int, default=30, help='max entity nodes in final DAG')
    ap.add_argument('--max_edges', type=int, default=40, help='max KV edges in final DAG')

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

    print(f'[DONE] input={len(samples)} output={len(out)} saved_to={args.output}')


if __name__ == '__main__':
    main()
