#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert AllTriple dataset -> kblam-qa dataset (top-1 2-hop path by embedding similarity)

✅ New behavior (per your request):
1) Do NOT use dataset key_string to judge triple type.
2) Treat EVERY triple as BOTH:
   - attribute triple  : "the <description_type> of <name>"
   - relation triple   : "<name> <description_type>"
3) New build_path_strings(t1, t2) returns 4 possible 2-hop path strings:
   AA, AR, RA, RR
4) Flatten ALL candidates -> embed -> cosine similarity -> top-1
5) For final top-1 path, regenerate each triple's key_string by the chosen variant:
   attribute: "the <description_type> of <name>"
   relation : "<name> <description_type>"

Input sample (AllTriple):
{
  "question": str,
  "answer": str,
  "supporting_facts": [[title1, sent_id], [title2, sent_id]],
  "context": [
    {"title": str, "triple_list": [ {triple}, ... ]},
    ...
  ]
}

Output sample (kblam-qa):
{
  "Q": str,
  "A": str,
  "triple_lists": [ {triple}, {triple} ]   # top-1 2-hop path (2 triples)
}
"""

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import re

# -----------------------------
# IO helpers
# -----------------------------
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

@dataclass
class path_key_key:
    path: str
    key1: str
    key2: str
    desc1: str
    desc2: str

def build_path_and_key_string(t1: dict, t2: dict):
    is_rel1 = t1["type"] == "RELATION"
    is_rel2 = t2["type"] == "RELATION"

    # AA
    if not is_rel1 and not is_rel2:
        return path_key_key(
            f"the {t2['description_type']} of the {t1['description_type']} of {t1['name']}",
            f"the {t1['description_type']} of {t1['name']}",
            f"the {t2['description_type']} of {t2['name']}",
            t1["description"],
            t2["description"],
        )

    # RA
    if is_rel1 and not is_rel2:
        return path_key_key(
            f"the {t2['description_type']} of which {t1['name']} {t1['description_type']}",
            f"{t1['name']} {t1['description_type']}",
            f"the {t2['description_type']} of {t2['name']}",
            t1["description"],
            t2["description"],
        )

    # AR
    if not is_rel1 and is_rel2:
        return path_key_key(
            f"the one that the {t1['description_type']} of {t1['name']} {t2['description_type']}",
            f"the {t1['description_type']} of {t1['name']}",
            f"{t2['name']} {t2['description_type']}",
            t1["description"],
            t2["description"],
        )

    # RR
    return path_key_key(
        f"the one that {t1['name']} {t1['description_type']} {t2['description_type']}",
        f"{t1['name']} {t1['description_type']}",
        f"{t2['name']} {t2['description_type']}",
        t1["description"],
        t2["description"],
    )


def build_path_strings(t1: dict, t2: dict):
    is_rel1 = t1["type"] == "RELATION"
    is_rel2 = t2["type"] == "RELATION"

    # AA
    if not is_rel1 and not is_rel2:
        return [build_path_and_key_string(t1, t2)]

    # RA
    if is_rel1 and not is_rel2:
        t1_attr = t1.copy()
        t1_attr["description_type"] = t1_attr["attribute_desc_alias"]
        t1_attr["type"] = "ATTRIBUTE"

        pks1 = build_path_and_key_string(t1, t2)
        pks2 = build_path_and_key_string(t1_attr, t2)
        return [pks1, pks2]

    # AR
    if not is_rel1 and is_rel2:
        t2_attr = t2.copy()
        t2_attr["description_type"] = t2_attr["attribute_desc_alias"]
        t2_attr["type"] = "ATTRIBUTE"

        pks1 = build_path_and_key_string(t1, t2)
        pks2 = build_path_and_key_string(t1, t2_attr)
        return [pks1, pks2]

    # RR
    t1_attr = t1.copy()
    t1_attr["description_type"] = t1_attr["attribute_desc_alias"]
    t1_attr["type"] = "ATTRIBUTE"

    t2_attr = t2.copy()
    t2_attr["description_type"] = t2_attr["attribute_desc_alias"]
    t2_attr["type"] = "ATTRIBUTE"

    pks1 = build_path_and_key_string(t1, t2)
    pks2 = build_path_and_key_string(t1_attr, t2)
    pks3 = build_path_and_key_string(t1, t2_attr)
    pks4 = build_path_and_key_string(t1_attr, t2_attr)

    return [pks1, pks2, pks3, pks4]

    

# -----------------------------
# Core logic
# -----------------------------
def get_support_titles(sample: Dict[str, Any]) -> List[str]:
    """
    supporting_facts: [[title, sent_id], [title, sent_id]]
    Return unique titles preserving order.
    """
    titles = []
    for x in sample.get("supporting_facts", []):
        if isinstance(x, list) and len(x) >= 1 and isinstance(x[0], str):
            titles.append(x[0])

    seen = set()
    uniq = []
    for t in titles:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def collect_target_triples(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Filter contexts by supporting_facts titles, then merge their triple_list.
    """
    support_titles = set(get_support_titles(sample))
    triples: List[Dict[str, Any]] = []

    for ctx in sample.get("context", []):
        if not isinstance(ctx, dict):
            continue
        title = ctx.get("title")
        if title in support_titles:
            for t in ctx.get("triple_list", []) or []:
                if isinstance(t, dict):
                    triples.append(t)
    return triples


def build_edges(triples: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """
    A directed edge i->j exists if triples[i]["description"] == triples[j]["name"]
    """
    name_to_indices: Dict[str, List[int]] = {}
    for j, t in enumerate(triples):
        nm = t.get("name")
        if isinstance(nm, str) and nm:
            name_to_indices.setdefault(nm, []).append(j)

    edges: List[Tuple[int, int]] = []
    for i, t in enumerate(triples):
        desc = t.get("description")
        if not isinstance(desc, str) or not desc:
            continue
        for j in name_to_indices.get(desc, []):
            edges.append((i, j))
    return edges

import re
from typing import List, Dict, Tuple


def _norm_text(s: str) -> str:
    """
    Normalize text for approximate matching:
    - lowercase
    - remove punctuation
    - collapse spaces
    """
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_edges_approx(
    triples: List[Dict[str, any]],
    min_len: int = 3,
) -> List[Tuple[int, int]]:
    """
    Approximate edge:
      i -> j if normalized(description_i) contains normalized(name_j)
         or vice versa (length-gated)

    This is recall-oriented; precision is handled by path embedding later.
    """

    norm_names = []
    norm_descs = []

    for t in triples:
        name = t.get("name", "")
        desc = t.get("description", "")
        norm_names.append(_norm_text(name) if isinstance(name, str) else "")
        norm_descs.append(_norm_text(desc) if isinstance(desc, str) else "")

    edges: List[Tuple[int, int]] = []

    for i in range(len(triples)):
        di = norm_descs[i]
        if not di:
            continue

        for j in range(len(triples)):
            if i == j:
                continue

            nj = norm_names[j]
            if not nj or len(nj) < min_len:
                continue

            # core approximate match
            if nj in di or di in nj:
                edges.append((i, j))
            # if nj in di or di in nj or set(nj.split()) & set(di.split()):
            #     edges.append((i, j))

    return edges

# 全局变量
graph_recall=0
def pick_topk_path_multi_view(
    question: str,
    answer: str,
    triples: List[Dict[str, Any]],
    edges: List[Tuple[int, int]],
    embedder: SentenceTransformer,
    k: int = 1,
    batch_size: int = 256,
) -> Optional[Tuple[path_key_key, float]]:
    """
    Return (i, j, variant_id, score) for the best 2-hop path candidate.
    variant_id in {0,1,2,3} mapping to AA,AR,RA,RR.

    score is cosine similarity (embeddings normalized => dot product).
    """
    global graph_recall
    if not edges:
        print(f"No edges found, question: {question}")
        return None

    # ------------------------------------------------
    # 1️⃣ Collect & flatten all path candidates
    # ------------------------------------------------
    path_texts: List[str] = []
    pks_list: List[path_key_key] = []

    for (i, j) in edges:
        try:
            pks_list.extend(build_path_strings(triples[i], triples[j]))
        except Exception as e:
            raise ValueError(f"Error building path strings for {triples[i]} -> {triples[j]}: {e}")
    
    has_gold_path = False
    for pks in pks_list:
        if answer in pks.desc2:
            has_gold_path = True
            break

    if has_gold_path:
        graph_recall += 1

    for pks in pks_list:
        path_texts.append(pks.path)

    if not path_texts:
        print("No path texts found")
        return None

    # ------------------------------------------------
    # 2️⃣ Encode question & paths
    # ------------------------------------------------
    q_emb = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]  # (D,)

    p_emb = embedder.encode(
        path_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch_size,
    )  # (P, D)

    # ------------------------------------------------
    # 3️⃣ Cosine similarity & argmax
    # ------------------------------------------------
    sims = p_emb @ q_emb  # (P,)

    # 按照相似度排序
    sorted_idx = np.argsort(-sims)
    
    # for i in sorted_idx:
    #     pks_text = path_texts[i]
    #     print(f"{i}: {pks_text} => {sims[i]:.4f}")

    topk_idx = sorted_idx[:k].tolist()

    topk_paths = [pks_list[i] for i in topk_idx]
    topk_scores = [float(sims[i]) for i in topk_idx]

    return topk_paths, topk_scores


def pick_topk_path_v2(
    question: str,
    answer: str,
    triples: List[Dict[str, Any]],
    edges: List[Tuple[int, int]],
    embedder: SentenceTransformer,
    k: int = 1,
    batch_size: int = 256,
    *,
    decay: float = 0.5,
    lambda_rr: float = 0.5,
    verbose: bool = True,
) -> Optional[Tuple[List[path_key_key], List[float]]]:
    """
    Same interface as pick_topk_path_multi_view, but score is computed by
    weighted key-level similarity of two hops:

        score = w1 * sim(q, key1) + w2 * decay * sim(q, key2)

    Return:
        (topk_paths, topk_scores)
        - topk_paths: List[path_key_key]
        - topk_scores: List[float]
    """
    global graph_recall

    if not edges:
        print(f"No edges found, question: {question}")
        return None

    assert 0.0 < decay <= 1.0, "decay must be in (0, 1]."

    # ------------------------------------------------
    # 1️⃣ Collect & flatten all path candidates (复用原逻辑)
    # ------------------------------------------------
    pks_list: List[path_key_key] = []
    for (i, j) in edges:
        try:
            pks_list.extend(build_path_strings(triples[i], triples[j]))
        except Exception as e:
            raise ValueError(
                f"Error building path strings for {triples[i]} -> {triples[j]}: {e}"
            )

    if not pks_list:
        print("No path candidates found")
        return None

    # graph_recall (复用原逻辑：answer 是否出现在 desc2)
    has_gold_path = False
    for pks in pks_list:
        if answer in pks.desc2:
            has_gold_path = True
            break
    if has_gold_path:
        graph_recall += 1

    # ------------------------------------------------
    # 2️⃣ Encode question & all key1/key2 in batch
    # ------------------------------------------------
    q_emb = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]  # (D,)

    key1_texts = [pks.key1 for pks in pks_list]
    key2_texts = [pks.key2 for pks in pks_list]

    k1_emb = embedder.encode(
        key1_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch_size,
    )  # (P, D)

    k2_emb = embedder.encode(
        key2_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch_size,
    )  # (P, D)

    # ------------------------------------------------
    # 3️⃣ Similarity + weighted fusion (第二跳衰减)
    # ------------------------------------------------
    sims1 = k1_emb @ q_emb  # (P,)
    sims2 = k2_emb @ q_emb  # (P,)

    # scores = (w1 * sims1) + (w2 * decay * sims2)  # (P,)

    sims12 = np.sum(k1_emb * k2_emb, axis=1)  # cosine since normalized
    scores = (
        sims1
        + decay * sims2
        - lambda_rr * sims12
    )


    # ------------------------------------------------
    # 4️⃣ Sort & top-k (与 multi_view 一样)
    # ------------------------------------------------
    sorted_idx = np.argsort(-scores)
    topk_idx = sorted_idx[:k].tolist()

    topk_paths = [pks_list[i] for i in topk_idx]
    topk_scores = [float(scores[i]) for i in topk_idx]
    if verbose:
        if answer not in topk_paths[0].desc2:
            print(f"Question: {question}")
            print(f"Answer LOSS---")
            for i in sorted_idx:
                pks = pks_list[i]
                print(f"{scores[i]:.4f} {pks}")



    return topk_paths, topk_scores



def pick_top1_path_multi_view(
    question: str,
    answer: str,
    triples: List[Dict[str, Any]],
    edges: List[Tuple[int, int]],
    embedder: SentenceTransformer,
    batch_size: int = 256,
) -> Optional[Tuple[path_key_key, float]]:
    """
    Return (i, j, variant_id, score) for the best 2-hop path candidate.
    variant_id in {0,1,2,3} mapping to AA,AR,RA,RR.

    score is cosine similarity (embeddings normalized => dot product).
    """
    global graph_recall
    if not edges:
        print(f"No edges found, question: {question}")
        return None

    # ------------------------------------------------
    # 1️⃣ Collect & flatten all path candidates
    # ------------------------------------------------
    path_texts: List[str] = []
    pks_list: List[path_key_key] = []

    for (i, j) in edges:
        try:
            pks_list.extend(build_path_strings(triples[i], triples[j]))
        except Exception as e:
            raise ValueError(f"Error building path strings for {triples[i]} -> {triples[j]}: {e}")
    
    has_gold_path = False
    for pks in pks_list:
        if answer in pks.desc2:
            has_gold_path = True
            break

    if has_gold_path:
        graph_recall += 1

    for pks in pks_list:
        path_texts.append(pks.path)

    if not path_texts:
        print("No path texts found")
        return None

    # ------------------------------------------------
    # 2️⃣ Encode question & paths
    # ------------------------------------------------
    q_emb = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]  # (D,)

    p_emb = embedder.encode(
        path_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch_size,
    )  # (P, D)

    # ------------------------------------------------
    # 3️⃣ Cosine similarity & argmax
    # ------------------------------------------------
    sims = p_emb @ q_emb  # (P,)
    best_idx = int(np.argmax(sims))

    # 按照相似度排序
    sorted_idx = np.argsort(-sims)
    
    # for i in sorted_idx:
    #     pks_text = path_texts[i]
    #     print(f"{i}: {pks_text} => {sims[i]:.4f}")

    return pks_list[best_idx], float(sims[best_idx])




def fallback_top2_triples(
    question: str,
    triples: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    batch_size: int = 256,
    topk: int = 2,
) -> Optional[List[int]]:
    """
    Fallback when no 2-hop edge exists.
    Return indices of top-k single triples by similarity(Q, triple.key_string).
    """
    if not triples:
        return None

    texts = []
    for t in triples:
        ks = t.get("key_string")
        if not isinstance(ks, str) or not ks.strip():
            ks = f"{t.get('name','')} {t.get('description_type','')}".strip()
        texts.append(ks)

    # embeddings
    q_emb = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]                                  # (D,)

    t_emb = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch_size,
    )                                     # (N, D)

    sims = t_emb @ q_emb                  # cosine similarity
    k = min(topk, len(triples))
    topk_idx = np.argsort(-sims)[:k]      # descending

    return topk_idx.tolist()



def convert_dataset(
    samples: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    max_samples: Optional[int] = None,
    k: int = 1,
    batch_size: int = 256,
    keep_score: bool = False,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    if max_samples is None:
        max_samples = len(samples)

    out: List[Dict[str, Any]] = []
    pick_time: List[float] = []
    global graph_recall
    answer_recall=0

    for i in tqdm(range(len(samples))):
        if len(out) >= max_samples:
            break
        s=samples[i]
        q = s.get("question")
        a = s.get("answer")
        if not isinstance(q, str) or not isinstance(a, str):
            continue

        triples = collect_target_triples(s)
        # edges = build_edges(triples)
        edges = build_edges_approx(triples)

        st = time.time()
        # ret = pick_top1_path_multi_view(q, a, triples, edges, embedder, batch_size=batch_size)
        # ret = pick_topk_path_multi_view(q, a, triples, edges, embedder, batch_size=batch_size, k=k)
        ret=pick_topk_path_v2(q, a, triples, edges, embedder, batch_size=batch_size, k=k, verbose=False)

        if ret is None: 
            continue
        pick_time.append(time.time() - st)
        path_triple_lists = []
        path_score_list = []
        found_answer = False
        pks_list, scores = ret
        for best_pks, score in zip(pks_list, scores):
            path_triple = [
                {
                    "key_string": best_pks.key1,
                    "description": best_pks.desc1,
                },
                {
                    "key_string": best_pks.key2,
                    "description": best_pks.desc2,
                },
            ]
            path_triple_lists.append(path_triple)
            path_score_list.append(score)
            if a in best_pks.desc2:
                found_answer = True
        if found_answer:
            answer_recall+=1

        row: Dict[str, Any] = {"id": s["_id"], "Q": q, "A": a, "triple_lists": path_triple_lists}
        if keep_score:
            row["_path_scores"] = path_score_list
        out.append(row)


    if pick_time:
        print(f"Average pick time: {float(np.mean(pick_time)):.4f}s")
    else:
        print("Average pick time: N/A (no samples processed)")
    
    print(f"Graph recall: {graph_recall/len(out):.4f}")
    print(f"Answer recall: {answer_recall/len(out):.4f}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="all_triples dataset file (.json or .jsonl)")
    ap.add_argument("--output", required=True, help="output kblam-qa jsonl/json path")
    ap.add_argument(
        "--st_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name or local path",
    )
    ap.add_argument("--batch_size", type=int, default=256, help="embedding batch size")
    ap.add_argument("--keep_score", action="store_true", help="keep similarity scores & variant id")
    ap.add_argument("--limit", type=int, default=None, help="limit number of samples (take last N)")
    ap.add_argument("--k", type=int, default=1, help="top-k paths to pick")
    args = ap.parse_args()

    embedder = SentenceTransformer(args.st_model)

    samples = read_json_or_jsonl(args.input)

    out = convert_dataset(
        samples=samples,
        embedder=embedder,
        max_samples=args.limit,
        k=args.k,
        batch_size=args.batch_size,
        keep_score=args.keep_score,
    )

    if args.output.endswith(".jsonl"):
        write_jsonl(args.output, out)
    elif args.output.endswith(".json"):
        write_json(args.output, out)
    else:
        raise ValueError(f"Unknown file format: {args.output}")

    print(f"[DONE] input={len(samples)}  output={len(out)}  saved_to={args.output}")


if __name__ == "__main__":
    main()
