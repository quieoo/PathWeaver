import json
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np


def load_json_or_jsonl(path: str) -> Any:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unwrap_dataset_container(data: Any, source_name: str) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "samples"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(
        f"Unsupported {source_name} format. Expected a list or a JSON object "
        "with one of keys: data/examples/samples."
    )


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    return _unwrap_dataset_container(load_json_or_jsonl(dataset_path), "dataset")


def load_queryset(queryset_path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if queryset_path is None:
        return None
    return _unwrap_dataset_container(load_json_or_jsonl(queryset_path), "queryset")


def get_sample_id(row: Dict[str, Any], fallback: Any) -> str:
    sample_id = row.get("id", row.get("_id", fallback))
    return str(sample_id)


def get_question(row: Dict[str, Any]) -> Any:
    return row.get("question", row.get("Q"))


def get_answer(row: Dict[str, Any]) -> Any:
    return row.get("answer", row.get("A"))


def as_sentences(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(sentence).strip() for sentence in value if str(sentence).strip()]
    return [str(value).strip()] if str(value).strip() else []


def safe_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def iter_row_contexts(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize Hotpot/2Wiki context entries, MuSiQue paragraphs, and SQuAD context."""
    contexts: List[Dict[str, Any]] = []

    raw_context = row.get("context", [])
    if isinstance(raw_context, str):
        text = raw_context.strip()
        if text:
            contexts.append(
                {
                    "title": str(row.get("title", "")).strip(),
                    "sentences": as_sentences(text),
                    "text": text,
                    "idx": row.get("idx"),
                    "is_supporting": bool(row.get("is_supporting", False)),
                }
            )
    else:
        for ctx in raw_context or []:
            title = ""
            sentences: List[str] = []
            idx = None
            is_supporting = False

            if isinstance(ctx, dict):
                title = str(ctx.get("title", ctx.get("paragraph_title", ""))).strip()
                sentences = as_sentences(
                    ctx.get("sentences")
                    or ctx.get("paragraph_text")
                    or ctx.get("text")
                    or ctx.get("context")
                )
                idx = ctx.get("idx")
                is_supporting = bool(ctx.get("is_supporting", False))
            elif isinstance(ctx, (list, tuple)) and len(ctx) >= 2:
                title = str(ctx[0]).strip()
                sentences = as_sentences(ctx[1])
                if len(ctx) >= 3:
                    is_supporting = bool(ctx[2])

            text = " ".join(sentence for sentence in sentences if sentence).strip()
            if text:
                contexts.append(
                    {
                        "title": title,
                        "sentences": sentences,
                        "text": text,
                        "idx": idx,
                        "is_supporting": is_supporting,
                    }
                )

    for para in row.get("paragraphs", []) or []:
        if isinstance(para, dict):
            title = str(para.get("title", para.get("paragraph_title", ""))).strip()
            text = str(
                para.get("paragraph_text")
                or para.get("text")
                or para.get("context")
                or ""
            ).strip()
            idx = para.get("idx")
            is_supporting = bool(para.get("is_supporting", False))
        else:
            title = ""
            text = str(para).strip()
            idx = None
            is_supporting = False

        if not text:
            continue
        contexts.append(
            {
                "title": title,
                "sentences": as_sentences(text),
                "text": text,
                "idx": idx,
                "is_supporting": is_supporting,
            }
        )

    return contexts


def stringify_context(title: Any, text: Any) -> str:
    title = str(title or "").strip()
    body = str(text or "").strip()
    if title:
        return f"Title: {title}\n{body}".strip()
    return body


def iter_row_docs(row: Dict[str, Any]) -> Iterable[str]:
    for ctx in iter_row_contexts(row):
        doc = stringify_context(ctx.get("title", ""), ctx.get("text", ""))
        if doc:
            yield doc


def build_memory_docs(dataset: List[Dict[str, Any]], memory_rows: Optional[int]) -> List[str]:
    docs: List[str] = []
    seen = set()
    rows = dataset if memory_rows is None else dataset[:memory_rows]
    for row in rows:
        for doc in iter_row_docs(row):
            if doc and doc not in seen:
                seen.add(doc)
                docs.append(doc)
    return docs


def extract_doc_title(doc_text: str) -> str:
    first_line = str(doc_text).splitlines()[0].strip() if doc_text else ""
    if first_line.startswith("Title:"):
        return first_line.split("Title:", 1)[1].strip()
    return first_line


def extract_oracle_contexts(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    contexts = iter_row_contexts(row)
    by_title: Dict[str, List[Dict[str, Any]]] = {}
    by_idx: Dict[int, Dict[str, Any]] = {}
    for ctx in contexts:
        title = str(ctx.get("title", "")).strip()
        if title:
            by_title.setdefault(title, []).append(ctx)
        idx = safe_int(ctx.get("idx"))
        if idx is not None:
            by_idx[idx] = ctx

    oracle_contexts: List[str] = []
    oracle_titles: List[str] = []
    seen_texts = set()

    def add_context(ctx: Optional[Dict[str, Any]], sent_idx: Optional[int] = None):
        if not ctx:
            return
        text = ""
        sentences = ctx.get("sentences", [])
        if sent_idx is not None and 0 <= sent_idx < len(sentences):
            text = str(sentences[sent_idx]).strip()
        if not text:
            text = str(ctx.get("text", "")).strip()
        if not text or text in seen_texts:
            return
        seen_texts.add(text)
        oracle_contexts.append(text)
        title = str(ctx.get("title", "")).strip()
        if title:
            oracle_titles.append(title)

    for qd in row.get("question_decomposition", []) or []:
        if not isinstance(qd, dict):
            continue
        support_idx = safe_int(qd.get("paragraph_support_idx"))
        if support_idx is not None:
            add_context(by_idx.get(support_idx))

    for ctx in contexts:
        if ctx.get("is_supporting"):
            add_context(ctx)

    for sp in row.get("supporting_facts", []) or []:
        title = ""
        sent_idx = None
        if isinstance(sp, str):
            title = sp.strip()
        elif isinstance(sp, (list, tuple)) and len(sp) > 0:
            title = str(sp[0]).strip()
            sent_idx = safe_int(sp[1]) if len(sp) > 1 else None
        if not title:
            continue
        for ctx in by_title.get(title, []):
            add_context(ctx, sent_idx)

    return oracle_contexts, oracle_titles


def get_supporting_fact_titles(row: Dict[str, Any]) -> List[str]:
    _, oracle_titles = extract_oracle_contexts(row)
    titles: List[str] = list(oracle_titles)

    for sp in row.get("supporting_facts", []) or []:
        if isinstance(sp, str):
            title = sp.strip()
            if title:
                titles.append(title)
        elif isinstance(sp, (list, tuple)) and len(sp) > 0:
            title = str(sp[0]).strip()
            if title:
                titles.append(title)

    deduped: List[str] = []
    seen = set()
    for title in titles:
        if title not in seen:
            seen.add(title)
            deduped.append(title)
    return deduped


def build_query_samples(dataset: List[Dict[str, Any]], n_samples: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    if len(dataset) == 0 or n_samples <= 0:
        return []
    if seed is not None:
        rng = np.random.default_rng(seed)
        query_idx = rng.integers(0, len(dataset), size=n_samples)
    else:
        query_idx = list(range(min(n_samples, len(dataset))))
    samples = []
    for idx in query_idx:
        row = dataset[int(idx)]
        question = get_question(row)
        answer = get_answer(row)
        if question is None or answer is None:
            raise KeyError(f"Query sample is missing question/answer fields: {get_sample_id(row, '<unknown>')}")
        samples.append(
            {
                "id": row.get("id", row.get("_id")),
                "question": question,
                "answer": answer,
                "supporting_titles": get_supporting_fact_titles(row),
            }
        )
    return samples


def compute_retrieval_recall_stats(
    gold_titles: List[str],
    retrieved_titles: List[str],
) -> Dict[str, float]:
    gold_set = {title.strip() for title in gold_titles if str(title).strip()}
    retrieved_set = {title.strip() for title in retrieved_titles if str(title).strip()}
    if not gold_set:
        return {
            "recall": 0.0,
            "hit": 0.0,
            "all_hit": 0.0,
        }

    matched = gold_set & retrieved_set
    return {
        "recall": len(matched) / len(gold_set),
        "hit": float(len(matched) > 0),
        "all_hit": float(matched == gold_set),
    }
