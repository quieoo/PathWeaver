import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


_SPACE_RE = re.compile(r'\s+')
_ZW_RE = re.compile(r'[\u200b\u200c\u200d\uFEFF]')
_PUNCT_RE = re.compile(r'[^a-z0-9\s]+')
_STOPWORDS = {
    'the', 'a', 'an', 'of', 'in', 'on', 'at', 'for', 'to', 'from', 'by', 'with', 'is', 'was',
    'were', 'are', 'be', 'been', 'being', 'and', 'or', 'that', 'this', 'these', 'those', 'what',
    'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how', 'as', 'into', 'about'
}


def norm_text(x: Any) -> str:
    if x is None:
        return ''
    s = str(x)
    s = _ZW_RE.sub('', s)
    s = s.strip()
    s = _SPACE_RE.sub(' ', s)
    return s


def norm_match(x: Any) -> str:
    return norm_text(x).lower()


def normalize_lex(s: str) -> str:
    s = (s or '').lower()
    s = _PUNCT_RE.sub(' ', s)
    toks = [t for t in s.split() if t and t not in _STOPWORDS]
    return ' '.join(toks)


def value_matches_answer(value: str, answer: str) -> bool:
    v = norm_match(value)
    a = norm_match(answer)
    if not v or not a:
        return False
    return v == a or a in v or v in a


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith('.jsonl'):
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and 'data' in obj and isinstance(obj['data'], list):
        return obj['data']
    raise ValueError('Unsupported JSON root format.')


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def embed_texts_cached(embedder: SentenceTransformer, texts: Iterable[str], batch_size: int) -> Dict[str, np.ndarray]:
    uniq: List[str] = []
    seen: Set[str] = set()
    for t in texts:
        t = norm_text(t)
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq:
        return {}
    emb = embedder.encode(
        uniq,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if isinstance(emb, list):
        emb = np.array(emb)
    emb = emb.astype(np.float32)
    return {t: emb[i] for i, t in enumerate(uniq)}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def build_reverse_adj(adj: List[List[int]]) -> List[List[int]]:
    n = len(adj)
    rev = [[] for _ in range(n)]
    for i in range(n):
        row = adj[i]
        for j, flag in enumerate(row):
            if flag:
                rev[j].append(i)
    return rev


def ancestor_closure(start: int, rev_adj: List[List[int]]) -> Set[int]:
    seen: Set[int] = set()
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(rev_adj[cur])
    return seen


def terminal_nodes(adj: List[List[int]]) -> List[int]:
    sinks = []
    for i, row in enumerate(adj):
        if not any(row):
            sinks.append(i)
    return sinks


def edge_embedding_score(question_emb: np.ndarray, node: Dict[str, Any], emb_cache: Dict[str, np.ndarray], mode: str) -> float:
    key = norm_text(node.get('key', ''))
    value = norm_text(node.get('value', ''))
    relation = norm_text(node.get('relation', ''))
    src_entity = norm_text(node.get('src_entity', ''))
    dst_entity = norm_text(node.get('dst_entity', ''))

    def emb(text: str) -> np.ndarray:
        if text not in emb_cache:
            raise KeyError(f'missing embedding for text: {text!r}')
        return emb_cache[text]

    scores = {
        'q_key': cosine(question_emb, emb(key)) if key else 0.0,
        'q_value': cosine(question_emb, emb(value)) if value else 0.0,
        'q_relation': cosine(question_emb, emb(relation)) if relation else 0.0,
        'q_src': cosine(question_emb, emb(src_entity)) if src_entity else 0.0,
        'q_dst': cosine(question_emb, emb(dst_entity)) if dst_entity else 0.0,
    }
    if mode in scores:
        return scores[mode]
    if mode == 'max_q_fields':
        return max(scores.values()) if scores else 0.0
    if mode == 'mean_q_fields':
        vals = list(scores.values())
        return float(sum(vals) / max(1, len(vals)))
    raise ValueError(f'Unsupported --edge_score_mode: {mode}')


def summarize(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            'count': 0,
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'p25': None,
            'p75': None,
            'p90': None,
            'p95': None,
            'std': None,
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        'count': int(arr.size),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p90': float(np.percentile(arr, 90)),
        'p95': float(np.percentile(arr, 95)),
        'std': float(arr.std()),
    }


def collect_texts(samples: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for sample in samples:
        texts.append(norm_text(sample.get('question', '')))
        dag = sample.get('dag', {}) or {}
        for node in dag.get('kv_nodes', []) or []:
            texts.extend([
                norm_text(node.get('key', '')),
                norm_text(node.get('value', '')),
                norm_text(node.get('relation', '')),
                norm_text(node.get('src_entity', '')),
                norm_text(node.get('dst_entity', '')),
            ])
    return texts


def compute_cdf(values: List[float]) -> Dict[str, List[float]]:
    if not values:
        return {'x': [], 'y': []}
    arr = np.sort(np.asarray(values, dtype=np.float32))
    y = np.arange(1, arr.size + 1, dtype=np.float32) / float(arr.size)
    return {
        'x': arr.astype(float).tolist(),
        'y': y.astype(float).tolist(),
    }


def plot_cdf(values: List[float], title: str, xlabel: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.figure(figsize=(6.5, 4.8))
    if values:
        arr = np.sort(np.asarray(values, dtype=np.float32))
        y = np.arange(1, arr.size + 1, dtype=np.float32) / float(arr.size)
        plt.plot(arr, y, linewidth=2.0)
        plt.xlim(left=float(arr.min()))
        plt.ylim(0.0, 1.0)
    else:
        plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def get_sink_and_non_sink_ids(adj: List[List[int]]) -> Tuple[List[int], List[int]]:
    sinks = terminal_nodes(adj)
    sink_set = set(sinks)
    non_sinks = [i for i in range(len(adj)) if i not in sink_set]
    return sinks, non_sinks


def analyze(
    samples: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    batch_size: int,
    edge_score_mode: str,
    answer_only: bool = False,
) -> Dict[str, Any]:
    all_texts = collect_texts(samples)
    emb_cache = embed_texts_cached(embedder, all_texts, batch_size=batch_size)

    inbound_edge_counts: List[int] = []
    sink_relevance_scores: List[float] = []
    sink_records: List[Dict[str, Any]] = []

    num_nonempty_dag_samples = 0
    num_sinks_total = 0
    num_samples_total = len(samples)

    answer_recall_hits = 0
    non_sink_recall_hits = 0          # v5-compatible name: answer appears anywhere in final graph
    strict_non_sink_recall_hits = 0   # true answer in non-sink nodes only

    per_sample_metrics: List[Dict[str, Any]] = []

    for sid, sample in enumerate(tqdm(samples, desc='Analyze sink distributions')):
        question = norm_text(sample.get('question', ''))
        answer = norm_text(sample.get('answer', ''))
        dag = sample.get('dag', {}) or {}
        kv_nodes = dag.get('kv_nodes', []) or []
        adj = dag.get('adj', []) or []

        sample_answer_hit_on_sink = False
        sample_answer_hit_anywhere = False
        sample_answer_hit_on_non_sink = False

        if kv_nodes and adj:
            num_nonempty_dag_samples += 1

            q_emb = emb_cache[question]
            rev_adj = build_reverse_adj(adj)
            sinks, non_sinks = get_sink_and_non_sink_ids(adj)

            edge_scores = [edge_embedding_score(q_emb, node, emb_cache, edge_score_mode) for node in kv_nodes]

            sample_answer_hit_on_sink = any(value_matches_answer(kv_nodes[i].get('value', ''), answer) for i in sinks) if answer else False
            sample_answer_hit_anywhere = any(value_matches_answer(node.get('value', ''), answer) for node in kv_nodes) if answer else False
            sample_answer_hit_on_non_sink = any(value_matches_answer(kv_nodes[i].get('value', ''), answer) for i in non_sinks) if answer else False

            sinks_to_analyze = sinks
            if answer_only:
                answer_sinks = [
                    sink_idx for sink_idx in sinks
                    if answer and value_matches_answer(kv_nodes[sink_idx].get('value', ''), answer)
                ]
                sinks_to_analyze = answer_sinks[:1]
            num_sinks_total += len(sinks_to_analyze)

            for sink_idx in sinks_to_analyze:
                anc = ancestor_closure(sink_idx, rev_adj)
                reachable_edge_ids = sorted(anc)
                inbound_count = len(reachable_edge_ids)
                rel_score_sum = float(sum(edge_scores[i] for i in reachable_edge_ids))

                inbound_edge_counts.append(inbound_count)
                sink_relevance_scores.append(rel_score_sum)

                sink_node = kv_nodes[sink_idx]
                sink_records.append({
                    'sample_index': sid,
                    'question': question,
                    'answer': answer,
                    'sink_kv_index': int(sink_idx),
                    'sink_key': norm_text(sink_node.get('key', '')),
                    'sink_value': norm_text(sink_node.get('value', '')),
                    'sink_src_entity': norm_text(sink_node.get('src_entity', '')),
                    'sink_dst_entity': norm_text(sink_node.get('dst_entity', '')),
                    'sink_matches_answer': bool(value_matches_answer(sink_node.get('value', ''), answer)) if answer else False,
                    'reachable_edge_count': int(inbound_count),
                    'relevance_score_sum': float(rel_score_sum),
                    'reachable_edge_ids': reachable_edge_ids,
                    'reachable_edge_scores': [float(edge_scores[i]) for i in reachable_edge_ids],
                })

        if sample_answer_hit_on_sink:
            answer_recall_hits += 1
        if sample_answer_hit_anywhere:
            non_sink_recall_hits += 1
        if sample_answer_hit_on_non_sink:
            strict_non_sink_recall_hits += 1

        per_sample_metrics.append({
            'sample_index': sid,
            'question': question,
            'answer': answer,
            'dag_nonempty': bool(kv_nodes and adj),
            'answer_hit_on_sink': bool(sample_answer_hit_on_sink),
            'answer_hit_anywhere': bool(sample_answer_hit_anywhere),
            'answer_hit_on_non_sink': bool(sample_answer_hit_on_non_sink),
        })

    summary = {
        'sink_inbound_edge_count_summary': summarize(inbound_edge_counts),
        'sink_relevance_score_summary': summarize(sink_relevance_scores),
        'dataset_overview': {
            'num_samples_total': int(num_samples_total),
            'num_nonempty_dag_samples': int(num_nonempty_dag_samples),
            'num_sinks_total': int(num_sinks_total),
            'avg_sinks_per_nonempty_sample': float(num_sinks_total / max(1, num_nonempty_dag_samples)),
            'answer_only': bool(answer_only),
        },
        'retrieval_metrics': {
            'answer_recall': float(answer_recall_hits / max(1, num_samples_total)),
            'non_sink_recall': float(non_sink_recall_hits / max(1, num_samples_total)),
            'strict_non_sink_recall': float(strict_non_sink_recall_hits / max(1, num_samples_total)),
            'answer_recall_hits': int(answer_recall_hits),
            'non_sink_recall_hits': int(non_sink_recall_hits),
            'strict_non_sink_recall_hits': int(strict_non_sink_recall_hits),
            'metric_note': {
                'answer_recall': 'Whether the answer value appears on any sink kv_node.',
                'non_sink_recall': 'V5-compatible: whether the answer value appears anywhere in the final kv_nodes, despite the historical name.',
                'strict_non_sink_recall': 'Whether the answer value appears on any non-sink kv_node only.',
            },
        },
    }

    if answer_only:
        summary['retrieval_metrics']['answer_sink_presence_probability'] = float(answer_recall_hits / max(1, num_samples_total))
        summary['retrieval_metrics']['answer_sink_presence_hits'] = int(answer_recall_hits)
        summary['retrieval_metrics']['metric_note']['answer_sink_presence_probability'] = (
            'When --answer_only is set: probability that a sample contains at least one sink whose value matches the answer.'
        )

    return {
        'summary': summary,
        'full_distribution': {
            'sink_inbound_edge_counts': inbound_edge_counts,
            'sink_relevance_scores': sink_relevance_scores,
            'sink_records': sink_records,
            'per_sample_metrics': per_sample_metrics,
        },
        'cdf': {
            'sink_inbound_edge_count_cdf': compute_cdf(inbound_edge_counts),
            'sink_relevance_score_cdf': compute_cdf(sink_relevance_scores),
        },
    }


def main():
    ap = argparse.ArgumentParser(description='Analyze sink inbound-edge counts / relevance scores and draw CDF plots.')
    ap.add_argument('--input', type=str, required=True, help='Path to infer output json/jsonl that contains question/answer/dag.')
    ap.add_argument('--st_model', type=str, required=True, help='SentenceTransformer model path/name.')
    ap.add_argument('--output', type=str, required=True, help='Path to save analysis json.')
    ap.add_argument('--embed_batch_size', type=int, default=256)
    ap.add_argument('--plot_dir', type=str, default='', help='Directory to save generated CDF plots. Default: same dir as --output.')
    ap.add_argument('--plot_prefix', type=str, default='', help='Optional filename prefix for generated plots.')
    ap.add_argument(
        '--edge_score_mode',
        type=str,
        default='q_key',
        choices=['q_key', 'q_value', 'q_relation', 'q_src', 'q_dst', 'max_q_fields', 'mean_q_fields'],
        help='Embedding-based edge relevance used before summing over all edges that can reach a sink.',
    )
    ap.add_argument(
        '--answer_only',
        action='store_true',
        help='If set, only count one sink per sample whose value matches the answer; if multiple match, keep the first one.',
    )
    args = ap.parse_args()

    samples = read_json_or_jsonl(args.input)
    embedder = SentenceTransformer(args.st_model)
    result = analyze(
        samples,
        embedder,
        batch_size=args.embed_batch_size,
        edge_score_mode=args.edge_score_mode,
        answer_only=args.answer_only,
    )
    write_json(args.output, result)

    plot_dir = args.plot_dir.strip() or (os.path.dirname(args.output) or '.')
    base = os.path.splitext(os.path.basename(args.output))[0]
    prefix = f'{args.plot_prefix.strip()}_' if args.plot_prefix.strip() else ''

    path_count = os.path.join(plot_dir, f'{prefix}{base}_sink_inbound_edge_count_cdf.png')
    path_score = os.path.join(plot_dir, f'{prefix}{base}_sink_relevance_score_cdf.png')

    plot_cdf(
        result['full_distribution']['sink_inbound_edge_counts'],
        title='CDF of sink inbound-edge counts',
        xlabel='Reachable edge count per sink',
        output_path=path_count,
    )
    plot_cdf(
        result['full_distribution']['sink_relevance_scores'],
        title=f'CDF of sink relevance scores ({args.edge_score_mode})',
        xlabel='Sum of embedding-based relevance scores',
        output_path=path_score,
    )

    print('[Sink inbound-edge count summary]')
    print(json.dumps(result['summary']['sink_inbound_edge_count_summary'], ensure_ascii=False, indent=2))
    print('[Sink relevance-score summary]')
    print(json.dumps(result['summary']['sink_relevance_score_summary'], ensure_ascii=False, indent=2))
    print('[Retrieval metrics]')
    print(json.dumps(result['summary']['retrieval_metrics'], ensure_ascii=False, indent=2))
    print(f'[Saved JSON] {args.output}')
    print(f'[Saved Plot] {path_count}')
    print(f'[Saved Plot] {path_score}')


if __name__ == '__main__':
    main()
