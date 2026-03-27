import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


@dataclass
class SampleMetric:
    sample_id: Any
    question: str
    answer: str
    coverage: Dict[str, Any] = field(default_factory=dict)
    structure: Dict[str, Any] = field(default_factory=dict)
    relevance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sample_id': self.sample_id,
            'question': self.question,
            'answer': self.answer,
            'coverage': self.coverage,
            'structure': self.structure,
            'relevance': self.relevance,
        }


class MetricsAggregator:
    def __init__(self) -> None:
        self.count = 0
        self.sum_fields = defaultdict(float)
        self.bool_fields = defaultdict(int)
        self.samples: List[Dict[str, Any]] = []

    def add(self, metric: SampleMetric) -> None:
        self.count += 1
        d = metric.to_dict()
        self.samples.append(d)
        for section in ('coverage', 'structure', 'relevance'):
            for k, v in d[section].items():
                name = f'{section}.{k}'
                if isinstance(v, bool):
                    self.bool_fields[name] += int(v)
                elif isinstance(v, (int, float)) and v is not None:
                    self.sum_fields[name] += float(v)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'num_samples': self.count,
            'coverage': {},
            'structure': {},
            'relevance': {},
        }
        for name, v in self.bool_fields.items():
            sec, key = name.split('.', 1)
            out[sec][key] = _safe_div(v, self.count)
        for name, v in self.sum_fields.items():
            sec, key = name.split('.', 1)
            if name in self.bool_fields:
                continue
            out[sec][key] = _safe_div(v, self.count)
        return out

    def concise_summary(self) -> Dict[str, Any]:
        s = self.summary()
        return {
            'num_samples': s.get('num_samples', 0),
            'coverage': {
                'graph_has_answer': s.get('coverage', {}).get('graph_has_answer', 0.0),
                'subgraph_has_answer': s.get('coverage', {}).get('subgraph_has_answer', 0.0),
                'answer_in_sink': s.get('coverage', {}).get('answer_in_sink', 0.0),
                'answer_edge_coverage_ratio': s.get('coverage', {}).get('answer_edge_coverage_ratio', 0.0),
            },
            'structure': {
                'supported_answer_sink_ge2': s.get('structure', {}).get('supported_answer_sink_ge2', 0.0),
                'weakly_isolated_answer_sink': s.get('structure', {}).get('weakly_isolated_answer_sink', 0.0),
                'num_kept_nodes': s.get('structure', {}).get('num_kept_nodes', 0.0),
                'num_kept_edges': s.get('structure', {}).get('num_kept_edges', 0.0),
            },
            'relevance': {
                'relevant_answer_parent_ge_0_2': s.get('relevance', {}).get('relevant_answer_parent_ge_0_2', 0.0),
                'answer_parent_relevance_max_jaccard': s.get('relevance', {}).get('answer_parent_relevance_max_jaccard', 0.0),
            },
        }

    def dump(self, path: str) -> None:
        obj = {
            'summary': self.summary(),
            'concise_summary': self.concise_summary(),
            'per_sample': self.samples,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)


def _edge_question_text(e: Any) -> str:
    parts = [
        getattr(e, 'src_name', ''),
        getattr(e, 'key', ''),
        getattr(e, 'value', ''),
        getattr(e, 'dst_name', ''),
        getattr(e, 'relation', ''),
        getattr(e, 'title', ''),
    ]
    return ' '.join([str(x) for x in parts if x])


def _jaccard(token_set_fn, a: str, b: str) -> float:
    A = token_set_fn(a)
    B = token_set_fn(b)
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A | B))


def _shortest_hops(topic_nodes: List[int], target_nodes: Set[int], out_eids_by_src: Dict[int, List[int]], kvedges: Dict[int, Any]) -> Optional[int]:
    if not topic_nodes or not target_nodes:
        return None
    q = deque()
    dist: Dict[int, int] = {}
    for t in topic_nodes:
        dist[t] = 0
        q.append(t)
    best = None
    while q:
        u = q.popleft()
        if u in target_nodes:
            best = dist[u]
            break
        for eid in out_eids_by_src.get(u, []):
            v = kvedges[eid].dst
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return best


def evaluate_retrieval_sample(
    sample: Dict[str, Any],
    question: str,
    answer: str,
    topic_nodes: List[int],
    kvedges: Dict[int, Any],
    kept_nodes: Set[int],
    kept_edges: Set[int],
    *,
    value_matches_answer_fn,
    token_set_fn,
) -> SampleMetric:
    sample_id = sample.get('id', sample.get('_id', sample.get('sample_id', None)))
    metric = SampleMetric(sample_id=sample_id, question=question, answer=answer)

    graph_answer_eids = {eid for eid, e in kvedges.items() if value_matches_answer_fn(getattr(e, 'value', ''), answer)}
    kept_answer_eids = {eid for eid in kept_edges if value_matches_answer_fn(getattr(kvedges[eid], 'value', ''), answer)}

    out_eids_by_src = defaultdict(list)
    in_eids_by_dst = defaultdict(list)
    outdeg = defaultdict(int)
    indeg = defaultdict(int)
    for eid in kept_edges:
        e = kvedges[eid]
        out_eids_by_src[e.src].append(eid)
        in_eids_by_dst[e.dst].append(eid)
        outdeg[e.src] += 1
        indeg[e.dst] += 1

    sinks = {n for n in kept_nodes if outdeg.get(n, 0) == 0}
    answer_nodes_all = {kvedges[eid].dst for eid in graph_answer_eids}
    answer_nodes_kept = {kvedges[eid].dst for eid in kept_answer_eids}
    answer_sink_nodes = {n for n in answer_nodes_kept if n in sinks}
    answer_non_sink_nodes = {n for n in answer_nodes_kept if n not in sinks}

    parent_eids_for_answer = set()
    for n in answer_nodes_kept:
        for peid in in_eids_by_dst.get(n, []):
            parent_eids_for_answer.add(peid)

    parent_rels = [_jaccard(token_set_fn, question, _edge_question_text(kvedges[eid])) for eid in sorted(parent_eids_for_answer)]
    answer_edge_rels = [_jaccard(token_set_fn, question, _edge_question_text(kvedges[eid])) for eid in sorted(kept_answer_eids)]

    min_hops_any_answer = _shortest_hops(topic_nodes, answer_nodes_kept, out_eids_by_src, kvedges)
    min_hops_sink_answer = _shortest_hops(topic_nodes, answer_sink_nodes, out_eids_by_src, kvedges)

    max_indeg_answer_sink = max((indeg.get(n, 0) for n in answer_sink_nodes), default=0)
    mean_indeg_answer_sink = _safe_div(sum(indeg.get(n, 0) for n in answer_sink_nodes), len(answer_sink_nodes))

    # 弱孤立：答案在 sink，且只有 <=1 条入边，且从 topic 到答案 hops <= 1 或不存在更长证据链
    weakly_isolated = bool(answer_sink_nodes) and max_indeg_answer_sink <= 1 and (min_hops_sink_answer is None or min_hops_sink_answer <= 1)

    metric.coverage = {
        'graph_has_answer': bool(graph_answer_eids),
        'subgraph_has_answer': bool(kept_answer_eids),
        'answer_in_sink': bool(answer_sink_nodes),
        'answer_in_non_sink': bool(answer_non_sink_nodes),
        'num_answer_edges_in_graph': len(graph_answer_eids),
        'num_answer_edges_in_subgraph': len(kept_answer_eids),
        'answer_edge_coverage_ratio': _safe_div(len(kept_answer_eids), len(graph_answer_eids)),
    }
    metric.structure = {
        'num_kept_nodes': len(kept_nodes),
        'num_kept_edges': len(kept_edges),
        'num_sinks': len(sinks),
        'num_answer_nodes_kept': len(answer_nodes_kept),
        'num_answer_sink_nodes': len(answer_sink_nodes),
        'answer_path_exists': min_hops_any_answer is not None,
        'answer_sink_path_exists': min_hops_sink_answer is not None,
        'answer_min_hops': -1 if min_hops_any_answer is None else min_hops_any_answer,
        'answer_sink_min_hops': -1 if min_hops_sink_answer is None else min_hops_sink_answer,
        'answer_sink_max_indegree': max_indeg_answer_sink,
        'answer_sink_mean_indegree': mean_indeg_answer_sink,
        'supported_answer_sink_ge1': bool(answer_sink_nodes) and max_indeg_answer_sink >= 1,
        'supported_answer_sink_ge2': bool(answer_sink_nodes) and max_indeg_answer_sink >= 2,
        'weakly_isolated_answer_sink': weakly_isolated,
    }
    metric.relevance = {
        'answer_parent_relevance_max_jaccard': max(parent_rels) if parent_rels else 0.0,
        'answer_parent_relevance_mean_jaccard': _safe_div(sum(parent_rels), len(parent_rels)),
        'answer_edge_relevance_max_jaccard': max(answer_edge_rels) if answer_edge_rels else 0.0,
        'answer_edge_relevance_mean_jaccard': _safe_div(sum(answer_edge_rels), len(answer_edge_rels)),
        'relevant_answer_parent_ge_0_1': (max(parent_rels) >= 0.1) if parent_rels else False,
        'relevant_answer_parent_ge_0_2': (max(parent_rels) >= 0.2) if parent_rels else False,
    }
    return metric
