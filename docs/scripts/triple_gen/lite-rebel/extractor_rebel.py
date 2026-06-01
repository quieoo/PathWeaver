from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import sys
import unicodedata


_SIBLING_NO_LLM_DIR = Path(__file__).resolve().parent.parent / "triple_gen_no_llm"
if str(_SIBLING_NO_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIBLING_NO_LLM_DIR))

from extractor import (  # type: ignore
    ExtractionConfig,
    NoLLMTripleExtractor,
    TripleCandidate,
    bfs_relation_path,
    canonicalize_entity,
    collect_candidate_repair_sentences,
    collect_pages,
    dedupe_stage_graph,
    find_answer_nodes,
    get_triple_type,
    get_supporting_titles,
    graph_entity_matches,
    graph_value_matches,
    infer_question_anchors,
    is_literal_value,
    norm_text,
    normalize_relation_phrase,
    safe_sample_id,
    simple_sentence_tokenize,
)


def extract_triplets_from_rebel_text(text: str) -> List[Dict[str, str]]:
    triplets: List[Dict[str, str]] = []
    relation = ""
    subject = ""
    object_ = ""
    current = "x"
    clean_text = (
        text.replace("<s>", " ")
        .replace("</s>", " ")
        .replace("<pad>", " ")
        .strip()
    )
    for token in clean_text.split():
        if token == "<triplet>":
            current = "t"
            if subject and relation and object_:
                triplets.append(
                    {
                        "head": subject.strip(),
                        "relation": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
            relation = ""
            subject = ""
            object_ = ""
        elif token == "<subj>":
            current = "s"
            if subject and relation and object_:
                triplets.append(
                    {
                        "head": subject.strip(),
                        "relation": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
    if subject and relation and object_:
        triplets.append(
            {
                "head": subject.strip(),
                "relation": relation.strip(),
                "tail": object_.strip(),
            }
        )
    return triplets


def drop_leading_article(text: str) -> str:
    lowered = norm_text(text)
    for prefix in ("the ", "a ", "an "):
        if lowered.lower().startswith(prefix):
            return norm_text(lowered[len(prefix):])
    return lowered


def looks_like_pronoun(text: str) -> bool:
    return norm_text(text).lower() in {
        "he",
        "she",
        "it",
        "they",
        "them",
        "this",
        "that",
        "these",
        "those",
        "his",
        "her",
        "its",
        "their",
    }


_ROLE_PREFIX_RE = re.compile(
    r"^(?:actress|actor|director|writer|producer|novelist|poet|composer|king|queen|prince|princess|duke|duchess|count|countess)\s+",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def alias_key(text: str) -> str:
    s = canonicalize_entity(text)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = s.replace("&", " and ")
    s = _NON_ALNUM_RE.sub(" ", s)
    return " ".join(s.split())


def strip_role_prefix(text: str) -> str:
    return canonicalize_entity(_ROLE_PREFIX_RE.sub("", norm_text(text)))


def likely_person_name(text: str) -> bool:
    s = strip_role_prefix(text)
    if not s:
        return False
    parts = [p for p in s.split() if p]
    if len(parts) < 2:
        return False
    return sum(part[:1].isupper() for part in parts) >= 2


def relation_hint_set(question: str) -> Set[str]:
    q = norm_text(question).lower()
    hints: Set[str] = set()
    if "director" in q or "directed" in q:
        hints.add("director")
    if "mother" in q:
        hints.update({"mother", "parent", "child"})
    if "father" in q or "paternal" in q:
        hints.update({"father", "parent", "child"})
    if "maternal" in q:
        hints.update({"mother", "parent", "child"})
    if "grandfather" in q or "grandmother" in q:
        hints.update({"father", "mother", "parent", "child"})
    if "born" in q or "where" in q:
        hints.update({"place of birth", "birth place"})
    if "die" in q or "death" in q:
        hints.update({"date of death", "death date", "place of death"})
    if "when" in q:
        hints.update({"date", "publication date", "date of death", "date of birth"})
    if "award" in q or "won" in q:
        hints.add("award received")
    if "country" in q:
        hints.add("country")
    return hints


def alias_match(left: str, right: str) -> bool:
    a = alias_key(left)
    b = alias_key(right)
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 8 and len(b) >= 8 and (a in b or b in a):
        return True
    return False


def resolve_graph_entity(name: str, graph_entities: Sequence[str]) -> str:
    canonical = canonicalize_entity(name)
    if not canonical:
        return ""
    for ent in graph_entities:
        if canonical.lower() == canonicalize_entity(ent).lower():
            return canonicalize_entity(ent)
    for ent in graph_entities:
        if alias_match(canonical, ent):
            return canonicalize_entity(ent)
    return canonical


def find_alias_entities_in_graph(stage_graph: Dict[str, Any], mention: str) -> Set[str]:
    out: Set[str] = set()
    for ent in stage_graph.get("entity_list", []) or []:
        if alias_match(ent, mention):
            out.add(canonicalize_entity(ent))
    return out


def infer_question_anchors_rebel(
    sample: Dict[str, Any],
    stage_graph: Dict[str, Any],
    extractor: NoLLMTripleExtractor,
) -> List[str]:
    base = infer_question_anchors(sample, stage_graph, extractor)
    if base:
        return sorted(set(base))

    question = norm_text(sample.get("question", ""))
    if not question:
        return []

    graph_entities = [canonicalize_entity(ent) for ent in stage_graph.get("entity_list", []) or []]
    question_entities = extractor.extract_entity_candidates(question, title="")
    matched: Set[str] = set()

    for q_ent in question_entities:
        matched.update(find_alias_entities_in_graph(stage_graph, q_ent))

    question_key = alias_key(question)
    for ent in graph_entities:
        ent_key = alias_key(ent)
        if ent_key and ent_key in question_key:
            matched.add(ent)

    for title in get_supporting_titles(sample):
        matched.update(find_alias_entities_in_graph(stage_graph, title))

    return sorted(matched)


def find_answer_nodes_rebel(stage_graph: Dict[str, Any], answer: str) -> Tuple[Set[str], List[Dict[str, Any]]]:
    answer_entities, answer_attributes = find_answer_nodes(stage_graph, answer)
    if answer_entities or answer_attributes:
        return answer_entities, answer_attributes

    fuzzy_entities: Set[str] = set()
    fuzzy_attributes: List[Dict[str, Any]] = []
    for ent in stage_graph.get("entity_list", []) or []:
        if alias_match(ent, answer):
            fuzzy_entities.add(canonicalize_entity(ent))
    for tri in stage_graph.get("triples", []) or []:
        tri_type = get_triple_type(tri)
        if tri_type == "RELATION" and alias_match(tri.get("tail", ""), answer):
            fuzzy_entities.add(canonicalize_entity(tri.get("tail", "")))
        if tri_type == "ATTRIBUTE" and graph_value_matches(tri.get("tail", ""), answer):
            fuzzy_attributes.append(tri)
    return fuzzy_entities, fuzzy_attributes


def bfs_relation_path_rebel(
    stage_graph: Dict[str, Any],
    anchors: Sequence[str],
    targets: Set[str],
    max_hops: int = 4,
) -> List[Dict[str, Any]]:
    expanded_anchors: Set[str] = set()
    expanded_targets: Set[str] = set(targets)

    for anchor in anchors:
        expanded_anchors.update(find_alias_entities_in_graph(stage_graph, anchor))
    for target in list(targets):
        expanded_targets.update(find_alias_entities_in_graph(stage_graph, target))

    return bfs_relation_path(
        stage_graph,
        sorted(expanded_anchors or set(anchors)),
        expanded_targets,
        max_hops=max_hops,
    )


def normalize_relation_for_rebel_stage2(relation: str, question_hints: Set[str]) -> str:
    rel = normalize_relation_phrase(relation)
    if not rel:
        return ""
    if rel in {"born in", "birth place"}:
        return "place of birth"
    if rel in {"died in"}:
        return "place of death"
    if rel in {"died", "death date"}:
        return "date of death"
    if rel in {"born", "birth date"}:
        return "date of birth"
    if rel in {"won", "awarded with"}:
        return "award received"
    if rel in {"directed by", "directed"}:
        return "director"
    if rel == "is the son of" or rel == "son of":
        return "father" if "father" in question_hints else "parent"
    if rel == "is the daughter of" or rel == "daughter of":
        return "mother" if "mother" in question_hints else "parent"
    return rel


def extract_rebel_stage2_regex_triples(
    sentence: str,
    *,
    page_title: str,
    question_hints: Set[str],
    graph_entities: Sequence[str],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    text = norm_text(sentence)
    title = resolve_graph_entity(page_title, graph_entities) if page_title else ""

    # Directed-by patterns are a reliable bridge for film -> director questions.
    directed_patterns = [
        re.search(r"^(?P<head>.+?) is a .*? film directed by (?P<tail>[^,.]+?)(?:\s+based on|[.,;]|$)", text, re.IGNORECASE),
        re.search(r"^(?P<head>.+?) directed by (?P<tail>[^,.]+?)(?:\s+based on|[.,;]|$)", text, re.IGNORECASE),
        re.search(r"^(?P<head>.+?) was written and directed by (?P<tail>[^,(.]+?)(?:[.,;]|$)", text, re.IGNORECASE),
    ]
    for match in directed_patterns:
        if not match:
            continue
        raw_head = title or match.group("head")
        head = resolve_graph_entity(raw_head, graph_entities)
        tail = resolve_graph_entity(strip_role_prefix(match.group("tail")), graph_entities)
        if head and tail and head != tail:
            out.append({"head": head, "relation": "director", "tail": tail, "triple_type": "RELATION"})

    # Kinship patterns help connect answer pages to anchor entities.
    kin_match = re.search(
        r"^(?P<child>.+?)\s+(?:is|was)\s+(?:the\s+)?(?:[^,.]*?\s+)?(?P<role>son|daughter)\s+of\s+(?P<parents>.+?)(?:[.;]|$)",
        text,
        re.IGNORECASE,
    )
    if kin_match:
        raw_child = kin_match.group("child")
        if looks_like_pronoun(raw_child) and title:
            raw_child = title
        elif title and len(raw_child.split()) == 1 and title.split() and raw_child.lower() == title.split()[0].lower():
            raw_child = title
        child = resolve_graph_entity(raw_child, graph_entities)
        parents_text = kin_match.group("parents")
        for raw_parent in re.split(r"\s+and\s+|,\s*", parents_text):
            parent = resolve_graph_entity(strip_role_prefix(raw_parent), graph_entities)
            if not child or not parent or child == parent:
                continue
            relation = "parent"
            lowered_parent = raw_parent.lower()
            if "mother" in question_hints and any(tok in lowered_parent for tok in {"actress", "mother", "queen", "princess", "duchess"}):
                relation = "mother"
            elif "father" in question_hints and any(tok in lowered_parent for tok in {"director", "father", "king", "prince", "duke", "count"}):
                relation = "father"
            out.append({"head": child, "relation": relation, "tail": parent, "triple_type": "RELATION"})

    # Parent-of wording is common in biographies.
    parent_match = re.search(
        r"^(?P<head>.+?)\s+(?:is|was)\s+the\s+(?P<rel>mother|father)\s+of\s+(?P<tail>[^,.]+)",
        text,
        re.IGNORECASE,
    )
    if parent_match:
        raw_head = parent_match.group("head")
        if looks_like_pronoun(raw_head) and title:
            raw_head = title
        elif title and len(raw_head.split()) == 1 and title.split() and raw_head.lower() == title.split()[0].lower():
            raw_head = title
        head = resolve_graph_entity(strip_role_prefix(raw_head), graph_entities)
        tail = resolve_graph_entity(strip_role_prefix(parent_match.group("tail")), graph_entities)
        rel = normalize_relation_phrase(parent_match.group("rel"))
        if head and tail and head != tail:
            out.append({"head": head, "relation": rel, "tail": tail, "triple_type": "RELATION"})

    if title:
        death_match = re.search(r"\bdied\b[^,;()]*[, ](?P<tail>\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},\s*\d{4}|\d{4})", text, re.IGNORECASE)
        if death_match:
            out.append({"head": title, "relation": "date of death", "tail": death_match.group("tail"), "triple_type": "ATTRIBUTE"})
        birth_match = re.search(r"\bborn\b[^,;()]*[, ](?P<tail>\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},\s*\d{4}|\d{4})", text, re.IGNORECASE)
        if birth_match:
            out.append({"head": title, "relation": "date of birth", "tail": birth_match.group("tail"), "triple_type": "ATTRIBUTE"})
        award_match = re.search(r"\bwon\b\s+(?P<tail>[^,.]+)", text, re.IGNORECASE)
        if award_match and "award received" in question_hints:
            out.append({"head": title, "relation": "award received", "tail": canonicalize_entity(award_match.group("tail")), "triple_type": "RELATION"})

    return out


def candidate_relevant_to_question(
    candidate: Dict[str, str],
    *,
    anchors: Sequence[str],
    answer: str,
    question_hints: Set[str],
) -> bool:
    head = candidate.get("head", "")
    relation = normalize_relation_phrase(candidate.get("relation", ""))
    tail = candidate.get("tail", "")
    if question_hints and relation in question_hints:
        return True
    if any(alias_match(head, anchor) or alias_match(tail, anchor) for anchor in anchors):
        return True
    if graph_entity_matches(tail, answer) or graph_value_matches(tail, answer) or alias_match(head, answer):
        return True
    return False


def add_candidate_to_graph(
    revised: Dict[str, Any],
    candidate: Dict[str, str],
    *,
    note_prefix: str,
) -> bool:
    head = canonicalize_entity(candidate.get("head", ""))
    relation = normalize_relation_phrase(candidate.get("relation", ""))
    tail = norm_text(candidate.get("tail", "")).strip(" .")
    triple_type = norm_text(candidate.get("triple_type", "")).upper() or ("ATTRIBUTE" if is_literal_value(tail) else "RELATION")
    if triple_type not in {"ATTRIBUTE", "RELATION"}:
        triple_type = "ATTRIBUTE"
    if not head or not relation or not tail:
        return False
    if triple_type == "RELATION":
        tail = canonicalize_entity(tail)
        if not tail or head == tail:
            return False
    sig = (head, relation, tail, triple_type)
    existing = {
        (
            canonicalize_entity(x.get("head", "")),
            normalize_relation_phrase(x.get("relation", "")),
            norm_text(x.get("tail", "")).strip(" ."),
            get_triple_type(x),
        )
        for x in revised.get("triples", []) or []
        if isinstance(x, dict)
    }
    if sig in existing:
        return False
    revised.setdefault("triples", []).append(
        {"head": head, "relation": relation, "tail": tail, "triple_type": triple_type}
    )
    if head not in revised.setdefault("entity_list", []):
        revised["entity_list"].append(head)
    if triple_type == "RELATION" and tail not in revised["entity_list"]:
        revised["entity_list"].append(tail)
    revised.setdefault("revision_notes", []).append(f"{note_prefix}: {head} | {relation} | {tail}")
    return True


def heuristic_graph_revision_rebel(
    sample: Dict[str, Any],
    stage1: Dict[str, Any],
    extractor: "RebelTripleExtractor",
    *,
    max_hops: int = 4,
) -> Dict[str, Any]:
    question = norm_text(sample.get("question", ""))
    answer = norm_text(sample.get("answer", ""))
    revised = {
        "_id": safe_sample_id(stage1, 0),
        "answer_sufficient": False,
        "missing_links": [],
        "revision_notes": [],
        "entity_list": copy.deepcopy(stage1.get("entity_list", []) or []),
        "triples": copy.deepcopy(stage1.get("triples", []) or []),
    }
    if not question or not answer:
        revised["missing_links"] = ["missing question or answer"]
        return revised

    revised = dedupe_stage_graph(revised)
    question_hints = relation_hint_set(question)
    anchors = infer_question_anchors_rebel(sample, revised, extractor.repair_extractor)
    answer_entities, answer_attributes = find_answer_nodes_rebel(revised, answer)
    attr_supported = any(
        any(alias_match(canonicalize_entity(tri.get("head", "")), anchor) for anchor in anchors)
        for tri in answer_attributes
    )
    relation_path = bfs_relation_path_rebel(revised, anchors, answer_entities, max_hops=max_hops)
    if attr_supported or answer_entities & set(anchors) or relation_path:
        revised["answer_sufficient"] = True
        return revised

    repair_sentences = collect_candidate_repair_sentences(sample, answer, anchors)
    graph_entities = list(revised.get("entity_list", []) or [])
    added = 0

    for title, sentence in repair_sentences:
        known_entities = set(graph_entities)
        known_entities.update(extractor.extract_entity_candidates(sentence, title=title))

        regex_candidates = extract_rebel_stage2_regex_triples(
            sentence,
            page_title=title,
            question_hints=question_hints,
            graph_entities=graph_entities,
        )
        for candidate in regex_candidates:
            if not candidate_relevant_to_question(candidate, anchors=anchors, answer=answer, question_hints=question_hints):
                continue
            if add_candidate_to_graph(revised, candidate, note_prefix="added regex bridge"):
                added += 1

        heuristic_candidates = []
        for tri in extractor.repair_extractor.extract_sentence_triples(
            sentence,
            page_title=title,
            known_entities=known_entities,
            last_subject=title,
        ):
            relation = normalize_relation_for_rebel_stage2(tri.relation, question_hints)
            heuristic_candidates.append(
                {
                    "head": resolve_graph_entity(tri.head, graph_entities),
                    "relation": relation,
                    "tail": resolve_graph_entity(tri.tail, graph_entities) if (tri.triple_type or "").upper() == "RELATION" else tri.tail,
                    "triple_type": tri.triple_type or ("ATTRIBUTE" if is_literal_value(tri.tail) else "RELATION"),
                }
            )
        for candidate in heuristic_candidates:
            if not candidate_relevant_to_question(candidate, anchors=anchors, answer=answer, question_hints=question_hints):
                continue
            if add_candidate_to_graph(revised, candidate, note_prefix="added heuristic bridge"):
                added += 1

    if not added and anchors and answer_attributes:
        for tri in answer_attributes:
            head = canonicalize_entity(tri.get("head", ""))
            if not head or any(alias_match(head, anchor) for anchor in anchors):
                continue
            bridge = {"head": anchors[0], "relation": "related to", "tail": head, "triple_type": "RELATION"}
            if add_candidate_to_graph(revised, bridge, note_prefix="added entity bridge"):
                added += 1
                break

    revision_notes = list(revised.get("revision_notes", []) or [])
    revised = dedupe_stage_graph(revised)
    revised["revision_notes"] = revision_notes

    anchors = infer_question_anchors_rebel(sample, revised, extractor.repair_extractor)
    answer_entities, answer_attributes = find_answer_nodes_rebel(revised, answer)
    attr_supported = any(
        any(alias_match(canonicalize_entity(tri.get("head", "")), anchor) for anchor in anchors)
        for tri in answer_attributes
    )
    relation_path = bfs_relation_path_rebel(revised, anchors, answer_entities, max_hops=max_hops)
    revised["answer_sufficient"] = bool(attr_supported or answer_entities & set(anchors) or relation_path)
    if revised["answer_sufficient"]:
        revised["missing_links"] = []
    else:
        missing: List[str] = []
        if not anchors:
            missing.append("missing question anchor in graph")
        if not answer_entities and not answer_attributes:
            missing.append("missing answer grounding in graph")
        if anchors and (answer_entities or answer_attributes) and not relation_path and not attr_supported:
            missing.append("missing explicit bridge chain from question anchor to answer")
        revised["missing_links"] = missing or ["missing explicit bridge chain from question anchor to answer"]
    return revised


@dataclass
class RebelConfig:
    model_name: str = "Babelscape/rebel-large"
    supporting_pages_only: bool = True
    include_question_entities: bool = False
    batch_size: int = 8
    max_input_length: int = 256
    max_new_tokens: int = 192
    num_beams: int = 3
    max_triples_per_sentence: int = 16
    sample_batch_size: int = 4
    hf_cache_dir: Optional[str] = None
    device: str = "auto"
    torch_dtype: str = "auto"


@dataclass
class RebelRuntimeStats:
    inference_calls: int = 0
    inference_sentences: int = 0
    inference_seconds: float = 0.0
    samples_built: int = 0
    triples_built: int = 0


class RebelTripleExtractor:
    def __init__(self, config: RebelConfig):
        self.config = config
        self._repair_extractor = NoLLMTripleExtractor(
            ExtractionConfig(
                supporting_pages_only=config.supporting_pages_only,
                include_question_entities=config.include_question_entities,
                use_spacy=False,
                max_triples_per_sentence=config.max_triples_per_sentence,
            )
        )
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = None
        self._dtype = None
        self._stats = RebelRuntimeStats()

    @property
    def repair_extractor(self) -> NoLLMTripleExtractor:
        return self._repair_extractor

    def get_runtime_stats(self) -> Dict[str, Any]:
        stats = copy.deepcopy(self._stats.__dict__)
        stats["device"] = str(self._device) if self._device is not None else None
        stats["torch_dtype"] = str(self._dtype) if self._dtype is not None else None
        if self._stats.inference_seconds > 0:
            stats["sentences_per_second"] = self._stats.inference_sentences / self._stats.inference_seconds
        else:
            stats["sentences_per_second"] = 0.0
        return stats

    def _resolve_torch_dtype(self, torch: Any, device: str) -> Any:
        dtype_name = norm_text(self.config.torch_dtype).lower() or "auto"
        if device == "cpu":
            return torch.float32
        if dtype_name == "auto":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported torch_dtype={self.config.torch_dtype}")
        return mapping[dtype_name]

    def _lazy_load_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "REBEL requires PyTorch. Install it first, for example: pip install torch"
            ) from exc

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "REBEL requires transformers. Install it first, for example: pip install transformers sentencepiece"
            ) from exc

        self._torch = torch
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
        self._dtype = self._resolve_torch_dtype(torch, device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.hf_cache_dir,
            use_fast=True,
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.hf_cache_dir,
            torch_dtype=self._dtype,
        )

        self._device = torch.device(device)
        self._model.to(self._device)
        self._model.eval()

    def build_graph(self, sample: Dict[str, Any], idx: int = 0) -> Dict[str, Any]:
        graphs = self.build_graphs([(sample, idx)])
        return graphs[safe_sample_id(sample, idx)]

    def _collect_sample_sentence_records(
        self,
        sample: Dict[str, Any],
        idx: int,
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, str]]]:
        sample_id = safe_sample_id(sample, idx)
        pages = collect_pages(sample, self.config.supporting_pages_only)
        question = norm_text(sample.get("question"))
        supporting_titles = set(get_supporting_titles(sample))
        meta = {
            "question": question,
            "supporting_titles": supporting_titles,
            "include_question_entities": self.config.include_question_entities,
        }
        records: List[Dict[str, str]] = []

        for page in pages:
            title = canonicalize_entity(page.get("title", ""))
            sentences: List[str] = []
            for raw_sentence in (page.get("sentences", []) or []):
                text = norm_text(raw_sentence)
                if not text:
                    continue
                sub_sentences = simple_sentence_tokenize(text)
                if sub_sentences:
                    sentences.extend(sub_sentences)
                else:
                    sentences.append(text)
            for sentence in sentences:
                records.append(
                    {
                        "sample_id": sample_id,
                        "page_title": title,
                        "sentence": sentence,
                    }
                )
        return sample_id, meta, records

    def build_graphs(
        self,
        sample_items: Sequence[Tuple[Dict[str, Any], int]],
    ) -> Dict[str, Dict[str, Any]]:
        sample_meta: Dict[str, Dict[str, Any]] = {}
        entity_candidates_by_id: Dict[str, Set[str]] = {}
        triple_candidates_by_id: Dict[str, List[TripleCandidate]] = {}
        sentence_records: List[Dict[str, str]] = []

        for sample, idx in sample_items:
            sample_id, meta, records = self._collect_sample_sentence_records(sample, idx)
            sample_meta[sample_id] = meta
            entity_candidates_by_id[sample_id] = set()
            triple_candidates_by_id[sample_id] = []
            for page in collect_pages(sample, self.config.supporting_pages_only):
                title = canonicalize_entity(page.get("title", ""))
                if title:
                    entity_candidates_by_id[sample_id].add(title)
            sentence_records.extend(records)

        batch_size = max(1, self.config.batch_size)
        for batch_start in range(0, len(sentence_records), batch_size):
            record_batch = sentence_records[batch_start : batch_start + batch_size]
            sentence_batch = [record["sentence"] for record in record_batch]
            decoded_batch = self.generate_rebel_outputs(sentence_batch)
            for record, decoded in zip(record_batch, decoded_batch):
                sample_id = record["sample_id"]
                title = record["page_title"]
                clause_entities = self.extract_entity_candidates(record["sentence"], title=title)
                entity_candidates_by_id[sample_id].update(clause_entities)
                supporting_titles = sample_meta[sample_id]["supporting_titles"]
                for tri in self.parse_rebel_sentence_output(
                    decoded,
                    page_title=title,
                    supporting_titles=supporting_titles,
                    known_entities=entity_candidates_by_id[sample_id] | clause_entities,
                )[: self.config.max_triples_per_sentence]:
                    triple_candidates_by_id[sample_id].append(tri)
                    entity_candidates_by_id[sample_id].add(tri.head)
                    if tri.triple_type == "RELATION":
                        entity_candidates_by_id[sample_id].add(canonicalize_entity(tri.tail))

        graphs: Dict[str, Dict[str, Any]] = {}
        for sample, idx in sample_items:
            sample_id = safe_sample_id(sample, idx)
            question = sample_meta[sample_id]["question"]
            if sample_meta[sample_id]["include_question_entities"] and question:
                entity_candidates_by_id[sample_id].update(self.lightweight_question_entities(question))
            graph = self.normalize_graph(
                sample_id,
                entity_candidates_by_id[sample_id],
                triple_candidates_by_id[sample_id],
            )
            graph = dedupe_stage_graph(graph)
            graphs[sample_id] = graph
            self._stats.samples_built += 1
            self._stats.triples_built += len(graph.get("triples", []) or [])
        return graphs

    def lightweight_question_entities(self, question: str) -> Set[str]:
        return self._repair_extractor.extract_entity_candidates(question, title="")

    def extract_entity_candidates(self, text: str, title: str = "") -> Set[str]:
        entities = self._repair_extractor.extract_entity_candidates(text, title=title)
        if title:
            entities.add(canonicalize_entity(title))
        return {canonicalize_entity(x) for x in entities if canonicalize_entity(x)}

    def extract_sentence_triples(
        self,
        sentence: str,
        *,
        page_title: str,
        known_entities: Set[str],
        last_subject: str,
    ) -> List[TripleCandidate]:
        if not norm_text(sentence):
            return []
        decoded = self.generate_rebel_outputs([sentence])[0]
        return self.parse_rebel_sentence_output(
            decoded,
            page_title=page_title or last_subject,
            supporting_titles=set(),
            known_entities=known_entities,
        )

    def generate_rebel_outputs(self, sentences: Sequence[str]) -> List[str]:
        normalized = [norm_text(s) for s in sentences if norm_text(s)]
        if not normalized:
            return []

        self._lazy_load_model()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        started_at = time.perf_counter()
        encoded = self._tokenizer(
            list(normalized),
            max_length=max(8, self.config.max_input_length),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=max(16, self.config.max_new_tokens),
                num_beams=max(1, self.config.num_beams),
            )
        elapsed = time.perf_counter() - started_at
        self._stats.inference_calls += 1
        self._stats.inference_sentences += len(normalized)
        self._stats.inference_seconds += elapsed
        return self._tokenizer.batch_decode(generated, skip_special_tokens=False)

    def parse_rebel_sentence_output(
        self,
        decoded_text: str,
        *,
        page_title: str,
        supporting_titles: Set[str],
        known_entities: Set[str],
    ) -> List[TripleCandidate]:
        triples: List[TripleCandidate] = []
        for raw in extract_triplets_from_rebel_text(decoded_text):
            head = canonicalize_entity(raw.get("head", ""))
            relation = normalize_relation_phrase(raw.get("relation", ""))
            tail = norm_text(raw.get("tail", "")).strip(" .")
            if not head or not relation or not tail:
                continue

            if looks_like_pronoun(head) and page_title:
                head = page_title
            elif page_title:
                head_cmp = drop_leading_article(head).lower()
                title_cmp = drop_leading_article(page_title).lower()
                if head_cmp and title_cmp and (head_cmp in title_cmp or title_cmp in head_cmp):
                    head = page_title

            if looks_like_pronoun(tail) and page_title:
                tail = page_title

            triple_type = self.classify_rebel_triple(
                head=head,
                relation=relation,
                tail=tail,
                supporting_titles=supporting_titles,
                known_entities=known_entities,
            )
            triples.append(
                TripleCandidate(
                    head=head,
                    relation=relation,
                    tail=tail,
                    triple_type=triple_type,
                    source="rebel",
                )
            )
        return triples

    def classify_rebel_triple(
        self,
        *,
        head: str,
        relation: str,
        tail: str,
        supporting_titles: Set[str],
        known_entities: Set[str],
    ) -> str:
        if is_literal_value(tail):
            return "ATTRIBUTE"
        if tail in supporting_titles:
            return "RELATION"
        canonical_tail = canonicalize_entity(tail)
        if canonical_tail in known_entities:
            return "RELATION"
        if canonical_tail[:1].isupper():
            return "RELATION"
        if relation in {"instance of", "subclass of", "part of", "located in", "country", "parent taxon"}:
            return "RELATION"
        return "ATTRIBUTE"

    def normalize_graph(
        self,
        sample_id: str,
        entity_candidates: Set[str],
        triple_candidates: Sequence[TripleCandidate],
    ) -> Dict[str, Any]:
        entity_seen: Set[str] = set()
        entities: List[str] = []
        for ent in sorted(entity_candidates):
            name = canonicalize_entity(ent)
            if not name:
                continue
            key = name.lower()
            if key in entity_seen:
                continue
            entity_seen.add(key)
            entities.append(name)

        triples: List[Dict[str, str]] = []
        triple_seen: Set[Tuple[str, str, str, str]] = set()
        entity_index = {x.lower(): x for x in entities}

        for tri in triple_candidates:
            head = canonicalize_entity(tri.head)
            relation = normalize_relation_phrase(tri.relation)
            tail = norm_text(tri.tail).strip(" .")
            triple_type = norm_text(tri.triple_type).upper() or "ATTRIBUTE"
            if triple_type not in {"ATTRIBUTE", "RELATION"}:
                triple_type = "ATTRIBUTE"
            if not head or not relation or not tail:
                continue

            head = entity_index.get(head.lower(), head)
            if head.lower() not in entity_index:
                entity_index[head.lower()] = head
                entities.append(head)

            if triple_type == "RELATION":
                tail = canonicalize_entity(tail)
                if not tail or head.lower() == tail.lower():
                    continue
                tail = entity_index.get(tail.lower(), tail)
                if tail.lower() not in entity_index:
                    entity_index[tail.lower()] = tail
                    entities.append(tail)

            sig = (head, relation, tail, triple_type)
            if sig in triple_seen:
                continue
            triple_seen.add(sig)
            triples.append(
                {
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "triple_type": triple_type,
                }
            )

        return {
            "_id": sample_id,
            "entity_list": entities,
            "triples": triples,
        }
