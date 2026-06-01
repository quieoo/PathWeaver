from __future__ import annotations

import copy
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import spacy
except ImportError:
    spacy = None


_SPACE_RE = re.compile(r"\s+")
_TITLE_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9&'().-]+){0,5})\b")
_YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b")
_NUMERIC_RE = re.compile(r"^-?\d+(?:[.,]\d+)?(?:\s*(?:%|km|m|cm|mm|kg|lb|million|billion))?$", re.IGNORECASE)
_DATEISH_RE = re.compile(
    r"\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"(?:\s+\d{1,2})?(?:,\s*\d{4}|\s+\d{4})?\b",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(r"^(?:E|T)\d+$", re.IGNORECASE)
_ATTRIBUTE_HINT_RE = re.compile(
    r"\b(population|area|length|height|weight|runtime|duration|language|genre|type|kind|birth|death|released|"
    r"publication|published|founded|formed|capital|currency|president|leader|spouse|occupation|nationality)\b",
    re.IGNORECASE,
)
_COPULA_RE = re.compile(r"\b(?:is|was|were|are)\b", re.IGNORECASE)
_PREP_RE = re.compile(r"\b(?:in|on|at|from|by|near|inside|outside|under|over|after|before)\b", re.IGNORECASE)

_STOP_ENTITY_WORDS = {
    "A", "An", "The", "This", "That", "These", "Those", "He", "She", "It", "They",
    "His", "Her", "Its", "Their", "Who", "What", "When", "Where", "Which",
}

_GENERIC_VALUE_WORDS = {
    "male", "female", "annual", "weekly", "monthly", "daily", "american", "british",
    "english", "french", "german", "canadian",
}


def norm_text(x: Any) -> str:
    if x is None:
        return ""
    return _SPACE_RE.sub(" ", str(x)).strip()


def normalize_relation_phrase(text: Any) -> str:
    s = norm_text(text).lower().replace("_", " ").replace("-", " ")
    s = _SPACE_RE.sub(" ", s).strip(" ,.;:")
    return s


def extract_sample_id(sample: Dict[str, Any]) -> str:
    return norm_text(sample.get("_id")) or norm_text(sample.get("id"))


def safe_sample_id(sample: Dict[str, Any], idx: int) -> str:
    return extract_sample_id(sample) or f"sample_{idx:08d}"


def is_placeholder_entity_code(text: Any) -> bool:
    return bool(_PLACEHOLDER_RE.fullmatch(norm_text(text)))


def is_literal_value(text: Any) -> bool:
    s = norm_text(text)
    if not s:
        return False
    if _NUMERIC_RE.fullmatch(s):
        return True
    if _YEAR_RE.fullmatch(s):
        return True
    if _DATEISH_RE.search(s):
        return True
    if s.lower() in _GENERIC_VALUE_WORDS:
        return True
    if len(s.split()) <= 3 and s.islower():
        return True
    return False


def canonicalize_entity(text: Any) -> str:
    s = norm_text(text)
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    s = s.strip(" ,.;:")
    return s


def normalize_context_entry(entry: Any) -> Optional[Dict[str, Any]]:
    if isinstance(entry, dict):
        raw_sentences = entry.get("sentences")
        if isinstance(raw_sentences, list):
            sentences = [norm_text(s) for s in raw_sentences if norm_text(s)]
        else:
            text = entry.get("paragraph_text") or entry.get("text") or entry.get("context")
            if isinstance(text, list):
                sentences = [norm_text(s) for s in text if norm_text(s)]
            elif isinstance(text, str):
                sentences = [norm_text(text)] if norm_text(text) else []
            else:
                sentences = []
        return {
            "title": norm_text(entry.get("title") or entry.get("heading") or entry.get("entity") or ""),
            "sentences": sentences,
            "is_supporting": bool(entry.get("is_supporting", False)),
        }

    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        title = norm_text(entry[0])
        raw_sentences = entry[1] if isinstance(entry[1], list) else []
        return {
            "title": title,
            "sentences": [norm_text(s) for s in raw_sentences if norm_text(s)],
            "is_supporting": False,
        }
    return None


def iter_sample_context_entries(sample: Dict[str, Any]) -> Iterable[Any]:
    raw_context = sample.get("context")
    if isinstance(raw_context, list):
        return raw_context
    raw_paragraphs = sample.get("paragraphs")
    if isinstance(raw_paragraphs, list):
        return raw_paragraphs
    return []


def get_supporting_titles(sample: Dict[str, Any]) -> List[str]:
    titles: List[str] = []
    seen: Set[str] = set()
    for item in sample.get("supporting_facts", []) or []:
        if not isinstance(item, (list, tuple)) or not item:
            continue
        title = norm_text(item[0])
        if title and title not in seen:
            titles.append(title)
            seen.add(title)
    if titles:
        return titles

    for raw_para in iter_sample_context_entries(sample):
        para = normalize_context_entry(raw_para)
        if para is None or not para.get("is_supporting"):
            continue
        title = para["title"]
        if title and title not in seen:
            titles.append(title)
            seen.add(title)
    return titles


def collect_pages(sample: Dict[str, Any], supporting_pages_only: bool) -> List[Dict[str, Any]]:
    supporting_titles = set(get_supporting_titles(sample))
    pages: List[Dict[str, Any]] = []
    for raw_para in iter_sample_context_entries(sample):
        para = normalize_context_entry(raw_para)
        if para is None:
            continue
        if supporting_pages_only and supporting_titles and para["title"] not in supporting_titles:
            continue
        pages.append(para)
    return pages


def split_sentence_clauses(sentence: str) -> List[str]:
    chunks = re.split(r"[;:]\s+|\s+and\s+|\s+but\s+", sentence)
    return [norm_text(chunk) for chunk in chunks if norm_text(chunk)]


def simple_sentence_tokenize(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", norm_text(text))
    return [p for p in parts if p]


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        return obj["data"]
    raise ValueError("Unsupported JSON root format. Expect list or {data:[...]}.")


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_output_ids(path: str) -> Set[str]:
    out: Set[str] = set()
    if not path or not Path(path).exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = extract_sample_id(obj) if isinstance(obj, dict) else ""
            if sample_id:
                out.add(sample_id)
    return out


@dataclass
class TripleCandidate:
    head: str
    relation: str
    tail: str
    triple_type: Optional[str] = None
    source: str = ""


@dataclass
class ExtractionConfig:
    supporting_pages_only: bool = False
    include_question_entities: bool = False
    use_spacy: bool = False
    max_triples_per_sentence: int = 32


class NoLLMTripleExtractor:
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._nlp = None
        if config.use_spacy and spacy is not None:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = None

    @property
    def spacy_enabled(self) -> bool:
        return self._nlp is not None

    def build_graph(self, sample: Dict[str, Any], idx: int = 0) -> Dict[str, Any]:
        sample_id = safe_sample_id(sample, idx)
        pages = collect_pages(sample, self.config.supporting_pages_only)
        question = norm_text(sample.get("question"))

        entity_candidates: Set[str] = set()
        triple_candidates: List[TripleCandidate] = []

        for page in pages:
            title = canonicalize_entity(page.get("title", ""))
            if title:
                entity_candidates.add(title)

            last_subject = title or ""
            sentences = page.get("sentences", []) or []
            for sentence in sentences:
                sub_sentences = simple_sentence_tokenize(sentence)
                if not sub_sentences:
                    sub_sentences = [sentence]
                for sub_sentence in sub_sentences:
                    clause_entities = self.extract_entity_candidates(sub_sentence, title=title)
                    entity_candidates.update(clause_entities)
                    clause_triples = self.extract_sentence_triples(
                        sub_sentence,
                        page_title=title,
                        known_entities=clause_entities | entity_candidates,
                        last_subject=last_subject,
                    )
                    triple_candidates.extend(clause_triples[: self.config.max_triples_per_sentence])
                    subject = self.infer_primary_subject(sub_sentence, clause_entities, title)
                    if subject:
                        last_subject = subject

        if self.config.include_question_entities and question:
            entity_candidates.update(self.extract_entity_candidates(question, title=""))

        graph = self.normalize_graph(sample_id, entity_candidates, triple_candidates)
        return graph

    def extract_entity_candidates(self, text: str, title: str = "") -> Set[str]:
        entities: Set[str] = set()
        if title:
            entities.add(canonicalize_entity(title))

        for match in _TITLE_ENTITY_RE.finditer(text):
            ent = canonicalize_entity(match.group(0))
            if self.is_valid_entity(ent):
                entities.add(ent)

        for pattern in [
            r"\b([A-Z][a-z]+(?:\s+of\s+[A-Z][a-z]+)+)\b",
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\b",
        ]:
            for m in re.finditer(pattern, text):
                ent = canonicalize_entity(m.group(1))
                if self.is_valid_entity(ent):
                    entities.add(ent)

        if self.spacy_enabled:
            doc = self._nlp(text)
            for ent in doc.ents:
                name = canonicalize_entity(ent.text)
                if self.is_valid_entity(name):
                    entities.add(name)

        return entities

    def infer_primary_subject(self, sentence: str, entities: Set[str], page_title: str) -> str:
        if page_title and page_title in sentence:
            return page_title
        ordered = sorted(entities, key=lambda x: (-len(x.split()), sentence.find(x) if x in sentence else 10**9))
        for ent in ordered:
            if ent and ent in sentence:
                return ent
        return page_title or ""

    def extract_sentence_triples(
        self,
        sentence: str,
        *,
        page_title: str,
        known_entities: Set[str],
        last_subject: str,
    ) -> List[TripleCandidate]:
        triples: List[TripleCandidate] = []
        clauses = split_sentence_clauses(sentence)
        if not clauses:
            clauses = [sentence]

        for clause in clauses:
            triples.extend(self.extract_possessive_attribute_triples(clause, known_entities, page_title))
            triples.extend(self.extract_copula_triples(clause, known_entities, page_title, last_subject))
            triples.extend(self.extract_preposition_triples(clause, known_entities, page_title, last_subject))
            triples.extend(self.extract_year_and_date_triples(clause, known_entities, page_title, last_subject))
            if self.spacy_enabled:
                triples.extend(self.extract_spacy_dependency_triples(clause, known_entities, page_title, last_subject))

        return triples

    def extract_possessive_attribute_triples(
        self,
        clause: str,
        known_entities: Set[str],
        page_title: str,
    ) -> List[TripleCandidate]:
        triples: List[TripleCandidate] = []
        patterns = [
            (r"^(?P<head>[^,]+?)'s (?P<rel>[a-zA-Z][a-zA-Z\s-]{1,40}?) is (?P<tail>.+)$", "possessive-is"),
            (r"^(?P<head>[^,]+?)'s (?P<rel>[a-zA-Z][a-zA-Z\s-]{1,40}?) was (?P<tail>.+)$", "possessive-was"),
        ]
        for pattern, source in patterns:
            m = re.search(pattern, clause)
            if not m:
                continue
            head = self.resolve_head(m.group("head"), known_entities, page_title)
            relation = self.normalize_attribute_relation(m.group("rel"))
            tail = norm_text(m.group("tail")).strip(" .")
            if head and relation and tail:
                triples.append(TripleCandidate(head, relation, tail, "ATTRIBUTE", source))
        return triples

    def extract_copula_triples(
        self,
        clause: str,
        known_entities: Set[str],
        page_title: str,
        last_subject: str,
    ) -> List[TripleCandidate]:
        triples: List[TripleCandidate] = []
        m = re.search(r"^(?P<head>.+?)\s+(?P<cop>is|was|were|are)\s+(?P<tail>.+)$", clause, re.IGNORECASE)
        if not m:
            return triples

        raw_head = norm_text(m.group("head")).strip(" ,")
        raw_tail = norm_text(m.group("tail")).strip(" .")
        head = self.resolve_head(raw_head, known_entities, page_title, last_subject=last_subject)
        if not head:
            return triples

        if self.looks_like_attribute_phrase(raw_tail):
            relation, value = self.attribute_from_copula_tail(raw_tail)
            if relation and value:
                triples.append(TripleCandidate(head, relation, value, "ATTRIBUTE", "copula-attribute"))
            return triples

        tail_entity = self.resolve_tail_entity(raw_tail, known_entities)
        if tail_entity:
            triples.append(TripleCandidate(head, "is", tail_entity, "RELATION", "copula-relation"))
        else:
            triples.append(TripleCandidate(head, "type", raw_tail, "ATTRIBUTE", "copula-fallback"))
        return triples

    def extract_preposition_triples(
        self,
        clause: str,
        known_entities: Set[str],
        page_title: str,
        last_subject: str,
    ) -> List[TripleCandidate]:
        triples: List[TripleCandidate] = []
        patterns = [
            (r"^(?P<head>.+?)\s+located in\s+(?P<tail>.+)$", "located in"),
            (r"^(?P<head>.+?)\s+in\s+(?P<tail>.+)$", "in"),
            (r"^(?P<head>.+?)\s+from\s+(?P<tail>.+)$", "from"),
            (r"^(?P<head>.+?)\s+by\s+(?P<tail>.+)$", "by"),
            (r"^(?P<head>.+?)\s+near\s+(?P<tail>.+)$", "near"),
        ]
        for pattern, relation in patterns:
            m = re.search(pattern, clause, re.IGNORECASE)
            if not m:
                continue
            head = self.resolve_head(m.group("head"), known_entities, page_title, last_subject=last_subject)
            raw_tail = norm_text(m.group("tail")).strip(" .")
            tail_entity = self.resolve_tail_entity(raw_tail, known_entities)
            if head and tail_entity:
                rel = relation if relation != "in" else "located in"
                triples.append(TripleCandidate(head, rel, tail_entity, "RELATION", "prep"))
        return triples

    def extract_year_and_date_triples(
        self,
        clause: str,
        known_entities: Set[str],
        page_title: str,
        last_subject: str,
    ) -> List[TripleCandidate]:
        triples: List[TripleCandidate] = []
        head = self.resolve_head("", known_entities, page_title, last_subject=last_subject, allow_page_title=True)
        if not head:
            return triples

        year_match = _YEAR_RE.search(clause)
        if year_match:
            relation = self.infer_year_relation(clause)
            triples.append(TripleCandidate(head, relation, year_match.group(0), "ATTRIBUTE", "year"))

        for date_match in _DATEISH_RE.finditer(clause):
            relation = self.infer_date_relation(clause)
            triples.append(TripleCandidate(head, relation, date_match.group(0), "ATTRIBUTE", "date"))
        return triples

    def extract_spacy_dependency_triples(
        self,
        clause: str,
        known_entities: Set[str],
        page_title: str,
        last_subject: str,
    ) -> List[TripleCandidate]:
        triples: List[TripleCandidate] = []
        doc = self._nlp(clause)
        for token in doc:
            if token.pos_ not in {"VERB", "AUX"}:
                continue
            subjects = [child for child in token.children if child.dep_ in {"nsubj", "nsubjpass"}]
            objects = [child for child in token.children if child.dep_ in {"dobj", "attr", "oprd", "pobj"}]
            if not subjects or not objects:
                continue
            for subj in subjects:
                head = self.resolve_head(subj.text, known_entities, page_title, last_subject=last_subject)
                if not head:
                    continue
                for obj in objects:
                    tail_text = norm_text(self._spacy_subtree_text(obj)).strip(" .")
                    tail_entity = self.resolve_tail_entity(tail_text, known_entities)
                    relation = normalize_relation_phrase(token.lemma_ or token.text)
                    if not relation:
                        continue
                    if tail_entity:
                        triples.append(TripleCandidate(head, relation, tail_entity, "RELATION", "spacy-svo"))
                    elif tail_text:
                        attr_relation = self.normalize_attribute_relation(relation)
                        triples.append(TripleCandidate(head, attr_relation, tail_text, "ATTRIBUTE", "spacy-svo"))
        return triples

    def _spacy_subtree_text(self, token: Any) -> str:
        try:
            subtree = list(token.subtree)
        except Exception:
            subtree = []
        if subtree:
            return " ".join(norm_text(tok.text) for tok in subtree if norm_text(tok.text))
        return norm_text(getattr(token, "text", ""))

    def resolve_head(
        self,
        raw_head: str,
        known_entities: Set[str],
        page_title: str,
        *,
        last_subject: str = "",
        allow_page_title: bool = True,
    ) -> str:
        text = norm_text(raw_head).strip(" ,")
        lowered = text.lower()
        if lowered in {"he", "she", "it", "they", "his", "her", "its", "their"}:
            return canonicalize_entity(last_subject or page_title)
        if not text and allow_page_title:
            return canonicalize_entity(last_subject or page_title)

        entity = self.resolve_tail_entity(text, known_entities)
        if entity:
            return entity

        if self.is_valid_entity(text):
            return canonicalize_entity(text)
        return canonicalize_entity(last_subject or page_title) if allow_page_title else ""

    def resolve_tail_entity(self, raw_tail: str, known_entities: Set[str]) -> str:
        tail = canonicalize_entity(raw_tail)
        if not tail:
            return ""
        if is_literal_value(tail):
            return ""
        if tail in known_entities:
            return tail
        for ent in sorted(known_entities, key=len, reverse=True):
            if ent and (tail == ent or tail.startswith(ent + ",") or ent in tail):
                return ent
        if self.is_valid_entity(tail):
            return tail
        return ""

    def normalize_attribute_relation(self, text: str) -> str:
        rel = normalize_relation_phrase(text)
        mapping = {
            "published": "publication year",
            "released": "release year",
            "born": "birth year",
            "died": "death year",
            "language": "language",
            "genre": "genre",
            "occupation": "occupation",
        }
        return mapping.get(rel, rel)

    def looks_like_attribute_phrase(self, tail: str) -> bool:
        if is_literal_value(tail):
            return True
        if _ATTRIBUTE_HINT_RE.search(tail):
            return True
        if len(tail.split()) <= 4 and tail.islower():
            return True
        return False

    def attribute_from_copula_tail(self, tail: str) -> Tuple[str, str]:
        s = norm_text(tail).strip(" .")
        if _YEAR_RE.fullmatch(s):
            return "year", s
        if _DATEISH_RE.search(s):
            return "date", s
        if s.lower().startswith(("a ", "an ", "the ")):
            return "type", re.sub(r"^(?:a|an|the)\s+", "", s, flags=re.IGNORECASE)
        return "type", s

    def infer_year_relation(self, clause: str) -> str:
        s = clause.lower()
        if "born" in s:
            return "birth year"
        if "died" in s:
            return "death year"
        if "published" in s:
            return "publication year"
        if "released" in s:
            return "release year"
        if "founded" in s or "formed" in s:
            return "founding year"
        return "year"

    def infer_date_relation(self, clause: str) -> str:
        s = clause.lower()
        if "born" in s:
            return "birth date"
        if "died" in s:
            return "death date"
        if "published" in s:
            return "publication date"
        if "released" in s:
            return "release date"
        return "date"

    def is_valid_entity(self, text: str) -> bool:
        ent = canonicalize_entity(text)
        if not ent:
            return False
        if is_placeholder_entity_code(ent):
            return False
        if ent in _STOP_ENTITY_WORDS:
            return False
        if _YEAR_RE.fullmatch(ent):
            return False
        if is_literal_value(ent):
            return False
        if len(ent) == 1:
            return False
        return True

    def normalize_graph(
        self,
        sample_id: str,
        entity_candidates: Set[str],
        triple_candidates: Sequence[TripleCandidate],
    ) -> Dict[str, Any]:
        entities: List[str] = []
        entity_seen: Set[str] = set()
        for ent in sorted(entity_candidates):
            name = canonicalize_entity(ent)
            if not self.is_valid_entity(name):
                continue
            key = name.lower()
            if key in entity_seen:
                continue
            entity_seen.add(key)
            entities.append(name)

        entity_set = set(entities)
        triples: List[Dict[str, str]] = []
        seen: Set[Tuple[str, str, str, str]] = set()

        for tri in triple_candidates:
            head = canonicalize_entity(tri.head)
            relation = normalize_relation_phrase(tri.relation)
            tail = norm_text(tri.tail).strip(" .")
            if not head or not relation or not tail:
                continue
            if head not in entity_set and self.is_valid_entity(head):
                entity_set.add(head)
                entities.append(head)

            triple_type = tri.triple_type or self.classify_triple_type(tail, entity_set)
            if triple_type == "RELATION":
                tail_entity = self.resolve_tail_entity(tail, entity_set)
                if not tail_entity:
                    triple_type = "ATTRIBUTE"
                else:
                    tail = tail_entity
                    if tail not in entity_set and self.is_valid_entity(tail):
                        entity_set.add(tail)
                        entities.append(tail)

            if triple_type == "ATTRIBUTE" and not tail:
                continue

            if triple_type == "RELATION" and head == tail:
                continue

            sig = (head, relation, tail, triple_type)
            if sig in seen:
                continue
            seen.add(sig)
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

    def classify_triple_type(self, tail: str, entity_set: Set[str]) -> str:
        if tail in entity_set:
            return "RELATION"
        if is_literal_value(tail):
            return "ATTRIBUTE"
        if self.is_valid_entity(tail):
            return "RELATION"
        return "ATTRIBUTE"


def get_triple_type(tri: Dict[str, Any]) -> str:
    triple_type = norm_text(tri.get("triple_type", "")).upper()
    if triple_type in {"ATTRIBUTE", "RELATION"}:
        return triple_type
    return "ATTRIBUTE" if is_literal_value(tri.get("tail", "")) else "RELATION"


def dedupe_stage_graph(stage_graph: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "_id": extract_sample_id(stage_graph),
        "entity_list": [],
        "triples": [],
    }

    entity_seen: Set[str] = set()
    for ent in stage_graph.get("entity_list", []) or []:
        name = canonicalize_entity(ent)
        if not name or is_placeholder_entity_code(name):
            continue
        key = name.lower()
        if key in entity_seen:
            continue
        entity_seen.add(key)
        out["entity_list"].append(name)

    triple_seen: Set[Tuple[str, str, str, str]] = set()
    for tri in stage_graph.get("triples", []) or []:
        if not isinstance(tri, dict):
            continue
        head = canonicalize_entity(tri.get("head", ""))
        relation = normalize_relation_phrase(tri.get("relation", ""))
        tail = norm_text(tri.get("tail", "")).strip(" .")
        triple_type = get_triple_type(tri)
        if not head or not relation or not tail:
            continue
        if triple_type == "RELATION":
            tail = canonicalize_entity(tail)
            if not tail or is_placeholder_entity_code(tail):
                continue
        sig = (head, relation, tail, triple_type)
        if sig in triple_seen:
            continue
        triple_seen.add(sig)
        out["triples"].append(
            {
                "head": head,
                "relation": relation,
                "tail": tail,
                "triple_type": triple_type,
            }
        )
        if head not in out["entity_list"]:
            out["entity_list"].append(head)
        if triple_type == "RELATION" and tail not in out["entity_list"]:
            out["entity_list"].append(tail)

    return out


def graph_value_matches(value: str, answer: str) -> bool:
    left = norm_text(value).lower()
    right = norm_text(answer).lower()
    if not left or not right:
        return False
    if left == right:
        return True
    if left in right or right in left:
        return True
    return False


def graph_entity_matches(entity: str, answer: str) -> bool:
    return graph_value_matches(canonicalize_entity(entity), canonicalize_entity(answer))


def build_graph_neighbors(stage_graph: Dict[str, Any]) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    neighbors: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for tri in stage_graph.get("triples", []) or []:
        if get_triple_type(tri) != "RELATION":
            continue
        head = canonicalize_entity(tri.get("head", ""))
        tail = canonicalize_entity(tri.get("tail", ""))
        if not head or not tail:
            continue
        neighbors.setdefault(head, []).append((tail, tri))
        neighbors.setdefault(tail, []).append((head, tri))
    return neighbors


def find_answer_nodes(stage_graph: Dict[str, Any], answer: str) -> Tuple[Set[str], List[Dict[str, Any]]]:
    answer_entities: Set[str] = set()
    answer_attributes: List[Dict[str, Any]] = []
    for ent in stage_graph.get("entity_list", []) or []:
        if graph_entity_matches(ent, answer):
            answer_entities.add(canonicalize_entity(ent))
    for tri in stage_graph.get("triples", []) or []:
        tri_type = get_triple_type(tri)
        if tri_type == "RELATION" and graph_entity_matches(tri.get("tail", ""), answer):
            answer_entities.add(canonicalize_entity(tri.get("tail", "")))
        if tri_type == "ATTRIBUTE" and graph_value_matches(tri.get("tail", ""), answer):
            answer_attributes.append(tri)
    return answer_entities, answer_attributes


def bfs_relation_path(
    stage_graph: Dict[str, Any],
    anchors: Sequence[str],
    targets: Set[str],
    max_hops: int = 4,
) -> List[Dict[str, Any]]:
    if not anchors or not targets:
        return []
    neighbors = build_graph_neighbors(stage_graph)
    queue: deque[Tuple[str, List[Dict[str, Any]], int]] = deque()
    visited: Set[str] = set()
    for anchor in anchors:
        if anchor in targets:
            return []
        queue.append((anchor, [], 0))
        visited.add(anchor)

    while queue:
        node, path, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for nxt, tri in neighbors.get(node, []):
            if nxt in visited:
                continue
            next_path = path + [tri]
            if nxt in targets:
                return next_path
            visited.add(nxt)
            queue.append((nxt, next_path, depth + 1))
    return []


def infer_question_anchors(
    sample: Dict[str, Any],
    stage_graph: Dict[str, Any],
    extractor: NoLLMTripleExtractor,
) -> List[str]:
    question = norm_text(sample.get("question", ""))
    if not question:
        return []
    question_entities = extractor.extract_entity_candidates(question)
    graph_entities = [canonicalize_entity(ent) for ent in stage_graph.get("entity_list", []) or []]
    matched: List[str] = []
    question_lower = question.lower()

    for ent in graph_entities:
        if ent in question_entities or ent.lower() in question_lower:
            matched.append(ent)

    if matched:
        return sorted(set(matched))

    title_hits: List[str] = []
    for page in collect_pages(sample, supporting_pages_only=False):
        title = canonicalize_entity(page.get("title", ""))
        if title and title.lower() in question_lower:
            title_hits.append(title)
    return sorted(set(title_hits))


def infer_relation_from_question(question: str) -> str:
    q = norm_text(question).lower()
    if "when" in q:
        if "born" in q:
            return "birth date"
        if "release" in q or "published" in q:
            return "release date"
        return "date"
    if "where" in q:
        return "located in"
    if "who" in q:
        return "person"
    if "capital" in q:
        return "capital"
    if "population" in q:
        return "population"
    if "language" in q:
        return "language"
    if "genre" in q:
        return "genre"
    return "related to"


def collect_candidate_repair_sentences(sample: Dict[str, Any], answer: str, anchors: Sequence[str]) -> List[Tuple[str, str]]:
    answer_lower = norm_text(answer).lower()
    anchor_lowers = [a.lower() for a in anchors if a]
    out: List[Tuple[str, str]] = []
    for page in collect_pages(sample, supporting_pages_only=False):
        title = canonicalize_entity(page.get("title", ""))
        for sentence in page.get("sentences", []) or []:
            sent_lower = sentence.lower()
            if answer_lower and answer_lower in sent_lower:
                out.append((title, sentence))
                continue
            if any(anchor in sent_lower for anchor in anchor_lowers):
                out.append((title, sentence))
    return out


def heuristic_graph_revision(
    sample: Dict[str, Any],
    stage1: Dict[str, Any],
    extractor: NoLLMTripleExtractor,
    *,
    max_hops: int = 4,
) -> Dict[str, Any]:
    question = norm_text(sample.get("question", ""))
    answer = norm_text(sample.get("answer", ""))
    revised = {
        "_id": extract_sample_id(stage1),
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
    revised["answer_sufficient"] = False
    revised["missing_links"] = []
    revised["revision_notes"] = []
    anchors = infer_question_anchors(sample, revised, extractor)
    answer_entities, answer_attributes = find_answer_nodes(revised, answer)

    if not anchors:
        revised["missing_links"] = ["missing question anchor in graph"]
    else:
        attr_supported = any(canonicalize_entity(tri.get("head", "")) in anchors for tri in answer_attributes)
        relation_path = bfs_relation_path(revised, anchors, answer_entities, max_hops=max_hops)
        if attr_supported or answer_entities & set(anchors) or relation_path:
            revised["answer_sufficient"] = True
            return revised

    repair_sentences = collect_candidate_repair_sentences(sample, answer, anchors)
    added = 0
    for title, sentence in repair_sentences:
        known_entities = set(revised.get("entity_list", []) or [])
        known_entities.update(extractor.extract_entity_candidates(sentence, title=title))
        for tri in extractor.extract_sentence_triples(
            sentence,
            page_title=title,
            known_entities=known_entities,
            last_subject=title,
        ):
            head = canonicalize_entity(tri.head)
            relation = normalize_relation_phrase(tri.relation)
            tail = norm_text(tri.tail).strip(" .")
            triple_type = tri.triple_type or ("ATTRIBUTE" if is_literal_value(tail) else "RELATION")
            if hasattr(extractor, "normalize_relation_semantics"):
                relation, triple_type = extractor.normalize_relation_semantics(  # type: ignore[attr-defined]
                    relation,
                    head=head,
                    tail=tail,
                    triple_type=triple_type,
                )
            if triple_type == "RELATION":
                tail = canonicalize_entity(tail)
            if hasattr(extractor, "should_accept_revision_candidate"):
                if not extractor.should_accept_revision_candidate(  # type: ignore[attr-defined]
                    head=head,
                    relation=relation,
                    tail=tail,
                    triple_type=triple_type,
                    known_entities=known_entities | set(revised.get("entity_list", []) or []),
                ):
                    continue
            candidate = {
                "head": head,
                "relation": relation,
                "tail": tail,
                "triple_type": triple_type,
            }
            sig = (candidate["head"], candidate["relation"], candidate["tail"], candidate["triple_type"])
            existing = {
                (canonicalize_entity(x.get("head", "")), normalize_relation_phrase(x.get("relation", "")), norm_text(x.get("tail", "")).strip(" ."), get_triple_type(x))
                for x in revised.get("triples", []) or []
                if isinstance(x, dict)
            }
            if sig in existing:
                continue
            if answer and not (
                graph_value_matches(candidate["tail"], answer)
                or graph_entity_matches(candidate["tail"], answer)
                or any(graph_entity_matches(candidate["head"], anchor) for anchor in anchors)
            ):
                continue
            revised["triples"].append(candidate)
            if candidate["head"] not in revised["entity_list"]:
                revised["entity_list"].append(candidate["head"])
            if candidate["triple_type"] == "RELATION" and candidate["tail"] not in revised["entity_list"]:
                revised["entity_list"].append(candidate["tail"])
            revised["revision_notes"].append(
                f"added missing {'attribute' if candidate['triple_type'] == 'ATTRIBUTE' else 'relation'} triple: "
                f"{candidate['head']} | {candidate['relation']} | {candidate['tail']}"
            )
            added += 1

    if not added and anchors:
        relation_guess = infer_relation_from_question(question)
        for anchor in anchors:
            if graph_value_matches(anchor, answer):
                continue
            if is_literal_value(answer):
                revised["triples"].append(
                    {
                        "head": anchor,
                        "relation": relation_guess,
                        "tail": answer,
                        "triple_type": "ATTRIBUTE",
                    }
                )
                revised["revision_notes"].append(
                    f"added heuristic attribute bridge: {anchor} | {relation_guess} | {answer}"
                )
                added += 1
                break

    revision_notes = list(revised.get("revision_notes", []) or [])
    revised = dedupe_stage_graph(revised)
    revised["revision_notes"] = revision_notes
    answer_entities, answer_attributes = find_answer_nodes(revised, answer)
    anchors = infer_question_anchors(sample, revised, extractor)
    attr_supported = any(canonicalize_entity(tri.get("head", "")) in anchors for tri in answer_attributes)
    relation_path = bfs_relation_path(revised, anchors, answer_entities, max_hops=max_hops)

    revised["answer_sufficient"] = bool(
        attr_supported or answer_entities & set(anchors) or relation_path
    )
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


def merge_stage1_and_stage2_graph(stage1: Dict[str, Any], stage2: Dict[str, Any]) -> Dict[str, Any]:
    merged = {
        "_id": extract_sample_id(stage2) or extract_sample_id(stage1),
        "entity_list": list(stage1.get("entity_list", []) or []) + list(stage2.get("entity_list", []) or []),
        "triples": list(stage1.get("triples", []) or []) + list(stage2.get("triples", []) or []),
        "answer_sufficient": bool(stage2.get("answer_sufficient", False)),
        "missing_links": [norm_text(x) for x in (stage2.get("missing_links", []) or []) if norm_text(x)],
        "revision_notes": [norm_text(x) for x in (stage2.get("revision_notes", []) or []) if norm_text(x)],
    }
    merged = dedupe_stage_graph(merged)
    merged["answer_sufficient"] = bool(stage2.get("answer_sufficient", False))
    merged["missing_links"] = [norm_text(x) for x in (stage2.get("missing_links", []) or []) if norm_text(x)]
    merged["revision_notes"] = [norm_text(x) for x in (stage2.get("revision_notes", []) or []) if norm_text(x)]
    return merged


def programmatic_forward_kv(tri: Dict[str, Any]) -> Dict[str, str]:
    triple_type = get_triple_type(tri)
    head = canonicalize_entity(tri.get("head", ""))
    rel = normalize_relation_phrase(tri.get("relation", ""))
    tail = norm_text(tri.get("tail", "")).strip(" .")
    if triple_type == "ATTRIBUTE":
        return {"key_string": f"the {rel} of {head} is", "value_string": tail}
    return {"key_string": f"{head} {rel}", "value_string": tail}


def programmatic_reverse_kv(tri: Dict[str, Any]) -> Dict[str, str]:
    triple_type = get_triple_type(tri)
    head = canonicalize_entity(tri.get("head", ""))
    rel = normalize_relation_phrase(tri.get("relation", ""))
    tail = norm_text(tri.get("tail", "")).strip(" .")
    if triple_type == "ATTRIBUTE":
        return {"key_string": f"{tail} is the {rel} of", "value_string": head}
    return {"key_string": f"the entity that {rel} {tail} is", "value_string": head}


def build_programmatic_final_from_graph(sample: Dict[str, Any], graph_for_kv: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "_id": extract_sample_id(graph_for_kv) or extract_sample_id(sample),
        "triple_list": [],
        "answer_sufficient": bool(graph_for_kv.get("answer_sufficient", False)),
        "missing_links": [norm_text(x) for x in (graph_for_kv.get("missing_links", []) or []) if norm_text(x)],
        "revision_notes": [norm_text(x) for x in (graph_for_kv.get("revision_notes", []) or []) if norm_text(x)],
    }
    graph_for_kv = dedupe_stage_graph(graph_for_kv)
    for tri in graph_for_kv.get("triples", []) or []:
        out["triple_list"].append(
            {
                "type": get_triple_type(tri),
                "name": canonicalize_entity(tri.get("head", "")),
                "description_type": normalize_relation_phrase(tri.get("relation", "")),
                "description": norm_text(tri.get("tail", "")).strip(" ."),
                "kv_lists": [
                    programmatic_forward_kv(tri),
                    programmatic_reverse_kv(tri),
                ],
            }
        )
    return out


def merge_final_stage_into_sample(sample: Dict[str, Any], stage_final: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(sample)
    out["_id"] = extract_sample_id(stage_final) or extract_sample_id(sample)
    out["triple_list"] = copy.deepcopy(stage_final.get("triple_list", []) or [])
    out["answer_sufficient"] = bool(stage_final.get("answer_sufficient", False))
    out["missing_links"] = [norm_text(x) for x in (stage_final.get("missing_links", []) or []) if norm_text(x)]
    out["revision_notes"] = [norm_text(x) for x in (stage_final.get("revision_notes", []) or []) if norm_text(x)]
    return out


def normalize_final_sample_output(sample: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(sample)
    normalized_context: List[Dict[str, Any]] = []
    for raw_para in iter_sample_context_entries(out):
        para = normalize_context_entry(raw_para)
        if para is None:
            continue
        normalized_context.append(
            {
                "title": para["title"],
                "sentences": [norm_text(s) for s in (para.get("sentences", []) or []) if norm_text(s)],
            }
        )
    out["context"] = normalized_context
    out["triple_list"] = out.get("triple_list", []) or []
    out["answer_sufficient"] = bool(out.get("answer_sufficient", False))
    out["missing_links"] = [norm_text(x) for x in (out.get("missing_links", []) or []) if norm_text(x)]
    out["revision_notes"] = [norm_text(x) for x in (out.get("revision_notes", []) or []) if norm_text(x)]
    out.pop("paragraphs", None)
    return out


def save_cached_json(path: Optional[str], obj: Dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(target)


def load_cached_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def cache_path(cache_dir: Optional[str], sample_id: str, suffix: str) -> Optional[str]:
    if not cache_dir:
        return None
    return str(Path(cache_dir) / f"{sample_id}.{suffix}.json")
