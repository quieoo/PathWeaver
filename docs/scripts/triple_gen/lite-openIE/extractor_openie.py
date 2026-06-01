from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib import error, parse, request


_SIBLING_NO_LLM_DIR = Path(__file__).resolve().parent.parent / "triple_gen_no_llm"
if str(_SIBLING_NO_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIBLING_NO_LLM_DIR))

from extractor import (  # type: ignore
    ExtractionConfig,
    NoLLMTripleExtractor,
    TripleCandidate,
    canonicalize_entity,
    collect_pages,
    dedupe_stage_graph,
    get_supporting_titles,
    is_literal_value,
    norm_text,
    normalize_relation_phrase,
    safe_sample_id,
)

_PRONOUNS = {
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
_WEAK_RELATIONS = {"is", "was", "are", "were", "has", "have", "had"}
_AWARD_WORDS = {"award", "awards", "prize", "prizes", "medal", "medals"}
_ROLE_HEAD_WORDS = {
    "actor",
    "actress",
    "award",
    "baron",
    "born",
    "count",
    "countess",
    "director",
    "duchess",
    "duke",
    "earl",
    "elector",
    "emperor",
    "father",
    "first",
    "governor",
    "king",
    "lord",
    "margrave",
    "mother",
    "parent",
    "prince",
    "princess",
    "prior",
    "producer",
    "queen",
    "saint",
    "sir",
    "writer",
}


def json_dumps_stable(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def iter_sentence_openie_triples(annotated: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for sentence in annotated.get("sentences", []) or []:
        for triple in sentence.get("openie", []) or []:
            if isinstance(triple, dict):
                yield sentence, triple


def pick_first_text(data: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and norm_text(value):
            return norm_text(value)
        if isinstance(value, list):
            joined = norm_text(" ".join(norm_text(x) for x in value if norm_text(x)))
            if joined:
                return joined
    return ""


def sentence_ner_entities(sentence: Dict[str, Any]) -> Set[str]:
    entities: Set[str] = set()
    current: List[str] = []
    current_tag = "O"
    for tok in sentence.get("tokens", []) or []:
        if not isinstance(tok, dict):
            continue
        word = norm_text(tok.get("originalText") or tok.get("word") or tok.get("lemma"))
        ner = norm_text(tok.get("ner") or "O")
        if not word:
            continue
        if ner and ner != "O":
            if current and ner != current_tag:
                ent = canonicalize_entity(" ".join(current))
                if ent:
                    entities.add(ent)
                current = []
            current.append(word)
            current_tag = ner
            continue
        if current:
            ent = canonicalize_entity(" ".join(current))
            if ent:
                entities.add(ent)
            current = []
            current_tag = "O"
    if current:
        ent = canonicalize_entity(" ".join(current))
        if ent:
            entities.add(ent)
    return entities


def drop_leading_article(text: str) -> str:
    lowered = norm_text(text)
    for prefix in ("the ", "a ", "an "):
        if lowered.lower().startswith(prefix):
            return norm_text(lowered[len(prefix):])
    return lowered


def sentence_text(sentence: Dict[str, Any]) -> str:
    text = norm_text(sentence.get("text"))
    if text:
        return text
    toks: List[str] = []
    for tok in sentence.get("tokens", []) or []:
        if not isinstance(tok, dict):
            continue
        word = norm_text(tok.get("originalText") or tok.get("word"))
        if word:
            toks.append(word)
    return norm_text(" ".join(toks))


@dataclass
class OpenIEConfig:
    corenlp_url: str = "http://localhost:9000"
    supporting_pages_only: bool = True
    include_question_entities: bool = False
    request_timeout: float = 120.0
    strict: bool = True
    max_entailments_per_clause: int = 150
    triple_all_nominals: bool = False
    resolve_coref: bool = True
    with_ner: bool = True
    min_confidence: float = 0.35
    max_triples_per_sentence: int = 8


class StanfordOpenIEExtractor:
    def __init__(self, config: OpenIEConfig):
        self.config = config
        self._repair_extractor = NoLLMTripleExtractor(
            ExtractionConfig(
                supporting_pages_only=config.supporting_pages_only,
                include_question_entities=config.include_question_entities,
                use_spacy=False,
                max_triples_per_sentence=config.max_triples_per_sentence,
            )
        )

    def build_graph(self, sample: Dict[str, Any], idx: int = 0) -> Dict[str, Any]:
        sample_id = safe_sample_id(sample, idx)
        pages = collect_pages(sample, self.config.supporting_pages_only)
        question = norm_text(sample.get("question"))
        supporting_titles = set(get_supporting_titles(sample))

        entity_candidates: Set[str] = set()
        triple_candidates: List[TripleCandidate] = []

        for page in pages:
            title = canonicalize_entity(page.get("title", ""))
            if title:
                entity_candidates.add(title)
            sentences = [norm_text(s) for s in (page.get("sentences", []) or []) if norm_text(s)]
            if not sentences:
                continue

            annotated = self.annotate_page(sentences)
            page_entities, page_triples = self.extract_from_annotation(
                annotated,
                page_title=title,
                supporting_titles=supporting_titles,
            )
            entity_candidates.update(page_entities)
            triple_candidates.extend(page_triples)

        if self.config.include_question_entities and question:
            entity_candidates.update(self.lightweight_question_entities(question))

        graph = self.normalize_graph(sample_id, entity_candidates, triple_candidates)
        return dedupe_stage_graph(graph)

    def lightweight_question_entities(self, question: str) -> Set[str]:
        entities: Set[str] = set()
        for piece in question.replace("?", ".").split("."):
            chunk = norm_text(piece)
            if not chunk:
                continue
            words = chunk.split()
            running: List[str] = []
            for word in words:
                if word[:1].isupper():
                    running.append(word)
                    continue
                if running:
                    ent = canonicalize_entity(" ".join(running))
                    if ent:
                        entities.add(ent)
                    running = []
            if running:
                ent = canonicalize_entity(" ".join(running))
                if ent:
                    entities.add(ent)
        return entities

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
        return self._repair_extractor.extract_sentence_triples(
            sentence,
            page_title=page_title,
            known_entities=known_entities,
            last_subject=last_subject,
        )

    def is_valid_entity(self, text: str) -> bool:
        return self._repair_extractor.is_valid_entity(text)

    def resolve_tail_entity(self, raw_tail: str, known_entities: Set[str]) -> str:
        return self._repair_extractor.resolve_tail_entity(raw_tail, known_entities)

    def infer_primary_subject(self, sentence: str, entities: Set[str], page_title: str) -> str:
        return self._repair_extractor.infer_primary_subject(sentence, entities, page_title)

    def normalize_openie_endpoint(self, text: str, *, page_title: str = "") -> str:
        value = canonicalize_entity(text)
        if not value:
            return ""
        if value.lower() in _PRONOUNS and page_title:
            return canonicalize_entity(page_title)
        return value

    def looks_like_generic_role_head(self, text: str) -> bool:
        value = canonicalize_entity(text)
        if not value:
            return True
        lowered = value.lower()
        if lowered in _PRONOUNS:
            return True
        parts = lowered.split()
        if len(parts) == 1 and parts[0] in _ROLE_HEAD_WORDS:
            return True
        if len(parts) <= 3 and any(word in _AWARD_WORDS for word in parts):
            return True
        if parts[:1] and parts[0] in _ROLE_HEAD_WORDS and len(parts) <= 4:
            if " of " in lowered or lowered.startswith("the "):
                return True
        return False

    def normalize_relation_semantics(
        self,
        relation: str,
        *,
        head: str,
        tail: str,
        triple_type: str,
    ) -> Tuple[str, str]:
        rel = normalize_relation_phrase(relation)
        if not rel:
            return "", triple_type
        lowered_tail = norm_text(tail).lower()
        if rel in _WEAK_RELATIONS and "award" in lowered_tail and "winning" in lowered_tail:
            award_bits: List[str] = []
            for token in canonicalize_entity(tail).split():
                token_lower = token.lower().strip(",.;:")
                if token_lower == "winning":
                    break
                award_bits.append(token)
            award_name = canonicalize_entity(" ".join(award_bits))
            if award_name:
                return "award received", "RELATION"
        mapping = {
            "directed by": "director",
            "was directed by": "director",
            "written by": "writer",
            "was written by": "writer",
            "is son of": "parent",
            "son of": "parent",
            "is daughter of": "parent",
            "daughter of": "parent",
            "father of": "parent",
            "mother of": "parent",
            "born in": "place of birth",
            "died in": "place of death",
            "won": "award received",
            "awarded with": "award received",
        }
        rel = mapping.get(rel, rel)
        normalized_type = triple_type
        if rel in {"director", "writer", "parent", "award received", "place of birth", "place of death"}:
            normalized_type = "RELATION"
        if rel in _WEAK_RELATIONS and not is_literal_value(tail):
            normalized_type = "ATTRIBUTE"
        return rel, normalized_type

    def relation_is_too_weak(self, relation: str) -> bool:
        rel = normalize_relation_phrase(relation)
        return not rel or rel in _WEAK_RELATIONS or len(rel.split()) > 6

    def should_keep_openie_triple(
        self,
        *,
        head: str,
        relation: str,
        tail: str,
        confidence: float,
        known_entities: Set[str],
    ) -> bool:
        if confidence < self.config.min_confidence:
            return False
        if not head or not relation or not tail:
            return False
        if not self.is_valid_entity(head):
            return False
        if self.looks_like_generic_role_head(head):
            return False
        if head.lower() in _PRONOUNS or tail.lower() in _PRONOUNS:
            return False
        if self.relation_is_too_weak(relation) and not is_literal_value(tail):
            return False
        if len(tail.split()) > 8 and not is_literal_value(tail):
            return False
        if head.lower() == tail.lower():
            return False
        if not is_literal_value(tail) and not self.resolve_tail_entity(tail, known_entities):
            return False
        return True

    def should_keep_final_triple(
        self,
        *,
        head: str,
        relation: str,
        tail: str,
        triple_type: str,
        known_entities: Set[str],
    ) -> bool:
        if not head or not relation or not tail:
            return False
        if not self.is_valid_entity(head) or self.looks_like_generic_role_head(head):
            return False
        if triple_type == "RELATION":
            if relation in _WEAK_RELATIONS:
                return False
            if not self.resolve_tail_entity(tail, known_entities):
                return False
        if triple_type == "ATTRIBUTE" and relation in _WEAK_RELATIONS and not is_literal_value(tail):
            return False
        return True

    def should_accept_revision_candidate(
        self,
        *,
        head: str,
        relation: str,
        tail: str,
        triple_type: str,
        known_entities: Set[str],
    ) -> bool:
        return self.should_keep_final_triple(
            head=head,
            relation=relation,
            tail=tail,
            triple_type=triple_type,
            known_entities=known_entities,
        )

    def annotate_page(self, sentences: Sequence[str]) -> Dict[str, Any]:
        text = "\n".join(norm_text(s) for s in sentences if norm_text(s))
        if not text:
            return {"sentences": []}

        annotators = ["tokenize", "ssplit", "pos", "lemma", "depparse"]
        if self.config.with_ner or self.config.resolve_coref:
            annotators.append("ner")
        if self.config.resolve_coref:
            annotators.append("coref")
        annotators.extend(["natlog", "openie"])

        properties: Dict[str, Any] = {
            "annotators": ",".join(annotators),
            "outputFormat": "json",
            "ssplit.eolonly": "true",
            "openie.triple.strict": str(self.config.strict).lower(),
            "openie.max_entailments_per_clause": int(self.config.max_entailments_per_clause),
            "openie.triple.all_nominals": str(self.config.triple_all_nominals).lower(),
            "openie.resolve_coref": str(self.config.resolve_coref).lower(),
        }

        url = self.config.corenlp_url.rstrip("/") + "/?properties=" + parse.quote(json_dumps_stable(properties))
        body = text.encode("utf-8")
        req = request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "text/plain; charset=utf-8")
        try:
            with request.urlopen(req, timeout=self.config.request_timeout) as resp:
                payload = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"CoreNLP HTTP {exc.code} for {self.config.corenlp_url}: {detail[:500]}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach CoreNLP server at {self.config.corenlp_url}: {exc}"
            ) from exc
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"CoreNLP returned invalid JSON: {payload[:500]}") from exc

    def extract_from_annotation(
        self,
        annotated: Dict[str, Any],
        *,
        page_title: str,
        supporting_titles: Set[str],
    ) -> Tuple[Set[str], List[TripleCandidate]]:
        entities: Set[str] = set()
        triples: List[TripleCandidate] = []
        if page_title:
            entities.add(page_title)

        per_sentence_count: Dict[int, int] = {}
        last_subject = page_title or ""

        for sent_idx, sentence in enumerate(annotated.get("sentences", []) or []):
            sent_text = sentence_text(sentence)
            sentence_entities = self.extract_entity_candidates(sent_text, title=page_title) if sent_text else set()
            if self.config.with_ner:
                sentence_entities.update(sentence_ner_entities(sentence))
            entities.update(sentence_entities)

            openie_kept = 0
            known_entities = entities | sentence_entities

            for _, tri in iter_sentence_openie_triples({"sentences": [sentence]}):
                if per_sentence_count.get(sent_idx, 0) >= self.config.max_triples_per_sentence:
                    break

                head = self.normalize_openie_endpoint(
                    pick_first_text(tri, ["subject", "subjectGloss", "subjectLemmaGloss"]),
                    page_title=page_title,
                )
                relation = normalize_relation_phrase(
                    pick_first_text(tri, ["relation", "relationGloss", "relationLemmaGloss"])
                )
                tail = self.normalize_openie_endpoint(
                    pick_first_text(tri, ["object", "objectGloss", "objectLemmaGloss"]),
                    page_title=page_title,
                )

                confidence_raw = tri.get("confidence")
                try:
                    confidence = float(confidence_raw)
                except (TypeError, ValueError):
                    confidence = 1.0

                if page_title:
                    head_cmp = drop_leading_article(head).lower()
                    title_cmp = drop_leading_article(page_title).lower()
                    if head_cmp and (head_cmp in title_cmp or title_cmp in head_cmp):
                        head = page_title
                if not self.should_keep_openie_triple(
                    head=head,
                    relation=relation,
                    tail=tail,
                    confidence=confidence,
                    known_entities=known_entities,
                ):
                    continue

                triple_type = self.classify_openie_triple(head, relation, tail, supporting_titles, known_entities)
                relation, triple_type = self.normalize_relation_semantics(
                    relation,
                    head=head,
                    tail=tail,
                    triple_type=triple_type,
                )
                if not self.should_keep_final_triple(
                    head=head,
                    relation=relation,
                    tail=tail,
                    triple_type=triple_type,
                    known_entities=known_entities | entities,
                ):
                    continue
                entities.add(head)
                if triple_type == "RELATION":
                    resolved_tail = self.resolve_tail_entity(tail, known_entities)
                    if not resolved_tail:
                        continue
                    tail = resolved_tail
                    entities.add(tail)
                triples.append(
                    TripleCandidate(
                        head=head,
                        relation=relation,
                        tail=tail,
                        triple_type=triple_type,
                        source="openie",
                    )
                )
                per_sentence_count[sent_idx] = per_sentence_count.get(sent_idx, 0) + 1
                openie_kept += 1
                if head:
                    last_subject = head

            if openie_kept > 0 or not sent_text:
                continue

            fallback_triples = self.extract_sentence_triples(
                sent_text,
                page_title=page_title,
                known_entities=known_entities,
                last_subject=last_subject,
            )
            for tri in fallback_triples[: self.config.max_triples_per_sentence]:
                head = canonicalize_entity(tri.head)
                relation = normalize_relation_phrase(tri.relation)
                tail = norm_text(tri.tail).strip(" .")
                triple_type = norm_text(tri.triple_type).upper() or "ATTRIBUTE"
                relation, triple_type = self.normalize_relation_semantics(
                    relation,
                    head=head,
                    tail=tail,
                    triple_type=triple_type,
                )
                if not head or not relation or not tail or not self.is_valid_entity(head):
                    continue
                if not self.should_keep_final_triple(
                    head=head,
                    relation=relation,
                    tail=tail,
                    triple_type=triple_type,
                    known_entities=known_entities | entities,
                ):
                    continue
                if triple_type == "RELATION":
                    resolved_tail = self.resolve_tail_entity(tail, known_entities | entities)
                    if not resolved_tail or head.lower() == resolved_tail.lower():
                        continue
                    tail = resolved_tail
                    entities.add(tail)
                entities.add(head)
                triples.append(
                    TripleCandidate(
                        head=head,
                        relation=relation,
                        tail=tail,
                        triple_type=triple_type,
                        source=tri.source or "openie-fallback",
                    )
                )
                per_sentence_count[sent_idx] = per_sentence_count.get(sent_idx, 0) + 1
                last_subject = head

        return entities, triples

    def classify_openie_triple(
        self,
        head: str,
        relation: str,
        tail: str,
        supporting_titles: Set[str],
        known_entities: Set[str],
    ) -> str:
        if is_literal_value(tail):
            return "ATTRIBUTE"

        rel = normalize_relation_phrase(relation)
        if rel in _WEAK_RELATIONS and is_literal_value(tail):
            return "ATTRIBUTE"

        if tail in supporting_titles:
            return "RELATION"

        resolved_tail = self.resolve_tail_entity(tail, known_entities)
        if resolved_tail:
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
            if not self.is_valid_entity(name):
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
            relation, triple_type = self.normalize_relation_semantics(
                relation,
                head=head,
                tail=tail,
                triple_type=triple_type,
            )
            if not head or not relation or not tail or not self.is_valid_entity(head):
                continue
            if not self.should_keep_final_triple(
                head=head,
                relation=relation,
                tail=tail,
                triple_type=triple_type,
                known_entities=set(entities),
            ):
                continue

            head = entity_index.get(head.lower(), head)
            if head.lower() not in entity_index:
                entity_index[head.lower()] = head
                entities.append(head)

            if triple_type == "RELATION":
                resolved_tail = self.resolve_tail_entity(tail, set(entities))
                if not resolved_tail or head.lower() == resolved_tail.lower():
                    continue
                tail = entity_index.get(resolved_tail.lower(), resolved_tail)
                if tail.lower() not in entity_index:
                    entity_index[tail.lower()] = tail
                    entities.append(tail)
            elif not tail:
                continue

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
