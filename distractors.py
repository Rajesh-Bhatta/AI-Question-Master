"""Candidate generation, ranking, and validation for document-grounded MCQs."""

from dataclasses import dataclass
import math
import re
from itertools import combinations
from typing import Iterable, List, Optional, Sequence

import numpy as np
from nltk.corpus import wordnet as wn


FILLER_VALUES = {
    "maybe",
    "depends",
    "none of the above",
    "all of the above",
    "information not available",
    "not mentioned in the text",
    "insufficient information",
    "none of these",
}

TYPE_TERMS = {
    "PROGRAMMING_LANGUAGE": {"python", "java", "c", "c++", "c#", "ruby", "go", "javascript", "typescript", "php", "swift", "kotlin", "rust"},
    "DATABASE": {"mysql", "postgresql", "postgres", "oracle", "sqlite", "mongodb", "redis", "cassandra"},
    "NETWORK_PROTOCOL": {"tcp", "udp", "ip", "icmp", "http", "https", "ftp", "dns", "smtp", "ssh", "tls"},
    "ALGORITHM": {"binary search", "linear search", "merge sort", "quick sort", "quicksort", "bfs", "dfs", "dijkstra"},
    "DATA_STRUCTURE": {"array", "stack", "queue", "deque", "linked list", "tree", "graph", "heap", "hash table", "hashtable"},
    "DBMS_CONCEPT": {"normalization", "first normal form", "1nf", "second normal form", "2nf", "third normal form", "3nf", "bcnf", "functional dependency", "primary key", "foreign key", "transaction", "acid"},
    "ML_ALGORITHM": {"logistic regression", "linear regression", "decision tree", "random forest", "svm", "knn", "k nearest neighbors", "naive bayes", "neural network", "k means", "k-means"},
}


@dataclass
class Candidate:
    text: str
    source: str
    score: float = 0.0
    answer_type: str = "UNKNOWN"


def normalize_text(value: object) -> str:
    value = re.sub(r"[^\w\s+.#-]", " ", str(value).casefold(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _canonical(value: object) -> str:
    return normalize_text(value).replace(" ", "")


def _unique(values: Iterable[str]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value).replace("_", " ").strip())
        key = normalize_text(cleaned)
        if key and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def detect_answer_type(answer: str, question: str = "") -> str:
    normalized = normalize_text(answer)
    question_key = normalize_text(question)
    if re.fullmatch(r"(?:19|20)\d{2}", normalized):
        return "YEAR"
    if re.fullmatch(r"\d+(?:\.\d+)?", normalized):
        return "NUMBER"
    for answer_type, terms in TYPE_TERMS.items():
        if normalized in {normalize_text(term) for term in terms}:
            return answer_type
    if question_key.startswith(("who ", "whose ")) or " created " in f" {question_key} ":
        if len(re.findall(r"[A-Z][a-z]+", answer)) >= 2 or " van " in f" {normalized} ":
            return "PERSON"
    if re.search(r"\bwhen\b|\byear\b|\bdate\b", question_key) and re.search(r"\d", normalized):
        return "YEAR"
    if question_key.startswith(("which language", "what language")):
        return "PROGRAMMING_LANGUAGE"
    if question_key.startswith(("which protocol", "what protocol")):
        return "NETWORK_PROTOCOL"
    return "UNKNOWN"


def extract_context_candidates(context: str, question: str = "") -> List[str]:
    candidates = []
    candidates.extend(re.findall(r"\b(?:19|20)\d{2}\b", context))
    candidates.extend(re.findall(r"\b[A-Z][A-Za-z0-9+#.-]*(?:\s+[A-Z][A-Za-z0-9+#.-]*){0,3}\b", context))
    candidates.extend(re.findall(r"\b[A-Za-z][A-Za-z-]*(?:\s+(?:Normal|normal|Search|search|Tree|tree|Regression|regression|Protocol|protocol|Network|network|Model|model|Dependency|dependency)){1,3}\b", context))
    lowered = normalize_text(context)
    for terms in TYPE_TERMS.values():
        for term in terms:
            if normalize_text(term) in lowered:
                candidates.append(term)
    return _unique(candidates)


def get_wordnet_candidates(answer: str, limit: int = 20) -> List[str]:
    candidates = []
    try:
        for synset in wn.synsets(answer.replace(" ", "_"))[:5]:
            for hypernym in synset.hypernyms():
                for hyponym in hypernym.hyponyms():
                    candidates.extend(lemma.name().replace("_", " ") for lemma in hyponym.lemmas())
                    if len(candidates) >= limit:
                        return _unique(candidates)
    except LookupError:
        return []
    return _unique(candidates)


def generate_candidates(question: str, correct_answer: str, retrieved_context: str, extracted_answers: Sequence[str], allow_wordnet: bool = True) -> List[Candidate]:
    candidates = []
    candidates.extend(Candidate(value, "context") for value in extract_context_candidates(retrieved_context, question))
    candidates.extend(Candidate(value, "t5") for value in extracted_answers)
    answer_type = detect_answer_type(correct_answer, question)
    if allow_wordnet and answer_type not in {"PROGRAMMING_LANGUAGE", "DATABASE", "NETWORK_PROTOCOL", "ALGORITHM", "DATA_STRUCTURE", "DBMS_CONCEPT", "ML_ALGORITHM"}:
        candidates.extend(Candidate(value, "wordnet") for value in get_wordnet_candidates(correct_answer))
    return candidates


def filter_candidates(candidates: Sequence[Candidate], correct_answer: str, max_words: int = 8) -> List[Candidate]:
    answer_key = _canonical(correct_answer)
    filtered = []
    seen = {normalize_text(correct_answer)}
    for candidate in candidates:
        text = re.sub(r"\s+", " ", candidate.text.strip())
        key = normalize_text(text)
        if not text or key in seen or _canonical(text) == answer_key:
            continue
        if key in FILLER_VALUES or len(text.split()) > max_words or len(text) > 80:
            continue
        if re.search(r"[<>]|(?:pad|unk|sep)", text, flags=re.IGNORECASE):
            continue
        if len(re.findall(r"\b[A-Z][A-Z0-9+#-]{1,}\b", text)) >= 2:
            continue
        if len(_canonical(text)) >= max(3, len(answer_key) - 1) and (_canonical(text) in answer_key or answer_key in _canonical(text)):
            continue
        seen.add(key)
        candidate.text = text
        filtered.append(candidate)
    return filtered


def calculate_similarity(embedder, left: str, right: str) -> float:
    if embedder is None:
        return 0.0
    vectors = embedder.encode([left, right], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return float(np.dot(vectors[0], vectors[1]))


def _similarities(embedder, texts: Sequence[str]) -> Optional[np.ndarray]:
    if embedder is None or not texts:
        return None
    vectors = embedder.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")


def score_candidate(candidate: Candidate, question: str, correct_answer: str, retrieved_context: str, expected_type: str, embedder=None) -> float:
    candidate.answer_type = detect_answer_type(candidate.text, question)
    type_score = 1.0 if candidate.answer_type == expected_type and expected_type != "UNKNOWN" else (0.45 if candidate.answer_type == "UNKNOWN" else 0.0)
    question_similarity = calculate_similarity(embedder, f"Question: {question} Answer: {candidate.text}", question) if embedder else 0.35
    context_similarity = calculate_similarity(embedder, candidate.text, retrieved_context) if embedder else (0.75 if candidate.source in {"context", "t5"} else 0.25)
    answer_similarity = calculate_similarity(embedder, candidate.text, correct_answer) if embedder else 0.35
    quality = max(0.0, 1.0 - abs(answer_similarity - 0.45) / 0.55)
    if answer_similarity > 0.86:
        quality *= 0.15
    source_bonus = {"context": 0.10, "t5": 0.08, "wordnet": 0.0}.get(candidate.source, 0.0)
    candidate.score = 0.40 * question_similarity + 0.25 * context_similarity + 0.20 * type_score + 0.15 * quality + source_bonus
    return candidate.score


def select_distractors(candidates: Sequence[Candidate], question: str, correct_answer: str, retrieved_context: str, embedder=None, count: int = 3) -> List[str]:
    expected_type = detect_answer_type(correct_answer, question)
    ranked = sorted(
        (candidate for candidate in candidates if score_candidate(candidate, question, correct_answer, retrieved_context, expected_type, embedder) > 0.25),
        key=lambda candidate: candidate.score,
        reverse=True,
    )
    selected = []
    selected_vectors = []
    for candidate in ranked:
        if len(selected) >= count:
            break
        if embedder and selected_vectors:
            vector = _similarities(embedder, [candidate.text])[0]
            if max(float(np.dot(vector, previous)) for previous in selected_vectors) > 0.90:
                continue
        selected.append(candidate.text)
        if embedder:
            selected_vectors.append(_similarities(embedder, [candidate.text])[0])
    return selected


def validate_mcq(question: str, correct_answer: str, distractors: Sequence[str], embedder=None, context: str = "") -> dict:
    issues = []
    options = [correct_answer, *distractors]
    normalized = [normalize_text(option) for option in options]
    if len(distractors) != 3:
        issues.append("expected exactly 3 distractors")
    if len(options) != 4:
        issues.append("expected exactly 4 options")
    if len(set(normalized)) != len(normalized):
        issues.append("options contain duplicates")
    if normalized.count(normalize_text(correct_answer)) != 1:
        issues.append("correct answer must occur exactly once")
    if any(value in FILLER_VALUES for value in normalized):
        issues.append("filler option detected")
    expected_type = detect_answer_type(correct_answer, question)
    type_matches = sum(detect_answer_type(value, question) == expected_type for value in distractors)
    if expected_type != "UNKNOWN" and type_matches < 2:
        issues.append("fewer than two distractors match the answer type")
    similarities = []
    if embedder and distractors:
        similarities = [calculate_similarity(embedder, correct_answer, distractor) for distractor in distractors]
        if any(value > 0.88 for value in similarities):
            issues.append("distractor is too semantically close to the answer")
        if context and any(calculate_similarity(embedder, distractor, context) < 0.05 for distractor in distractors):
            issues.append("distractor is unrelated to retrieved context")
    lengths = [len(normalize_text(option).split()) for option in options]
    if lengths and max(lengths) > max(8, min(lengths) * 4):
        issues.append("options are not reasonably balanced")
    quality_score = max(0.0, min(1.0, 1.0 - len(issues) / 8))
    if similarities:
        quality_score = max(0.0, min(1.0, quality_score + sum(min(value, 0.8) for value in similarities) / len(similarities) * 0.1))
    return {"valid": not issues, "quality_score": round(quality_score, 2), "issues": issues}


def build_validated_mcq(question: str, correct_answer: str, retrieved_context: str, extracted_answers: Sequence[str], embedder=None) -> tuple[List[str], dict]:
    candidates = generate_candidates(question, correct_answer, retrieved_context, extracted_answers)
    candidates = filter_candidates(candidates, correct_answer)
    expected_type = detect_answer_type(correct_answer, question)
    ranked = sorted(
        candidates,
        key=lambda candidate: score_candidate(
            candidate, question, correct_answer, retrieved_context, expected_type, embedder
        ),
        reverse=True,
    )
    best_distractors = select_distractors(candidates, question, correct_answer, retrieved_context, embedder)
    best_validation = validate_mcq(question, correct_answer, best_distractors, embedder, retrieved_context)
    if best_validation["valid"]:
        return best_distractors, best_validation

    # Try a bounded number of alternate combinations before reporting low confidence.
    for combination in list(combinations(ranked[:10], 3))[:40]:
        distractors = [candidate.text for candidate in combination]
        validation = validate_mcq(question, correct_answer, distractors, embedder, retrieved_context)
        if validation["valid"]:
            return distractors, validation
        if validation["quality_score"] > best_validation["quality_score"]:
            best_distractors, best_validation = distractors, validation
    return best_distractors, best_validation
