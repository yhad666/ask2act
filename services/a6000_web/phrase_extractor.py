from __future__ import annotations

import re
from typing import Any, List, Sequence, Tuple


DROP_DEPS = {"det", "poss"}
LEFT_MOD_DEPS = {"amod", "compound", "nummod", "poss"}
RIGHT_MOD_DEPS = {"prep", "acl", "relcl"}
SUBORD_NOUN_DEPS = {"compound", "poss", "nmod", "appos"}
FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "give",
    "grab",
    "hand",
    "his",
    "hold",
    "i",
    "it",
    "me",
    "my",
    "of",
    "on",
    "our",
    "pick",
    "please",
    "take",
    "that",
    "the",
    "their",
    "this",
    "to",
    "up",
    "us",
    "your",
}


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = (item or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


class InstructionPhraseExtractor:
    def __init__(self) -> None:
        self._nlp = None

    def _load_nlp(self):
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")
        except Exception:
            self._nlp = False
        return self._nlp

    @staticmethod
    def _is_subordinate_noun(tok) -> bool:
        if tok.pos_ not in ("NOUN", "PROPN"):
            return False

        if tok.dep_ in SUBORD_NOUN_DEPS and tok.head.pos_ in ("NOUN", "PROPN"):
            return True

        if tok.dep_ == "pobj" and tok.head.pos_ == "ADP":
            if tok.head.head.pos_ in ("NOUN", "PROPN"):
                return True

        return False

    @staticmethod
    def _clean_tokens(tokens) -> List[Any]:
        out = []
        for token in tokens:
            if token.is_punct:
                continue
            if token.dep_ in DROP_DEPS:
                continue
            out.append(token)
        return out

    def _extract_with_spacy(self, text: str) -> Tuple[List[str], List[str]]:
        doc = self._load_nlp()(text)
        phrases: List[str] = []
        nouns: List[str] = []

        for noun in doc:
            if noun.pos_ not in ("NOUN", "PROPN"):
                continue
            if self._is_subordinate_noun(noun):
                continue

            nouns.append(noun.text.lower())
            tokens = {noun}

            for child in noun.lefts:
                if child.dep_ in LEFT_MOD_DEPS:
                    for token in child.subtree:
                        tokens.add(token)

            for child in noun.rights:
                if child.dep_ in RIGHT_MOD_DEPS:
                    for token in child.subtree:
                        tokens.add(token)

            if noun.dep_ in {"nsubj", "nsubjpass"} and noun.head.pos_ in {"VERB", "AUX"}:
                head = noun.head
                complements = [child for child in head.children if child.dep_ in {"acomp", "attr"}]
                expanded = []
                for comp in complements:
                    expanded.append(comp)
                    expanded += list(comp.conjuncts)
                for adj in expanded:
                    if adj.pos_ == "ADJ":
                        for token in adj.subtree:
                            tokens.add(token)

            tokens_list = sorted(self._clean_tokens(tokens), key=lambda item: item.i)
            phrase = " ".join(token.text for token in tokens_list).strip()
            if phrase:
                phrases.append(phrase.lower())

        return _dedupe_keep_order(phrases), _dedupe_keep_order(nouns)

    @staticmethod
    def _extract_fallback(text: str) -> Tuple[List[str], List[str]]:
        words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.lower())
        filtered = [word for word in words if word not in FALLBACK_STOPWORDS]
        if not filtered:
            return [], []
        phrase = " ".join(filtered)
        return [phrase], [filtered[-1]]

    def extract_np_like_phrases(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        if self._load_nlp():
            phrases, _ = self._extract_with_spacy(text)
            return phrases
        phrases, _ = self._extract_fallback(text)
        return phrases

    def extract_terms(self, text: str, base_terms: Sequence[str]) -> Tuple[List[str], List[str]]:
        text = (text or "").strip()
        phrases: List[str]
        nouns: List[str]
        if text and self._load_nlp():
            phrases, nouns = self._extract_with_spacy(text)
        else:
            phrases, nouns = self._extract_fallback(text)
        terms = _dedupe_keep_order([*phrases, *nouns, *[term.lower() for term in base_terms]])
        return phrases, terms
