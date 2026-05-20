from __future__ import annotations

import re
from typing import Any, List, Sequence, Tuple


DROP_DEPS = {"det", "poss"}
LEFT_MOD_DEPS = {"amod", "compound", "nummod", "poss"}
RIGHT_MOD_DEPS = {"prep", "acl", "relcl"}
SUBORD_NOUN_DEPS = {"compound", "poss", "nmod", "appos"}
TARGET_OBJECT_DEPS = {"dobj", "obj", "attr"}
VERB_ATTACHED_TARGET_PREPS = {
    "above",
    "at",
    "behind",
    "below",
    "beside",
    "between",
    "in",
    "inside",
    "near",
    "next",
    "on",
    "over",
    "under",
    "with",
}
SUPERLATIVE_DIRECTIONS = ("left", "right", "front", "back", "top", "bottom")
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


def _normalize_directional_superlatives(text: str) -> str:
    out = text
    directions = "|".join(SUPERLATIVE_DIRECTIONS)
    out = re.sub(rf"\b({directions})\s+most\b", lambda match: f"{match.group(1)}most", out, flags=re.IGNORECASE)
    out = re.sub(rf"\bmost\s+({directions})\b", lambda match: f"{match.group(1)}most", out, flags=re.IGNORECASE)
    return out


def _normalize_relation_words(text: str) -> str:
    out = text
    out = re.sub(r"\bneer\b", "near", out, flags=re.IGNORECASE)
    out = re.sub(r"\bnext\s+to\b", "near", out, flags=re.IGNORECASE)
    out = re.sub(r"\bclose\s+to\b", "near", out, flags=re.IGNORECASE)
    out = re.sub(r"\bnearest\b", "near", out, flags=re.IGNORECASE)
    out = re.sub(r"\bbesides\b", "beside", out, flags=re.IGNORECASE)
    return out


def _normalize_text(text: str) -> str:
    return _normalize_directional_superlatives(_normalize_relation_words(text))


def _replace_one_reference(phrase: str, noun_text: str) -> str:
    noun = (noun_text or "").strip().lower()
    if not noun:
        return phrase
    return re.sub(r"\b([a-z][a-z0-9_-]*)\s+one\b", rf"\1 {noun}", phrase, flags=re.IGNORECASE)


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
    def _prep_has_prior_target_object(prep) -> bool:
        if prep.pos_ != "ADP" or prep.dep_ != "prep":
            return False
        head = prep.head
        if head.pos_ not in ("VERB", "AUX"):
            return False
        for child in head.children:
            if child.i >= prep.i:
                continue
            if child.pos_ in ("NOUN", "PROPN") and child.dep_ in TARGET_OBJECT_DEPS:
                return True
        return False

    @staticmethod
    def _inside_target_relation(tok) -> bool:
        for ancestor in tok.ancestors:
            if ancestor.pos_ == "ADP" and ancestor.dep_ == "prep":
                if InstructionPhraseExtractor._prep_has_prior_target_object(ancestor):
                    return True
        return False

    @staticmethod
    def _is_subordinate_noun(tok) -> bool:
        if tok.pos_ not in ("NOUN", "PROPN"):
            return False

        if tok.dep_ in SUBORD_NOUN_DEPS and tok.head.pos_ in ("NOUN", "PROPN"):
            return True

        if tok.dep_ == "pobj" and tok.head.pos_ == "ADP":
            if tok.head.head.pos_ in ("NOUN", "PROPN"):
                return True
            if InstructionPhraseExtractor._prep_has_prior_target_object(tok.head):
                return True

        if InstructionPhraseExtractor._inside_target_relation(tok):
            return True

        return False

    @staticmethod
    def _verb_attached_target_preps(noun) -> List[Any]:
        if noun.dep_ not in TARGET_OBJECT_DEPS:
            return []
        head = noun.head
        if head.pos_ not in ("VERB", "AUX"):
            return []

        preps = []
        for child in head.rights:
            if child.i <= noun.i:
                continue
            if child.dep_ != "prep" or child.pos_ != "ADP":
                continue
            if child.text.lower() not in VERB_ATTACHED_TARGET_PREPS:
                continue
            preps.append(child)
        return preps

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
        text = _normalize_text(text)
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

            for prep in self._verb_attached_target_preps(noun):
                for token in prep.subtree:
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
            phrase = _replace_one_reference(phrase, noun.text)
            if phrase:
                phrases.append(phrase.lower())

        return _dedupe_keep_order(phrases), _dedupe_keep_order(nouns)

    @staticmethod
    def _extract_fallback(text: str) -> Tuple[List[str], List[str]]:
        text = _normalize_text(text)
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
        if phrases:
            terms = _dedupe_keep_order(phrases)
        else:
            terms = _dedupe_keep_order([*nouns, *[term.lower() for term in base_terms]])
        return phrases, terms
