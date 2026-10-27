import re
from rapidfuzz import fuzz
from chunker import SENTENCE_SPLIT


def find_best_sentence(answer: str, chunk_text: str, n: int, threshold: float = 70) -> tuple[str, float] | None:
    """Return the sentence in the chunk that the answer's [n] claims most likely used, with its score."""
    claim_sentences = [s for s in re.split(SENTENCE_SPLIT, answer) if f"[{n}]" in s]
    claim = " ".join(claim_sentences) if claim_sentences else answer
    claim = re.sub(r"\[\d+\]", "", claim)

    chunk_sentences = [s for s in re.split(SENTENCE_SPLIT, chunk_text) if len(s) > 20]
    if not chunk_sentences:
        return None

    score, sentence = max((fuzz.token_set_ratio(claim, s), s) for s in chunk_sentences)
    return (sentence, score) if score >= threshold else None

