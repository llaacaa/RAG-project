import re
from rapidfuzz import fuzz
from chunker import SENTENCE_SPLIT


def find_best_sentence(answer: str, chunk_text: str, threshold: float = 70) -> tuple[str, float] | None:
    """Return the sentence in the chunk that the answer most likely used, with its score."""
    claim = re.sub(r"\[\d+\]", "", answer)
    sentences = [s for s in re.split(SENTENCE_SPLIT, chunk_text) if len(s) > 20]
    if not sentences:
        return None
    score, sentence = max((fuzz.token_set_ratio(claim, s), s) for s in sentences)
    return (sentence, score) if score >= threshold else None
