"""
Measure how well the RAG pipeline works on a fixed set of questions.

Retrieval and answers are measured separately, so a wrong answer tells you
which half failed:
  - the right page wasn't retrieved  -> a retrieval problem (chunking, embeddings, k)
  - it was retrieved but the answer is wrong -> a generation problem (prompt, model)

Each question in eval/questions.json has:
  question         the question to ask
  expected_pages   pages where the answer really is ([] = the documents can't answer it)
  answer_contains  the answer passes if it contains ANY of these (case-insensitive)

Usage:
  python rag/evaluate.py                    # full run (retrieval + answers)
  python rag/evaluate.py --retrieval-only   # fast: skip the LLM
  python rag/evaluate.py -k 8 --label "k=8" # try a setting, saved for comparison
"""

import argparse
import json
import time
from datetime import datetime

from config import CHAT_MODEL, CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, ROOT
from generator import build_messages, generate
from retriever import retrieve
from store import get_collection

QUESTIONS_FILE = ROOT / "eval" / "questions.json"
RESULTS_DIR = ROOT / "eval" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers on eval/questions.json.")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip the LLM, only measure retrieval")
    parser.add_argument("--label", default="", help="A note saved with the results, e.g. 'chunk 400'")
    return parser.parse_args()


def retrieval_rank(hits: list[dict], expected_pages: list[int]) -> int | None:
    """Rank (1 = best) of the first hit on an expected page, or None if none was retrieved."""
    for rank, hit in enumerate(hits, start=1):
        if hit["page_number"] in expected_pages:
            return rank
    return None


def answer_passes(answer: str, answer_contains: list[str]) -> bool:
    # Models use both ' and ’ in "couldn't", so normalise before comparing
    normalised = answer.replace("’", "'").lower()
    return any(keyword.replace("’", "'").lower() in normalised for keyword in answer_contains)


def evaluate_question(item: dict, k: int, collection, retrieval_only: bool) -> dict:
    start = time.perf_counter()
    hits = retrieve(item["question"], k=k, collection=collection)
    answerable = bool(item["expected_pages"])

    result = {
        "question": item["question"],
        "answerable": answerable,
        "retrieved_pages": [hit["page_number"] for hit in hits],
        "best_distance": round(hits[0]["distance"], 3) if hits else None,
        "rank": retrieval_rank(hits, item["expected_pages"]) if answerable else None,
    }

    if not retrieval_only:
        answer = "".join(generate(build_messages(item["question"], hits)))
        result["answer"] = answer
        result["answer_ok"] = answer_passes(answer, item["answer_contains"])

    result["seconds"] = round(time.perf_counter() - start, 2)
    return result


def print_row(result: dict, retrieval_only: bool):
    if result["answerable"]:
        retrieval = f"rank {result['rank']}" if result["rank"] else "MISSED"
    else:
        retrieval = f"n/a ({result['best_distance']})"  # no right page; distance hints at "not found"

    line = f"{retrieval:<14}"
    if not retrieval_only:
        line += f"{'OK' if result['answer_ok'] else 'FAIL':<8}"
    line += result["question"]
    print(line)

    if not retrieval_only and not result["answer_ok"]:
        print(f"{'':22}answer: {result['answer'][:160]!r}")
    if result["answerable"] and not result["rank"]:
        print(f"{'':22}retrieved pages: {result['retrieved_pages']}")


def summarise(results: list[dict], k: int, retrieval_only: bool) -> dict:
    answerable = [r for r in results if r["answerable"]]
    ranks = [r["rank"] for r in answerable]

    summary = {
        f"hit@{k}": sum(rank is not None for rank in ranks),
        "hit@1": sum(rank == 1 for rank in ranks),
        # Mean reciprocal rank: 1.0 if the right page is always first, 0.5 if always second, ...
        "mrr": round(sum(1 / rank for rank in ranks if rank) / len(ranks), 3) if ranks else 0,
        "answerable_questions": len(answerable),
    }
    if not retrieval_only:
        unanswerable = [r for r in results if not r["answerable"]]
        summary["answers_ok"] = sum(r["answer_ok"] for r in answerable)
        summary["refused_correctly"] = sum(r["answer_ok"] for r in unanswerable)
        summary["unanswerable_questions"] = len(unanswerable)
    summary["avg_seconds"] = round(sum(r["seconds"] for r in results) / len(results), 2)
    return summary


def print_summary(summary: dict, k: int, retrieval_only: bool):
    n = summary["answerable_questions"]
    print("\n" + "=" * 60)
    print(f"Retrieval  hit@{k}: {summary[f'hit@{k}']}/{n}   hit@1: {summary['hit@1']}/{n}   MRR: {summary['mrr']}")
    if not retrieval_only:
        print(f"Answers    correct: {summary['answers_ok']}/{n}")
        print(f"Refusals   'couldn't find' when it should: "
              f"{summary['refused_correctly']}/{summary['unanswerable_questions']}")
    print(f"Speed      {summary['avg_seconds']}s per question")


if __name__ == "__main__":
    args = parse_args()
    collection = get_collection()
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    settings = {
        "label": args.label,
        "k": args.top_k,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embed_model": EMBED_MODEL,
        "chat_model": None if args.retrieval_only else CHAT_MODEL,
    }
    print(f"Settings: {settings}\n")
    print(f"{'retrieval':<14}{'' if args.retrieval_only else 'answer':<8}question")
    print("-" * 60)

    results = []
    for item in questions:
        result = evaluate_question(item, args.top_k, collection, args.retrieval_only)
        print_row(result, args.retrieval_only)
        results.append(result)

    summary = summarise(results, args.top_k, args.retrieval_only)
    print_summary(summary, args.top_k, args.retrieval_only)

    # Save everything, so runs with different settings can be compared later
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.json"
    out_file.write_text(
        json.dumps({"settings": settings, "summary": summary, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved to {out_file.relative_to(ROOT)}")
