import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question over your indexed documents."
    )
    parser.add_argument("question", help="The question to ask")
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Only show retrieved chunks, don't call the LLM",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=5, help="Number of chunks to retrieve"
    )
    return parser.parse_args()


if __name__ == "__main__":
    from store import get_collection
    from loaders import format_location
    from generator import build_messages, generate
    from retriever import retrieve
    import re

    args = parse_args()

    collection = get_collection()

    question = args.question

    hits = retrieve(question, k=args.top_k, collection=collection)
    if args.retrieve_only:

        for rank, hit in enumerate(hits, start=1):
            print(
                f"#{rank}  {hit['distance']:.3f}  {hit['source']}, {format_location(hit)}"
            )
            print(f"    {hit['text'][:200]!r}\n")
    else:
        from highlight import find_best_sentence
        from rich.console import Console
        from rich.text import Text

        console = Console()
        messages = build_messages(question, hits)

        answer = ""
        for piece in generate(messages):
            answer += piece
            print(piece, end="", flush=True)
        print()

        source_citations = sorted(
            {
                int(n)
                for n in re.findall(r"\[(\d+)\]", answer)
                if 1 <= int(n) <= len(hits)
            }
        )

        if source_citations:
            print("\nSources:")
            for n in source_citations:
                hit = hits[n - 1]
                print(f"[{n}] {hit['source']}, {format_location(hit)}")

                match = find_best_sentence(answer, hit["text"])
                if match:
                    sentence, score = match

                    start = hit["text"].find(sentence)
                    snippet = Text(hit["text"][max(0, start - 150): start + len(sentence) + 150])


                    snippet.highlight_words([sentence], style="bold yellow")
                    
                    console.print(snippet)
                    print()
