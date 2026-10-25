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

    args = parse_args()

    collection = get_collection()

    if args.retrieve_only:
        from retriever import retrieve

        hits = retrieve(args.question, k=args.top_k, collection=collection)

        for rank, hit in enumerate(hits, start=1):
            print(
                f"#{rank}  {hit['distance']:.3f}  {hit['source']}, {format_location(hit)}"
            )
            print(f"    {hit['text'][:200]!r}\n")
