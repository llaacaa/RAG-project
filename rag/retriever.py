import chromadb
from embeddings import embed


def retrieve(
    question: str, k: int = 5, collection: chromadb.Collection = None
) -> list[dict]:

    if collection is None:
        collection = get_collection()

    embedding = embed([question], kind="query")[0]

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {"text": doc, "distance": dist, **meta}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
