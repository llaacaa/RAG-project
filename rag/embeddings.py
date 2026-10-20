import ollama
from config import EMBED_MODEL


def embed(texts: list[str], kind: str) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using the Ollama API.

    Args:
        texts (list[str]): A list of texts to embed.
        kind (str): "document" (chunks being stored) or "query" (the user's question).

    """
    if kind not in ["document", "query"]:
        raise ValueError("kind must be either 'document' or 'query'")

    prefix = "search_document: " if kind == "document" else "search_query: "
    response = ollama.embed(model=EMBED_MODEL, input=[prefix + t for t in texts])
    
    return response.embeddings


if __name__ == "__main__":
    sentences = [
        "The lease expires on 31 Dec 2027",
        "The term ends at the end of 2027",
        "Tenants may keep one cat",
        "When does the contract end?",
    ]

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b)) / (
            sum(x * x for x in a) ** 0.5 * sum(y * y for y in b) ** 0.5
        )

    document_embeddings = embed(sentences[:3], kind="document")
    query_embedding = embed([sentences[3]], kind="query")[0]

    embeddings = []
    embeddings.extend(document_embeddings)
    embeddings.append(query_embedding)

    for i in range(len(embeddings)):
        for j in range(0, len(embeddings)):
            print(
                f'Similarity between sentence "{sentences[i]}" and sentence "{sentences[j]}" = {cosine(embeddings[i], embeddings[j]):.4f}'
            )
        print("\n")
