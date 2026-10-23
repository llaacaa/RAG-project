import chromadb
from config import CHROMA_DIR, COLLECTION_NAME


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    collection: chromadb.Collection = None,
):
    if collection is None:
        collection = get_collection()

    collection.upsert(
        ids=[f"{chunk['file_hash']}_{chunk['chunk_index']}" for chunk in chunks],
        embeddings=embeddings,
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {key: value for key, value in chunk.items() if key != "text"}
            for chunk in chunks
        ],
    )


def is_indexed(collection: chromadb.Collection, file_hash: str, source: str) -> bool:
    """
    Check if a file is already indexed in the collection by its hash and source.

    Args:
        collection (chromadb.Collection): The ChromaDB collection.
        file_hash (str): The SHA256 hash of the file.
        source (str): The source name of the file.

    Returns:
        bool: True if the file is indexed, False otherwise.
    """

    result = collection.get(
        where={"$and": [{"file_hash": file_hash}, {"source": source}]},
        limit=1,
    )

    return len(result["ids"]) > 0


def delete_source(collection: chromadb.Collection, source: str):
    """
    Delete all documents in the collection that match the given source.

    Args:
        collection (chromadb.Collection): The ChromaDB collection.
         source (str): The source name of the documents to delete.
    """
    collection.delete(where={"source": source})


def list_sources(collection: chromadb.Collection) -> set[str]:
    """
    List all unique sources in the collection.

    Args:
        collection (chromadb.Collection): The ChromaDB collection.

    Returns:
        set: A set of unique source names.
    """
    result = collection.get(include=["metadatas"])
    sources = {
        metadata["source"] for metadata in result["metadatas"] if "source" in metadata
    }
    return sources
