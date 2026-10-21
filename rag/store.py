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
        ids=[f"{chunk['source']}_{chunk['chunk_index']}" for chunk in chunks],
        embeddings=embeddings,
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{key: value for key, value in chunk.items() if key != "text"} for chunk in chunks]
    )



if __name__ == "__main__":
    from loaders import load_file
    from pathlib import Path
    from chunker import chunk_pages
    from embeddings import embed

    collection = get_collection()
    print(f"Collection '{COLLECTION_NAME}' has {collection.count()} documents.")

    for file in Path("docs").iterdir():
        try:
            pages = load_file(file)
        except ValueError as error:
            print(f"Skipping {file.name}: {error}")
            continue

        print(f"{file.name} -> {len(pages)} pages\n\n")

        chunks: list[dict] = chunk_pages(pages, size=800, overlap=150)

        embeddings = embed([chunk["text"] for chunk in chunks], "document")

        add_chunks(chunks, embeddings, collection=collection)

        print(
            f"{collection.count()} documents in collection after adding {len(chunks)} chunks from {file.name}.\n\n"
        )
