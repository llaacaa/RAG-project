import hashlib
import chromadb
from pathlib import Path
from store import is_indexed, add_chunks, delete_source, list_sources
from loaders import load_file
from chunker import chunk_pages
from embeddings import embed
from config import CHUNK_SIZE, CHUNK_OVERLAP
from typing import Literal, TypeAlias

INGEST_RESULT: TypeAlias = Literal["skipped", "indexed", "empty", "unsupported"]


def get_hash(file: str) -> str:
    """Get the SHA256 hash of a file."""

    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def ingest_file(path: str, collection: chromadb.Collection) -> INGEST_RESULT:
    file_hash = get_hash(path)
    source = Path(path).name
    if is_indexed(collection, file_hash, source):
        return "skipped"

    try:
        pages = load_file(path)
    except ValueError as error:
        print(f"Skipping {Path(path).name}: {error}")
        return "unsupported"

    delete_source(collection, source)

    chunks: list[dict] = chunk_pages(pages, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    if not chunks:
        return "empty"

    embeddings = embed([chunk["text"] for chunk in chunks], "document")
    add_chunks(
        chunks=[chunk | {"file_hash": file_hash} for chunk in chunks],
        embeddings=embeddings,
        collection=collection,
    )

    return "indexed"


def clean_files_from_folder(seen_files: set[str], collection: chromadb.Collection) -> list[str]:
    deleted_sources = []
    for source in list_sources(collection):
        if source not in seen_files:
            delete_source(collection, source)
            deleted_sources.append(source)
    return deleted_sources


def ingest_folder(
    folder_path: str, collection: chromadb.Collection
) -> tuple[dict[str, INGEST_RESULT], list[str]]:
    results = {}
    seen_sources = set()
    for file in Path(folder_path).iterdir():
        if file.is_file():
            result = ingest_file(str(file), collection)
            results[file.name] = result
            seen_sources.add(file.name)
    deleted_sources = clean_files_from_folder(seen_sources, collection)
    return results, deleted_sources


if __name__ == "__main__":
    from store import get_collection
    from config import COLLECTION_NAME

    collection = get_collection()
    print(f"BEFORE: Collection '{COLLECTION_NAME}' has {collection.count()} documents.")

    folder_path = "docs"
    results, deleted_sources = ingest_folder(folder_path, collection)

    for file_name, result in results.items():
        print(f"{file_name}: {result}")
    print(f" AFTER: Collection '{COLLECTION_NAME}' has {collection.count()} documents.")
    if deleted_sources:
        print(f"Deleted sources: {', '.join(deleted_sources)}")
