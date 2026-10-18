import re

def overlap_tail(previous_chunk: str, overlap: int) -> str:
    """Return the last `overlap` characters of a chunk, starting at a whole word."""
    if overlap <= 0 or not previous_chunk:
        return ""
    tail = previous_chunk[-overlap:]
    return tail[tail.find(" ") + 1:] + " "


def process_chunk(current_text:str, paragraph:str, size:int, overlap:int, chunks:list[str], separator:str) -> str:
    """Process a paragraph or sentence and add it to the current text or create a new chunk if needed."""
    if len(current_text) + len(paragraph) <= size:
        current_text += paragraph + separator
    else:
        if current_text:
            chunks.append(overlap_tail(chunks[-1] if chunks else "", overlap) + current_text.strip())
        current_text = paragraph + separator
    return current_text


def hard_split(text: str, size: int) -> list[str]:
    """Cut sentences into pieces of at most `size` characters, at spaces."""
    pieces, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        pieces.append(current)
    return pieces


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """
    Splits the text into chunks of a given size with a given overlap.
    
    Args:
        text (str): The text to split.
        size (int): The maximum size of each chunk.
        overlap (int): The number of characters to overlap between chunks.

    """
    paragraphs = text.split("\n\n")
    current_text = ""
    chunks = []

    for paragraph in paragraphs:
        if len(paragraph) > size:
            # Handle long paragraphs by splitting them into sentences
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            for sentence in sentences:
                for split_sentence in hard_split(sentence, size):
                    current_text = process_chunk(current_text, split_sentence, size, overlap, chunks, " ")
            current_text = current_text.rstrip() + "\n\n"
        else:
            current_text = process_chunk(current_text, paragraph, size, overlap, chunks, "\n\n")
    if current_text:
        chunks.append(overlap_tail(chunks[-1] if chunks else "", overlap) + current_text.strip())

    return chunks


def chunk_pages(pages: list[dict], size: int = 800, overlap: int = 150) -> list[dict]:
    chunks = []
    chunk_index = 0
    for page in pages:
        for chunk_text in split_text(page["text"], size, overlap):
            chunks.append({**page, "text": chunk_text, "chunk_index": chunk_index})
            chunk_index += 1

    return chunks


if __name__ == "__main__":
    # (description, text, size, overlap)
    tests = [
        ("long paragraph -> sentences",
         "This is a sentence. " * 20, 100, 0),
        ("very long sentence -> cut at words",
         "One very long sentence " * 10 + "end.", 60, 0),
        ("sentences split at line breaks",
         "Line one ends here.\nLine two of the same paragraph keeps going for a while.", 40, 0),
        ("overlap between chunks",
         "Paragraph one is here and it is about rent.\n\nParagraph two is about the expiry date of the lease.", 60, 20),
        ("paragraph break kept after a long paragraph",
         "A long paragraph sentence one here. And sentence two follows it now.\n\nShort next.", 60, 0),
    ]

    for description, text, size, overlap in tests:
        print(f"--- {description} (size={size}, overlap={overlap})")
        for chunk in split_text(text, size, overlap):
            print(f"{len(chunk):4} {chunk!r}")


    from loaders import load_file

    pages = load_file("docs/your_file.pdf")    

    for size in (400, 800, 1200):
        chunks = chunk_pages(pages, size=size, overlap=150)
        print(f"\n=== size={size}: {len(pages)} pages -> {len(chunks)} chunks")
        for c in chunks[:3]:
            print(f"  #{c['chunk_index']} page {c['page_number']} ({len(c['text'])} chars): {c['text'][:80]!r}")

