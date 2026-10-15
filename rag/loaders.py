import pymupdf as fitz
from pathlib import Path
import docx
import unicodedata
import re

type PageContent = list[dict]

# Every entry returned by a loader has the same keys:
#   page_number: real page number (PDF), or -1 when the format has no pages
#   section:     heading the text belongs to (DOCX), "Full text" (TXT/MD), or "" (PDF)
#   text:        the extracted text
#   source:      file name, for display ("lease.pdf")
#   path:        full path, for reopening the file later ("docs/lease.pdf")


def load_pdf(file_path: str) -> PageContent:
    """
    Load a PDF file and return one entry per page.

    Args:
        file_path (str): The path to the PDF file.
    """

    page_content = []

    with fitz.open(file_path) as pdf_document:
        # Page numbers start at 1, the way people count them
        for page_num, page in enumerate(pdf_document, start=1):
            page_content.append(
                {
                    "page_number": page_num,
                    "section": "",
                    "text": clean_text(page.get_text()),
                    "source": Path(file_path).name,
                    "path": str(file_path),
                }
            )

    return page_content


def load_txt(file_path: str) -> PageContent:
    """
    Load a TXT or MD file and return it as a single entry.

    Args:
        file_path (str): The path to the text file.
    """

    text = Path(file_path).read_text(encoding="utf-8")

    return [
        {
            "page_number": -1,
            "section": "Full text",
            "text": clean_text(text),
            "source": Path(file_path).name,
            "path": str(file_path),
        }
    ]


def load_docx(file_path: str) -> PageContent:
    """
    Load a DOCX file and return one entry per section.

    DOCX files don't store pages, so the text is split at headings instead.
    Text before the first heading goes into a section called "Start".

    Args:
        file_path (str): The path to the DOCX file.
    """

    doc = docx.Document(file_path)

    page_content = []
    current_section = "Start"
    current_text = []  # the "basket" of paragraphs for the current section

    def save_section():
        # Only save if the basket has something in it
        if current_text:
            page_content.append(
                {
                    "page_number": -1,
                    "section": current_section,
                    "text": clean_text("\n\n".join(current_text)),
                    "source": Path(file_path).name,
                    "path": str(file_path),
                }
            )

    for paragraph in doc.paragraphs:
        text = paragraph.text

        # Skip empty paragraphs (blank lines in Word)
        if not text:
            continue

        if paragraph.style.name.startswith("Heading"):
            # A new heading: save the previous section, then start a new one
            save_section()
            current_section = text
            current_text = []
        else:
            current_text.append(text)

    # The last section has no heading after it, so save it here
    save_section()

    return page_content


LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".md": load_txt,
    ".docx": load_docx,
}


def load_file(file_path: str) -> PageContent:
    """
    Load a file (PDF, TXT, MD or DOCX) by picking the right loader for its extension.

    Args:
        file_path (str): The path to the file.
    """

    suffix = Path(file_path).suffix.lower()
    loader = LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loader(file_path)


def format_location(entry: dict) -> str:
    """
    Return a human-readable location for an entry: "page 3" or a section name.
    """

    if entry["page_number"] != -1:
        return f"page {entry['page_number']}"
    return entry["section"]


def clean_text(text: str) -> str:
    """
    Clean extracted text while keeping paragraph breaks.

    Args:
        text (str): The text to clean.
    """

    # Turn ligatures and odd characters into normal ones ("ﬁ" -> "fi")
    text = unicodedata.normalize("NFKC", text)

    # Remove invisible control characters, but keep newlines and tabs
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c)[0] != "C")

    # Rejoin words split across lines with a hyphen ("agree-\nment" -> "agreement")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Collapse runs of spaces and tabs into one space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces at the start and end of each line
    text = re.sub(r" *\n *", "\n", text)

    # Allow at most one blank line between paragraphs
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


if __name__ == "__main__":
    # Load every file in docs/ and show a short preview of each
    for file in Path("docs").iterdir():
        try:
            pages = load_file(file)
        except ValueError as error:
            print(f"Skipping {file.name}: {error}")
            continue

        print(f"{file.name} -> {len(pages)} entries")
        for entry in pages[:3]:
            print(f"  [{format_location(entry)}] {entry['text'][:100]!r}")
        print("---")
