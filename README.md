Local RAG (Retrieval-Augmented Generation) app. You drop in leases, contracts and similar files, ask questions like *"When does the contract expire?"*, and get an answer **plus the exact place it came from** (file, page, quoted text). Stack: Python, a local LLM, ChromaDB.

- **Formats:** text PDFs, .docx, .txt/.md (no OCR)
- **Hardware:** small GPU or CPU only, so small models (3–4B)
- **No LangChain/LlamaIndex.** You build every piece by hand.