import ollama
from config import CHAT_MODEL
from loaders import format_location


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    system_prompt = """You answer questions about the user's documents using ONLY the numbered sources provided.
    - Cite every fact with the source number in square brackets, like [1] or [2][3].
    - If the sources do not contain the answer, reply exactly: "I couldn't find that in your documents."
    - Do not use outside knowledge. Do not guess.
    - Put citations at the end of the sentence they support.
    - Be concise.
    """

    user_prompt = "SOURCES:\n"
    for i, hit in enumerate(hits, start=1):
        user_prompt += (
            f"[{i}] ({hit['source']}, {format_location(hit)})\n\"{hit['text']}\"\n\n"
        )

    user_prompt += f"QUESTION: {question}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate(messages: list[dict]):
    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
        stream=True,
        options={"temperature": 0, "num_ctx": 8192},
    )

    for chunk in stream:
        piece = chunk.message.content
        if piece:
            yield piece
