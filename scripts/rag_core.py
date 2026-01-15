import os
from typing import List, Tuple, Optional

from dotenv import load_dotenv

import chromadb
from openai import OpenAI


# -----------------------------
# Local-only .env loading
# (Render uses dashboard env vars)
# -----------------------------
if os.path.exists(".env"):
    load_dotenv()


# -----------------------------
# Config (must match ingest)
# -----------------------------
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vector_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "combatives_notes")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Ensure persistence directory exists (Render: should be a mounted disk path)
os.makedirs(PERSIST_DIR, exist_ok=True)

# OpenAI client (one per process)
_client = OpenAI()

# Cache Chroma client/collection per process
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_collection():
    """
    Returns the Chroma collection. Creates the client/collection once per process.
    """
    global _chroma_client, _collection

    if _collection is not None:
        return _collection

    _chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    _collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def _collection_count_safe() -> Optional[int]:
    """
    Returns the number of items in the collection if available.
    Chroma supports count() on collections; if any error occurs, return None.
    """
    try:
        col = _get_collection()
        return int(col.count())
    except Exception:
        return None


def embed_query(text: str) -> List[float]:
    resp = _client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding


def format_context(docs: List[str], metas: List[dict]) -> str:
    blocks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        src = meta.get("source", "unknown")
        idx = meta.get("chunk_index", "?")
        blocks.append(f"[Source {i}: {src} | chunk {idx}]\n{doc}".strip())
    return "\n\n---\n\n".join(blocks)


def answer_question(
    question: str,
    top_k: int = 3,
    discipline_filter: str = "all",
    show_sources: bool = True,
    temperature: float = 0.2,
) -> Tuple[str, str]:
    # Key must be present in the environment (Render dashboard or local shell)
    if not os.getenv("OPENAI_API_KEY"):
        return ("OPENAI_API_KEY is not set in the environment.", "")

    question = (question or "").strip()
    if not question:
        return ("Please enter a question.", "")

    # Load collection and sanity-check it has data
    try:
        collection = _get_collection()
    except Exception as e:
        return (f"Failed to open Chroma DB at '{PERSIST_DIR}': {e}", "")

    count = _collection_count_safe()
    if count == 0:
        return (
            "Chroma collection is empty. Run the ingestion script to populate the vector database "
            f"('{COLLECTION_NAME}' at '{PERSIST_DIR}').",
            "",
        )

    # Optional metadata filter (requires ingest to set meta['discipline'])
    where = None
    discipline_filter = (discipline_filter or "").strip().lower()
    if discipline_filter and discipline_filter != "all":
        where = {"discipline": discipline_filter}

    q_emb = embed_query(question)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=int(top_k),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0] if results.get("documents") else []
    metas = results["metadatas"][0] if results.get("metadatas") else []
    dists = results["distances"][0] if results.get("distances") else []

    if not docs:
        # Helpful message if filter is too restrictive
        if where is not None:
            return (
                "No matches found in the vector database for the selected discipline filter. "
                "Try discipline_filter='all' or verify your ingest metadata includes 'discipline'.",
                "",
            )
        return ("No matches found in the vector database.", "")

    context = format_context(docs, metas)

    system = (
        "You are a combatives training assistant. "
        "Answer using ONLY the provided sources. "
        "If the sources do not contain the answer, say so and ask a clarifying question. "
        "Be concise, technical, and practical."
    )

    user = (
        f"Question:\n{question}\n\n"
        f"Sources:\n{context}\n\n"
        "Task:\n"
        "1) Answer the question grounded in the sources.\n"
        "2) If relevant, give a short drill or checklist.\n"
        "3) Cite which Source numbers you used (e.g., 'Sources: 1, 3')."
    )

    chat = _client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=float(temperature),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    answer = chat.choices[0].message.content.strip()

    sources_text = ""
    if show_sources:
        lines = []
        for i, (m, dist) in enumerate(zip(metas, dists), start=1):
            lines.append(
                f"Source {i}: {m.get('source','unknown')} | "
                f"chunk {m.get('chunk_index','?')} | distance {dist:.4f}"
            )
        sources_text = "\n".join(lines)

    return answer, sources_text