from dotenv import load_dotenv
load_dotenv()

import os
from typing import List, Tuple, Optional, Dict, Any

import chromadb
from openai import OpenAI

# --- Config (match your ingest script) ---
PERSIST_DIR = "./vector_db"
COLLECTION_NAME = "combatives_notes"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Create client once (good practice)
_client = OpenAI()

# Optional: cache the collection so you don't reopen it every request
_collection = None

def _get_collection():
    global _collection
    if _collection is None:
        chroma = chromadb.PersistentClient(path=PERSIST_DIR)
        _collection = chroma.get_or_create_collection(name=COLLECTION_NAME)
    return _collection

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
    if not os.getenv("OPENAI_API_KEY"):
        return ("OPENAI_API_KEY is not set in this shell session.", "")

    question = (question or "").strip()
    if not question:
        return ("Please enter a question.", "")

    collection = _get_collection()

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
                f"Source {i}: {m.get('source','unknown')} | chunk {m.get('chunk_index','?')} | distance {dist:.4f}"
            )
        sources_text = "\n".join(lines)

    return answer, sources_text