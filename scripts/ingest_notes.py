# This updates modified docs and ingest new documents

import os
from dotenv import load_dotenv
if os.path.exists(".env"):
    load_dotenv()

import re
import glob
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import yaml
import tiktoken
import chromadb
from openai import OpenAI


# -----------------------------
# Config (adjust as needed)
# -----------------------------
EMBEDDING_MODEL = "text-embedding-3-small"  # good quality + efficient :contentReference[oaicite:2]{index=2}
ENCODING_NAME = "cl100k_base"  # recommended for text-embedding-3-* :contentReference[oaicite:3]{index=3}

CHUNK_TOKENS = 800
CHUNK_OVERLAP = 120

#PERSIST_DIR = "./vector_db"
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vector_db")
os.makedirs(PERSIST_DIR, exist_ok=True)


COLLECTION_NAME = "combatives_notes"

#DEFAULT_NOTES_DIR = "./combatives-notes"  # e.g., combatives-notes/muay-thai/*.md
DEFAULT_NOTES_DIR = os.getenv("NOTES_DIR", "./combatives-notes")

# -----------------------------
# Helpers
# -----------------------------
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_chunk_id(source_relpath: str, file_hash: str, chunk_index: int) -> str:
    base = f"{source_relpath}::{file_hash}::{chunk_index}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:32]


def parse_markdown_with_front_matter(md_text: str) -> Tuple[Dict, str]:
    """
    Returns: (metadata_dict, body_text)
    Accepts YAML front matter delimited by --- ... ---
    """
    m = FRONT_MATTER_RE.match(md_text)
    if not m:
        return {}, md_text.strip()

    yaml_block = m.group(1)
    body = md_text[m.end():].strip()

    try:
        meta = yaml.safe_load(yaml_block) or {}
        if not isinstance(meta, dict):
            meta = {}
    except Exception:
        meta = {}

    return meta, body


def tokenize_split(text: str, encoding_name: str, chunk_tokens: int, overlap: int) -> List[str]:
    """
    Token-aware chunking so we don't exceed embedding limits.
    (Embedding inputs have max token limits; keep chunks comfortably under.)
    :contentReference[oaicite:4]{index=4}
    """
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    if not tokens:
        return []

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_tokens - overlap)

    while start < len(tokens):
        end = min(len(tokens), start + chunk_tokens)
        chunk = enc.decode(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks



def normalize_meta(meta: Dict) -> Dict:
    """
    Chroma metadata must be primitive types:
    str, int, float, bool, or None.
    YAML may parse dates into datetime.date objects, so we coerce.
    """
    out = dict(meta) if meta else {}

    # Normalize common string fields
    if "discipline" in out and isinstance(out["discipline"], str):
        out["discipline"] = out["discipline"].strip().lower().replace(" ", "-")

    if "focus" in out and isinstance(out["focus"], str):
        out["focus"] = out["focus"].strip()

    # Ensure class is numeric if present
    if "class" in out:
        try:
            out["class"] = int(out["class"])
        except Exception:
            pass

    # Coerce any non-primitive metadata (e.g., date objects) to strings
    for k, v in list(out.items()):
        if v is None or isinstance(v, (str, int, float, bool)):
            continue
        out[k] = str(v)

    return out




@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: Dict


def build_chunk_records(
    file_path: str,
    root_dir: str,
    md_text: str,
    meta: Dict,
    body: str,
) -> List[ChunkRecord]:
    relpath = os.path.relpath(file_path, root_dir).replace("\\", "/")
    file_hash = sha256_text(md_text)

    chunks = tokenize_split(
        text=body,
        encoding_name=ENCODING_NAME,
        chunk_tokens=CHUNK_TOKENS,
        overlap=CHUNK_OVERLAP,
    )

    norm_meta = normalize_meta(meta)

    records: List[ChunkRecord] = []
    for idx, chunk_text in enumerate(chunks):
        cid = stable_chunk_id(relpath, file_hash, idx)
        chunk_meta = {
            **norm_meta,
            "source": relpath,
            "file_hash": file_hash,
            "chunk_index": idx,
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_overlap": CHUNK_OVERLAP,
        }
        records.append(ChunkRecord(chunk_id=cid, text=chunk_text, metadata=chunk_meta))

    return records


def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """
    Batch embedding call.
    The embeddings endpoint supports arrays of input strings. :contentReference[oaicite:5]{index=5}
    """
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # Preserve order
    return [item.embedding for item in resp.data]


def upsert_records(collection, client: OpenAI, records: List[ChunkRecord], batch_size: int = 64) -> int:
    """
    Upserts in batches. Chroma will persist to disk.
    """
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        ids = [r.chunk_id for r in batch]
        docs = [r.text for r in batch]
        metas = [r.metadata for r in batch]
        embs = embed_texts(client, docs)

        collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embs,
        )
        total += len(batch)
    return total



def ingest_notes(notes_dir: Optional[str] = None) -> Dict[str, int]:
    """
    Import-safe entrypoint for FastAPI or other callers.
    Returns basic stats for logging/response.
    """
    notes_dir = notes_dir or DEFAULT_NOTES_DIR
    return main(notes_dir=notes_dir, run_sanity_query=False)




def main(notes_dir: str, run_sanity_query: bool = False) -> Dict[str, int]:

    files_seen = 0
    files_ingested = 0
    total_upserted = 0



    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in your environment.")

    client = OpenAI()

    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    md_files = sorted(glob.glob(os.path.join(notes_dir, "**", "*.md"), recursive=True))
    if not md_files:
        print(f"No .md files found under: {notes_dir}")
        return

    for fp in md_files:
        files_seen += 1

        with open(fp, "r", encoding="utf-8") as f:
            md_text = f.read()

        meta, body = parse_markdown_with_front_matter(md_text)
        if not body.strip():
            continue

        # delete old chunks for this source file
        source_relpath = os.path.relpath(fp, notes_dir).replace("\\", "/")
        collection.delete(where={"source": source_relpath})

        # we are ingesting this file (it had content)
        files_ingested += 1

        # Build fresh chunks for this file
        records = build_chunk_records(
            file_path=fp,
            root_dir=notes_dir,
            md_text=md_text,
            meta=meta,
            body=body,
        )

        if records:
            upserted = upsert_records(collection, client, records)
            total_upserted += upserted

    if total_upserted == 0:
        print("No chunkable content found.")
        return {
            "files_seen": files_seen,
            "files_ingested": files_ingested,
            "chunks_upserted": 0,
        }

    print(
        f"Ingest complete: files_seen={files_seen}, files_ingested={files_ingested}, "
        f"chunks_upserted={total_upserted} into collection '{COLLECTION_NAME}' at '{PERSIST_DIR}'."
    )

    stats = {
        "files_seen": files_seen,
        "files_ingested": files_ingested,
        "chunks_upserted": total_upserted,
    }


    #print(f"Upserted {inserted} chunks into Chroma collection '{COLLECTION_NAME}' at '{PERSIST_DIR}'.")

    # Small sanity query (optional)
    if run_sanity_query:
        # Small sanity query (optional)
        q = "How do I use teeps to maintain forward pressure?"
        q_emb = embed_texts(client, [q])[0]

        sample = collection.query(
            query_embeddings=[q_emb],
            n_results=3,
        )

        print("\nSample query results (top 3):")
        for i, doc in enumerate(sample["documents"][0], start=1):
            meta = sample["metadatas"][0][i - 1]
            print(f"\nResult #{i} | source={meta.get('source')} | chunk_index={meta.get('chunk_index')}")
            print(doc[:400], "...")
        return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest combatives markdown notes into a Chroma vector DB.")
    parser.add_argument(
        "--notes_dir",
        type=str,
        default=DEFAULT_NOTES_DIR,
        help="Root folder containing your .md notes (can contain subfolders).",
    )
    args = parser.parse_args()
    main(args.notes_dir, run_sanity_query=True)