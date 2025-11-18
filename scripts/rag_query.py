#!/usr/bin/env python3
"""
Step 5 – Retrieval augmented answering for backend use.

Loads the FAISS index and metadata built by build_vector_store.py,
retrieves the most relevant chunks, and prints a factual answer that
includes the Groww source URL.

Usage:
    py -3 scripts/rag_query.py --question "What is the expense ratio of this scheme?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import faiss  # type: ignore
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = ROOT / "vector_store"
INDEX_PATH = VECTOR_DIR / "faiss.index"
DOCUMENTS_PATH = VECTOR_DIR / "documents.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_index():
    if not INDEX_PATH.exists() or not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(
            "Vector store missing. Run `py -3 scripts\\build_vector_store.py` first."
        )
    index = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(DOCUMENTS_PATH.read_text(encoding="utf-8"))
    return index, metadata


def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    index, metadata = load_index()
    model = SentenceTransformer(EMBED_MODEL)
    query_vec = model.encode([query])

    distances, indices = index.search(query_vec.astype("float32"), top_k)
    hits = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        doc = metadata[idx]
        hits.append(
            {
                "rank": rank + 1,
                "distance": float(distances[0][rank]),
                "text": doc["text"],
                "metadata": doc["metadata"],
            }
        )
    return hits


def compose_answer(question: str, hits: List[Dict]) -> str:
    if not hits:
        return "I could not find a factual snippet for that question."

    top = hits[0]
    url = top["metadata"].get("url")
    if not url:
        return "I found information but the source URL is missing."
    
    text = top["text"]
    answer = (
        f"Answer: {text}\n\n"
        f"Source: {url}"
    )
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG query helper")
    parser.add_argument("--question", required=True, help="User question to answer")
    args = parser.parse_args()

    hits = retrieve(args.question, top_k=3)
    answer = compose_answer(args.question, hits)
    print(answer)


if __name__ == "__main__":
    main()

