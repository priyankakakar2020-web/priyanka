#!/usr/bin/env python3
"""
Gemini-backed RAG query executor (Step 5 of the pipeline).

Requirements:
    - `py -3 -m pip install -r requirements.txt`
    - Export GEMINI_API_KEY to your environment before running.

Usage:
    py -3 scripts/rag_query_gemini.py --question "What is the expense ratio of this scheme?"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import faiss  # type: ignore
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
VECTOR_DIR = ROOT / "vector_store"
INDEX_PATH = VECTOR_DIR / "faiss.index"
DOCUMENTS_PATH = VECTOR_DIR / "documents.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "gemini-1.5-flash"


def ensure_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Please create a key at "
            "https://makersuite.google.com/app/apikey and export it, e.g.\n"
            '  setx GEMINI_API_KEY "your-key-here"  (PowerShell)'
        )
    return api_key


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


def build_prompt(question: str, hits: List[Dict]) -> str:
    context_blocks = []
    for idx, hit in enumerate(hits, start=1):
        url = hit["metadata"].get("url")
        if not url:
            continue  # Skip entries without valid URLs
        context_blocks.append(
            f"Snippet {idx} (source: {url}):\n{hit['text']}"
        )
    context_text = "\n\n".join(context_blocks) or "No snippets available."

    instructions = (
        "You are a mutual fund FAQ assistant. Answer the user's question using only the provided snippets. "
        "Every answer must:\n"
        "1. Contain only verified facts from the snippets.\n"
        "2. Include exactly one explicit citation link to the most relevant source URL.\n"
        "3. Avoid investment advice, recommendations, or opinions.\n"
        "4. If the question cannot be answered from the snippets, say so and mention that only facts are provided.\n"
    )

    return (
        f"{instructions}\n\n"
        f"Snippets:\n{context_text}\n\n"
        f"User question: {question}\n"
        "Answer:"
    )


def query_gemini(prompt: str) -> str:
    ensure_api_key()
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini-based RAG answering")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--top_k", type=int, default=5, help="Number of snippets to retrieve")
    args = parser.parse_args()

    hits = retrieve(args.question, top_k=args.top_k)
    prompt = build_prompt(args.question, hits)
    answer = query_gemini(prompt)
    print(answer)


if __name__ == "__main__":
    main()


