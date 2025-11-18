#!/usr/bin/env python3
"""
Step 2–4 of the backend architecture:
1. Load scraped JSON files (output of the scraper step).
2. Chunk/normalize each fact with its source URL.
3. Embed the chunks and persist them in a FAISS vector store.

Run:
    py -3 scripts/build_vector_store.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import faiss  # type: ignore
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SCHEMES_DIR = DATA_DIR / "schemes"
GUIDES_DIR = DATA_DIR / "guides"
VECTOR_DIR = ROOT / "vector_store"
INDEX_PATH = VECTOR_DIR / "faiss.index"
DOCUMENTS_PATH = VECTOR_DIR / "documents.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, str]


def load_scheme_chunks() -> Iterable[Chunk]:
    for path in SCHEMES_DIR.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        scheme = payload["scheme_name"]
        source_url = payload["source_url"]

        # Core metadata sentence
        yield Chunk(
            text=f"{scheme} is a {payload['metadata'].get('category')} scheme "
            f"in the {payload['metadata'].get('sub_category')} category offered by "
            f"{payload['metadata'].get('fund_house')}. Data source: {source_url}",
            metadata={
                "type": "scheme_overview",
                "scheme": scheme,
                "url": source_url,
            },
        )

    #
        attributes = payload.get("attributes", {})
        for field, value in attributes.items():
            if isinstance(value, dict):
                display = value.get("display") or value.get("value")
                description = value.get("value") if isinstance(value.get("value"), str) else display
            else:
                description = value

            if description is None:
                continue

            pretty_field = field.replace("_", " ").title()
            sentence = f"{scheme} - {pretty_field}: {description}."

            if isinstance(value, dict):
                sentence += f" Source: {value.get('source_url', source_url)}"
                chunk_url = value.get("source_url", source_url)
            else:
                sentence += f" Source: {source_url}"
                chunk_url = source_url

            yield Chunk(
                text=sentence,
                metadata={
                    "type": "scheme_attribute",
                    "field": field,
                    "scheme": scheme,
                    "url": chunk_url,
                },
            )

        for doc in payload.get("documents", []):
            yield Chunk(
                text=f"{scheme} has a {doc.get('type')} document at {doc.get('url')}.",
                metadata={
                    "type": "scheme_document",
                    "scheme": scheme,
                    "url": doc.get("url", source_url),
                },
            )


def load_guide_chunks() -> Iterable[Chunk]:
    for path in GUIDES_DIR.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_url = payload["source_url"]
        guide_key = payload["guide_key"]

        for method in payload.get("methods", []):
            steps = " ".join(method.get("steps", []))
            yield Chunk(
                text=f"{method['label']}: {steps} Source: {source_url}",
                metadata={
                    "type": "guide",
                    "guide_key": guide_key,
                    "label": method["label"],
                    "url": source_url,
                },
            )


def collect_chunks() -> List[Chunk]:
    chunks = list(load_scheme_chunks())
    chunks.extend(load_guide_chunks())
    if not chunks:
        raise RuntimeError("No chunks generated – ensure data/schemes and data/guides exist.")
    return chunks


def build_index(chunks: List[Chunk]) -> None:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode([chunk.text for chunk in chunks], show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, str(INDEX_PATH))

    documents_payload = [
        {"text": chunk.text, "metadata": chunk.metadata} for chunk in chunks
    ]
    DOCUMENTS_PATH.write_text(json.dumps(documents_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks -> {INDEX_PATH} and {DOCUMENTS_PATH}")


def main() -> None:
    chunks = collect_chunks()
    build_index(chunks)


if __name__ == "__main__":
    main()

