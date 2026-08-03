"""
Chroma Inspection CLI
=========================================
Read-only debug/demo helper for looking at what's actually stored in the
local Chroma vector store (./chroma_db) that retrieval.py builds and
queries. Never modifies the store, never calls OpenAI.

Usage:
    python inspect_chroma.py                            # list every patient collection + chunk count
    python inspect_chroma.py "amit sharma"               # show chunks for one patient
    python inspect_chroma.py "amit sharma" --limit 20    # show more chunks
    python inspect_chroma.py "amit sharma" --type medication   # filter by chunk_type
"""

import argparse
from typing import Optional

from retrieval import CHROMA_DIR, _get_chroma_client, _sanitize_collection_name

CHUNK_TYPES = ["medication", "lab_result", "clinical_note", "allergy"]


def list_collections() -> None:
    """Prints every patient collection currently in the local Chroma store,
    with its sanitized collection name and chunk count."""
    client = _get_chroma_client()
    collections = client.list_collections()
    if not collections:
        print(f"No collections found in {CHROMA_DIR} — nothing has been indexed yet.")
        return
    print(f"{len(collections)} collection(s) in {CHROMA_DIR}:")
    for c in collections:
        coll = client.get_collection(c.name)
        print(f"  {c.name}  ({coll.count()} chunk(s))")


def show_patient(patient_key: str, limit: int, chunk_type: Optional[str]) -> None:
    """Prints up to `limit` chunks (text + metadata) for one patient,
    optionally filtered to a single chunk_type."""
    client = _get_chroma_client()
    name = _sanitize_collection_name(patient_key)
    try:
        coll = client.get_collection(name)
    except Exception:
        print(f"No collection found for patient_key '{patient_key}' (looked for '{name}').")
        return

    where = {"chunk_type": chunk_type} if chunk_type else None
    data = coll.get(limit=limit, where=where, include=["documents", "metadatas"])
    docs = data["documents"]
    metas = data["metadatas"]

    header = f"Collection '{name}' — {coll.count()} total chunk(s), showing up to {limit}"
    if chunk_type:
        header += f" (chunk_type={chunk_type})"
    print(header + ":\n")

    if not docs:
        print("  (no chunks matched)")
        return

    for i, (doc, meta) in enumerate(zip(docs, metas)):
        print(f"[{i}] {meta}")
        print(f"    {doc}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the local Chroma vector store (read-only).")
    parser.add_argument("patient_key", nargs="?", help="Patient to inspect (omit to list all collections)")
    parser.add_argument("--limit", type=int, default=10, help="Max chunks to show (default: 10)")
    parser.add_argument("--type", dest="chunk_type", default=None, choices=CHUNK_TYPES,
                         help="Filter to one chunk_type")
    args = parser.parse_args()

    if args.patient_key:
        show_patient(args.patient_key, args.limit, args.chunk_type)
    else:
        list_collections()
