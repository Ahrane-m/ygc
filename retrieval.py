"""
Retrieval-Augmented Q&A Layer (Phase 1)
=========================================
Sits on top of the ALREADY-EXTRACTED structured JSON produced by
medical_extractor.py — specifically the per-patient timeline returned by
build_patient_timeline(). It does NOT re-read raw documents.

Pipeline:
    patient timeline -> chunks (one per medication / lab result / clinical
    note / allergy list) -> embed each chunk's text with
    text-embedding-3-small -> store in a per-patient local Chroma
    collection -> at query time, embed the question, retrieve the top_k
    most similar chunks, and ask a chat model to answer strictly from that
    retrieved context.

Install:
    pip install chromadb --break-system-packages

Env:
    export OPENAI_API_KEY="sk-..."   (same key used by medical_extractor.py)
"""

import os
import re
import json
import hashlib
from typing import Any, Dict, List, Optional

import chromadb
from openai import OpenAIError

from medical_extractor import client, MODEL

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = MODEL  # reuse the same chat model configured in medical_extractor.py

CHROMA_DIR = "./chroma_db"
EMBEDDING_BATCH_SIZE = 100  # keep well under the API's per-request item limit


# ---------------------------------------------------------------------------
# 1. Chunking — turn a patient timeline into retrievable text chunks
# ---------------------------------------------------------------------------

def _chunk_id(patient_key: str, source_file: Optional[str], chunk_type: str, index: int) -> str:
    """Stable, deterministic chunk ID so re-indexing the same documents
    upserts in place instead of creating duplicates."""
    raw = f"{patient_key}|{source_file or 'unknown'}|{chunk_type}|{index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _medication_chunk_text(med: Dict[str, Any]) -> str:
    ingredients = ", ".join(med.get("ingredients") or []) or "unknown"
    duration = med.get("duration") or "not specified"
    return (
        f"Medication: {med.get('name', 'unknown')}. "
        f"Active ingredient(s): {ingredients}. "
        f"Dosage: {med.get('dosage', 'unknown')}. "
        f"Frequency: {med.get('frequency', 'unknown')}. "
        f"Duration: {duration}. "
        f"Prescribed on {med.get('date') or 'an unknown date'} "
        f"(source: {med.get('source_file') or 'unknown file'})."
    )


def _lab_result_chunk_text(lab: Dict[str, Any]) -> str:
    unit = lab.get("unit") or ""
    ref_range = lab.get("reference_range") or "not specified"
    return (
        f"Lab result: {lab.get('test_name', 'unknown test')} = "
        f"{lab.get('value', 'unknown')}{(' ' + unit) if unit else ''} "
        f"(flag: {lab.get('flag', 'unknown')}, reference range: {ref_range}). "
        f"Recorded on {lab.get('date') or 'an unknown date'} "
        f"(source: {lab.get('source_file') or 'unknown file'})."
    )


def _clinical_note_chunk_text(visit: Dict[str, Any]) -> str:
    source_file = visit.get("_source", {}).get("file")
    return (
        f"Clinical note from visit on {visit.get('date') or 'an unknown date'} "
        f"(source: {source_file or 'unknown file'}): {visit.get('clinical_notes')}"
    )


def _allergy_chunk_text(allergies: List[str]) -> str:
    return "Known allergies: " + ", ".join(allergies) + "."


def build_chunks_from_timeline(patient_key: str, timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Converts a patient timeline (the dict returned by build_patient_timeline())
    into a flat list of retrievable chunks. Each chunk is:
        {"id": str, "text": str, "metadata": {...}}

    One chunk is produced per medication entry, per lab result, per visit's
    clinical_notes (when present), and one chunk lists all known_allergies
    together. Chunk IDs are deterministic hashes so re-running this on the
    same documents upserts instead of duplicating.
    """
    chunks: List[Dict[str, Any]] = []

    for i, med in enumerate(timeline.get("medications_timeline", [])):
        chunks.append({
            "id": _chunk_id(patient_key, med.get("source_file"), "medication", i),
            "text": _medication_chunk_text(med),
            "metadata": {
                "patient_key": patient_key,
                "date": med.get("date") or "",
                "source_file": med.get("source_file") or "",
                "chunk_type": "medication",
            },
        })

    for i, lab in enumerate(timeline.get("lab_results_timeline", [])):
        chunks.append({
            "id": _chunk_id(patient_key, lab.get("source_file"), "lab_result", i),
            "text": _lab_result_chunk_text(lab),
            "metadata": {
                "patient_key": patient_key,
                "date": lab.get("date") or "",
                "source_file": lab.get("source_file") or "",
                "chunk_type": "lab_result",
            },
        })

    for i, visit in enumerate(timeline.get("visits", [])):
        if not visit.get("clinical_notes"):
            continue
        source_file = visit.get("_source", {}).get("file")
        chunks.append({
            "id": _chunk_id(patient_key, source_file, "clinical_note", i),
            "text": _clinical_note_chunk_text(visit),
            "metadata": {
                "patient_key": patient_key,
                "date": visit.get("date") or "",
                "source_file": source_file or "",
                "chunk_type": "clinical_note",
            },
        })

    allergies = timeline.get("known_allergies") or []
    if allergies:
        chunks.append({
            "id": _chunk_id(patient_key, None, "allergy", 0),
            "text": _allergy_chunk_text(allergies),
            "metadata": {
                "patient_key": patient_key,
                "date": "",
                "source_file": "",
                "chunk_type": "allergy",
            },
        })

    return chunks


# ---------------------------------------------------------------------------
# 2. Embedding + Chroma storage
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embeds a list of strings with text-embedding-3-small, batching to
    stay under the API's per-request item limit. Raises RuntimeError with
    context if the embedding call fails (auth, rate limit, network, etc.)."""
    if not texts:
        return []

    embeddings: List[List[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        except OpenAIError as e:
            raise RuntimeError(f"Embedding request failed for {len(batch)} chunk(s): {e}") from e
        embeddings.extend(item.embedding for item in response.data)

    return embeddings


def _sanitize_collection_name(patient_key: str) -> str:
    """Chroma collection names must be 3-63 chars, start/end alphanumeric,
    and contain only [a-zA-Z0-9._-]. This maps an arbitrary patient_key
    (e.g. 'amit sharma') into a safe, stable collection name."""
    name = re.sub(r"[^a-z0-9._-]+", "_", patient_key.strip().lower()).strip("_.-")
    if not name:
        name = "patient"
    if not name[0].isalnum():
        name = "p" + name
    if not name[-1].isalnum():
        name = name + "0"
    while len(name) < 3:
        name += "0"
    return name[:63]


def _get_chroma_client() -> "chromadb.ClientAPI":
    return chromadb.PersistentClient(path=CHROMA_DIR)


def _get_patient_collection(patient_key: str, create: bool):
    """Fetches (or creates) the Chroma collection for one patient. Returns
    None if create=False and no collection exists yet for this patient."""
    db = _get_chroma_client()
    name = _sanitize_collection_name(patient_key)
    if create:
        return db.get_or_create_collection(name=name, metadata={"patient_key": patient_key})
    try:
        return db.get_collection(name=name)
    except Exception:
        return None


def index_patient_timeline(patient_key: str, timeline: Dict[str, Any]) -> None:
    """
    Entry point for indexing: chunks a patient's timeline, embeds every
    chunk, and upserts them into that patient's local Chroma collection
    (persisted under ./chroma_db). Safe to call repeatedly on the same
    timeline — chunk IDs are deterministic, so re-indexing overwrites
    existing entries rather than duplicating them.
    """
    if not patient_key or not patient_key.strip():
        raise ValueError("patient_key is required and cannot be empty.")

    chunks = build_chunks_from_timeline(patient_key, timeline)
    if not chunks:
        print(f"  No indexable content found for patient '{patient_key}' — skipping indexing.")
        return

    embeddings = embed_texts([c["text"] for c in chunks])

    collection = _get_patient_collection(patient_key, create=True)
    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    print(f"  Indexed {len(chunks)} chunk(s) for patient '{patient_key}' into Chroma ({CHROMA_DIR}).")


# ---------------------------------------------------------------------------
# 3. Retrieval + Q&A
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = """
You are a patient-facing medical records assistant. You answer questions
using ONLY the retrieved context provided to you below — structured chunks
pulled from that patient's own extracted medical records (medications, lab
results, clinical notes, allergies).

Rules:
- Answer strictly from the retrieved context. If the context does not cover
  the question, say "I don't have enough information" rather than guessing
  or using outside medical knowledge.
- NEVER provide a diagnosis or interpret what a result "means" clinically.
- Whenever the question touches on risk, drug interactions, allergy
  conflicts, or changing/adjusting a dosage, explicitly recommend the
  patient consult a doctor or pharmacist, and set
  recommend_professional_consult to true.
- Cite the date and source_file of every chunk you rely on in "sources".
- Respond with STRICT JSON only, matching the required schema.
"""

ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "source_file": {"type": "string"},
                },
                "required": ["date", "source_file"],
                "additionalProperties": False,
            },
        },
        "recommend_professional_consult": {"type": "boolean"},
    },
    "required": ["answer", "confidence", "sources", "recommend_professional_consult"],
    "additionalProperties": False,
}

ANSWER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "patient_qa_answer",
        "strict": True,
        "schema": ANSWER_JSON_SCHEMA,
    },
}

_NO_INFO_ANSWER = {
    "answer": "I don't have enough information — no indexed records were found for this patient yet.",
    "confidence": 0.0,
    "sources": [],
    "recommend_professional_consult": False,
}


def answer_question(
    patient_key: str,
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 8,
    retrieval_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Answers a natural-language question about one patient, grounded only in
    that patient's already-indexed timeline chunks.

    1. Embeds the retrieval query (see retrieval_query below).
    2. Queries the patient's Chroma collection for the top_k most similar
       chunks.
    3. Builds a prompt from those chunks (each tagged with its date and
       source_file), plus chat_history and the (display) question.
    4. Calls the chat model with a system prompt that forbids diagnosis,
       requires deferring to a professional for risk/interaction/dosage
       questions, and forces structured JSON output.

    retrieval_query: optional string used for embedding/Chroma retrieval
        instead of `question`. Lets a caller (e.g. conversation.py) rewrite
        an ambiguous follow-up like "was that safe?" into a fully-specified
        search query, while `question` remains the literal text shown to
        the answering LLM as "the question asked". Defaults to `question`
        when omitted, so existing single-shot callers are unaffected.

    Returns the parsed JSON:
        {"answer": str, "confidence": float, "sources": [{"date", "source_file"}],
         "recommend_professional_consult": bool}

    Raises ValueError for a missing patient_key/question, RuntimeError if
    the embedding or chat call fails. Returns a graceful "no information"
    answer (no API calls) if the patient has never been indexed or their
    collection is empty.
    """
    if not patient_key or not patient_key.strip():
        raise ValueError("patient_key is required and cannot be empty.")
    if not question or not question.strip():
        raise ValueError("question is required and cannot be empty.")

    effective_retrieval_query = (
        retrieval_query if retrieval_query and retrieval_query.strip() else question
    )

    collection = _get_patient_collection(patient_key, create=False)
    if collection is None or collection.count() == 0:
        return dict(_NO_INFO_ANSWER)

    query_embedding = embed_texts([effective_retrieval_query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )
    docs = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    if not docs:
        return dict(_NO_INFO_ANSWER)

    context_blocks = [
        f"[date: {meta.get('date') or 'unknown'} | source_file: {meta.get('source_file') or 'unknown'} "
        f"| type: {meta.get('chunk_type') or 'unknown'}]\n{text}"
        for text, meta in zip(docs, metadatas)
    ]
    context_str = "\n\n".join(context_blocks)

    messages = [{"role": "system", "content": QA_SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({
        "role": "user",
        "content": f"Retrieved patient records:\n\n{context_str}\n\nQuestion: {question}",
    })

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            response_format=ANSWER_RESPONSE_FORMAT,
        )
    except OpenAIError as e:
        raise RuntimeError(f"Chat completion failed while answering question: {e}") from e

    return json.loads(response.choices[0].message.content)
