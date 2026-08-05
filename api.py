"""
HTTP API (Phase 3)
=========================================
Exposes the extraction -> timeline -> cross-check -> retrieval -> conversation
pipeline (medical_extractor.py, retrieval.py, conversation.py) over HTTP,
under the /api/v1/ prefix. This is a thin wrapper — all business logic stays
in those modules; this file only handles request/response marshalling,
validation, and HTTP status codes.

Run:
    uvicorn api:app --reload
    # then see interactive docs at http://127.0.0.1:8000/docs

Install (in addition to Phase 1/2 dependencies):
    pip install fastapi uvicorn[standard] python-multipart

Env:
    export OPENAI_API_KEY="sk-..."   (same key the rest of the pipeline uses)
"""

import logging
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

import conversation
from document_filter import NonMedicalDocumentError, assert_medical_document
from medical_extractor import (
    _normalize_patient_key,
    build_patient_timeline,
    cross_check_prescriptions,
    load_patient_documents,
    load_patient_report,
    process_document,
    save_patient_documents,
    save_patient_report,
)
from retrieval import answer_question, index_patient_timeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api")

SUPPORTED_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".webp")

app = FastAPI(title="Medical Records Q&A API", version="1.0.0")


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class QARequest(BaseModel):
    """Body for the single-shot (Phase 1) Q&A endpoint."""
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None
    top_k: int = Field(default=8, ge=1, le=50)


class MessageRequest(BaseModel):
    """Body for posting a message into a conversation session (Phase 2)."""
    question: str
    top_k: int = Field(default=8, ge=1, le=50)


# ---------------------------------------------------------------------------
# Documents / timeline / cross-check
# ---------------------------------------------------------------------------

@app.post("/api/v1/patients/{patient_key}/documents", status_code=201)
async def upload_documents(patient_key: str, files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Uploads one or more documents (PDF/image) for a patient. Extracts each
    with process_document(), merges the results with any documents
    previously uploaded for this patient, rebuilds the timeline, re-runs
    cross-checking, and re-indexes for Q&A — mirroring what the CLI does
    for a single batch, but accumulating across calls.
    """
    patient_key = _normalize_patient_key(patient_key)
    logger.info("upload_documents: patient=%s received %d file(s)", patient_key, len(files))
    if not files:
        logger.warning("upload_documents: patient=%s no files in request", patient_key)
        raise HTTPException(400, "No files were uploaded.")

    new_docs = []
    with TemporaryDirectory() as tmp_dir:
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                logger.warning(
                    "upload_documents: patient=%s rejected '%s' (unsupported type '%s')",
                    patient_key, upload.filename, suffix or "(none)",
                )
                raise HTTPException(
                    400,
                    f"Unsupported file type '{suffix or '(no extension)'}' for "
                    f"'{upload.filename}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
                )
            tmp_path = Path(tmp_dir) / upload.filename
            content = await upload.read()
            tmp_path.write_bytes(content)
            logger.info(
                "upload_documents: patient=%s processing '%s' (%d bytes)",
                patient_key, upload.filename, len(content),
            )
            try:
                result = process_document(str(tmp_path))
            except Exception as e:
                logger.error(
                    "upload_documents: patient=%s extraction failed for '%s': %s",
                    patient_key, upload.filename, e, exc_info=True,
                )
                raise HTTPException(422, f"Extraction failed for '{upload.filename}': {e}")

            if isinstance(result, dict) and result.get("multi_page"):
                logger.info(
                    "upload_documents: patient=%s '%s' extracted as %d page(s)",
                    patient_key, upload.filename, len(result["pages"]),
                )
                pages = result["pages"]
            else:
                pages = [result]

            # Reject non-medical files here, right after extraction and
            # before any of the expensive downstream work (timeline
            # rebuild, cross-check LLM call, re-indexing). No extra model
            # call is made — this reuses the document_type/medications/
            # lab_results/etc. already produced by process_document().
            for page_num, page in enumerate(pages, start=1):
                label = upload.filename if len(pages) == 1 else f"{upload.filename} (page {page_num})"
                try:
                    assert_medical_document(page, label)
                except NonMedicalDocumentError as e:
                    logger.warning(
                        "upload_documents: patient=%s rejected '%s': %s",
                        patient_key, label, e.reason,
                    )
                    raise HTTPException(422, str(e))

            new_docs.extend(pages)

    all_docs = load_patient_documents(patient_key) + new_docs
    logger.info(
        "upload_documents: patient=%s merged documents: +%d new, %d total",
        patient_key, len(new_docs), len(all_docs),
    )

    timeline = build_patient_timeline(all_docs)
    cross_check = cross_check_prescriptions(timeline)
    issue_count = sum(len(v) for v in cross_check.values() if isinstance(v, list))
    logger.info(
        "upload_documents: patient=%s timeline rebuilt, cross-check found %d issue(s)",
        patient_key, issue_count,
    )

    indexed, index_error = True, None
    try:
        index_patient_timeline(patient_key, timeline)
        logger.info("upload_documents: patient=%s re-indexed for Q&A", patient_key)
    except Exception as e:
        indexed, index_error = False, str(e)
        logger.error(
            "upload_documents: patient=%s indexing failed: %s", patient_key, e, exc_info=True,
        )

    save_patient_documents(patient_key, all_docs)
    save_patient_report(patient_key, timeline, cross_check)
    logger.info(
        "upload_documents: patient=%s request complete: documents_added=%d documents_total=%d indexed=%s",
        patient_key, len(new_docs), len(all_docs), indexed,
    )

    response = {
        "patient_key": patient_key,
        "documents_added": len(new_docs),
        "documents_total": len(all_docs),
        "timeline": timeline,
        "cross_check_report": cross_check,
        "indexed": indexed,
    }
    if not indexed:
        response["index_error"] = index_error
    return response


@app.get("/api/v1/patients/{patient_key}/timeline")
async def get_timeline(patient_key: str) -> Dict[str, Any]:
    """Returns the patient's merged timeline (medications, lab results,
    visits, allergies) from the most recent upload/processing run."""
    patient_key = _normalize_patient_key(patient_key)
    report = load_patient_report(patient_key)
    if report is None:
        raise HTTPException(404, f"No timeline found for patient '{patient_key}'.")
    return report["patient_timeline"]


@app.get("/api/v1/patients/{patient_key}/cross-check")
async def get_cross_check(patient_key: str) -> Dict[str, Any]:
    """Returns the patient's latest cross-check report (interactions,
    duplicates, dosage conflicts, allergy conflicts)."""
    patient_key = _normalize_patient_key(patient_key)
    report = load_patient_report(patient_key)
    if report is None:
        raise HTTPException(404, f"No cross-check report found for patient '{patient_key}'.")
    return report["cross_check_report"]


# ---------------------------------------------------------------------------
# Single-shot Q&A (Phase 1)
# ---------------------------------------------------------------------------

@app.post("/api/v1/patients/{patient_key}/qa")
async def qa(patient_key: str, body: QARequest) -> Dict[str, Any]:
    """Answers one question grounded in the patient's indexed timeline, with
    no session/conversation state (caller manages chat_history, if any)."""
    patient_key = _normalize_patient_key(patient_key)
    try:
        return answer_question(
            patient_key=patient_key,
            question=body.question,
            chat_history=body.chat_history,
            top_k=body.top_k,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(502, str(e))


# ---------------------------------------------------------------------------
# Multi-turn conversation (Phase 2)
# ---------------------------------------------------------------------------

@app.post("/api/v1/patients/{patient_key}/sessions", status_code=201)
async def create_session(patient_key: str) -> Dict[str, str]:
    """Starts a new conversation session for a patient and returns its
    session_id, to be used in subsequent /sessions/{session_id}/messages
    calls."""
    patient_key = _normalize_patient_key(patient_key)
    session_id = uuid.uuid4().hex
    conversation.get_or_create_session(patient_key, session_id)
    return {"patient_key": patient_key, "session_id": session_id}


@app.post("/api/v1/patients/{patient_key}/sessions/{session_id}/messages")
async def post_message(patient_key: str, session_id: str, body: MessageRequest) -> Dict[str, Any]:
    """Asks one question within an existing conversation session — the
    question is rewritten into a self-contained retrieval query using prior
    turns before Chroma retrieval, then answered against the original
    question + history. 404s if the session doesn't exist yet (create it via
    POST /sessions first)."""
    patient_key = _normalize_patient_key(patient_key)
    session = conversation.get_session(patient_key, session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found for patient '{patient_key}'.")
    try:
        return conversation.ask(session, body.question, top_k=body.top_k)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(502, str(e))


@app.get("/api/v1/patients/{patient_key}/sessions/{session_id}")
async def get_session_history(patient_key: str, session_id: str) -> Dict[str, Any]:
    """Returns the full, untrimmed transcript of a conversation session
    (for logging/export/debugging) — never summarized or truncated,
    regardless of how conversation.ask() compacts history for prompting."""
    patient_key = _normalize_patient_key(patient_key)
    session = conversation.get_session(patient_key, session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found for patient '{patient_key}'.")
    return {
        "patient_key": patient_key,
        "session_id": session_id,
        "turns": session.get_full_history(),
    }


@app.delete("/api/v1/patients/{patient_key}/sessions/{session_id}", status_code=204)
async def delete_session(patient_key: str, session_id: str) -> None:
    """Ends a conversation session, freeing its in-memory turn history."""
    patient_key = _normalize_patient_key(patient_key)
    if not conversation.delete_session(patient_key, session_id):
        raise HTTPException(404, f"Session '{session_id}' not found for patient '{patient_key}'.")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/v1/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
