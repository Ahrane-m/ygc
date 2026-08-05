"""
HTTP API (Phase 3 + 4)
=========================================
Exposes the extraction -> timeline -> cross-check -> trend-track -> retrieval
-> conversation pipeline (medical_extractor.py, lab_trends.py, retrieval.py,
conversation.py) over HTTP, under the /api/v1/ prefix. This is a thin
wrapper — all business logic stays in those modules; this file only handles
request/response marshalling, validation, and HTTP status codes.

Every route except /health requires an authenticated caller (see auth.py):
    Authorization: Bearer <jwt>
    X-User-Id: <user_id>

There is one patient per user — user_id from the verified token IS the
patient key used throughout the pipeline, so every read/write is naturally
scoped to the caller. Uploaded files are archived to Cloudinary
(storage.py) and their structured extraction + document_url is persisted
in MongoDB (db.py), keyed by user_id.

Run:
    uvicorn api:app --reload
    # then see interactive docs at http://127.0.0.1:8000/docs

Install (in addition to Phase 1/2 dependencies):
    pip install fastapi uvicorn[standard] python-multipart pymongo cloudinary pyjwt

Env:
    OPENAI_API_KEY, MONGODB_URI, CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY,
    CLOUDINARY_API_SECRET, JWT_SECRET
"""

import logging
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

import conversation
import db
import storage
from auth import get_current_user
from document_filter import NonMedicalDocumentError, assert_medical_document
from lab_trends import track_lab_trends
from medical_extractor import (
    _is_demo_document,
    build_patient_timeline,
    cross_check_prescriptions,
    process_document,
)
from retrieval import answer_question, index_patient_timeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api")

SUPPORTED_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".webp")

app = FastAPI(title="Medical Records Q&A API", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    db.ensure_indexes()


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
# Documents / timeline / cross-check / lab trends
# ---------------------------------------------------------------------------

@app.post("/api/v1/documents", status_code=201)
async def upload_documents(
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Uploads one or more documents (PDF/image) for the authenticated user.
    Extracts each, merges the results with any documents previously
    uploaded by this user, rebuilds the timeline, re-runs cross-checking
    and lab trend tracking, and re-indexes for Q&A.
    """
    logger.info("upload_documents: user=%s received %d file(s)", user_id, len(files))
    if not files:
        raise HTTPException(400, "No files were uploaded.")

    # Pass 1: extract + validate every file/page first. Nothing is uploaded
    # to Cloudinary or written to Mongo until the whole batch passes, so a
    # bad file later in the batch never leaves an orphaned upload behind
    # for a good file earlier in it.
    per_file_pages: List[Tuple[Path, str, List[Dict[str, Any]]]] = []
    new_docs: List[Dict[str, Any]] = []

    with TemporaryDirectory() as tmp_dir:
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                logger.warning(
                    "upload_documents: user=%s rejected '%s' (unsupported type '%s')",
                    user_id, upload.filename, suffix or "(none)",
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
                "upload_documents: user=%s processing '%s' (%d bytes)",
                user_id, upload.filename, len(content),
            )
            try:
                result = process_document(str(tmp_path))
            except Exception as e:
                logger.error(
                    "upload_documents: user=%s extraction failed for '%s': %s",
                    user_id, upload.filename, e, exc_info=True,
                )
                raise HTTPException(422, f"Extraction failed for '{upload.filename}': {e}")

            if isinstance(result, dict) and result.get("multi_page"):
                logger.info(
                    "upload_documents: user=%s '%s' extracted as %d page(s)",
                    user_id, upload.filename, len(result["pages"]),
                )
                pages = result["pages"]
            else:
                pages = [result]

            # Drop demo/placeholder pages and reject non-medical files here,
            # right after extraction and before any expensive downstream
            # work (Cloudinary upload, timeline rebuild, cross-check LLM
            # call, re-indexing) — no extra model call, reuses the
            # document_type/medications/lab_results/etc. already produced
            # by process_document().
            kept_pages: List[Dict[str, Any]] = []
            for page_num, page in enumerate(pages, start=1):
                label = upload.filename if len(pages) == 1 else f"{upload.filename} (page {page_num})"
                if _is_demo_document(page):
                    logger.warning(
                        "upload_documents: user=%s skipped demo/placeholder page '%s'", user_id, label,
                    )
                    continue
                try:
                    assert_medical_document(page, label)
                except NonMedicalDocumentError as e:
                    logger.warning(
                        "upload_documents: user=%s rejected '%s': %s", user_id, label, e.reason,
                    )
                    raise HTTPException(422, str(e))
                kept_pages.append(page)

            if kept_pages:
                per_file_pages.append((tmp_path, upload.filename, kept_pages))

        if not per_file_pages:
            raise HTTPException(
                422,
                "No medical content found in the uploaded file(s) (all pages were "
                "demo/placeholder documents).",
            )

        # Pass 2: everything validated — archive each original file to
        # Cloudinary once, and attach the resulting URL to every page that
        # came from it.
        for tmp_path, filename, kept_pages in per_file_pages:
            upload_info = storage.upload_patient_document(user_id, str(tmp_path), filename)
            for page in kept_pages:
                page["document_url"] = upload_info["document_url"]
                page["cloudinary_public_id"] = upload_info["cloudinary_public_id"]
                new_docs.append(page)

    existing_docs = db.load_documents(user_id)
    all_docs = existing_docs + new_docs
    logger.info(
        "upload_documents: user=%s merged documents: +%d new, %d total",
        user_id, len(new_docs), len(all_docs),
    )

    timeline = build_patient_timeline(all_docs)
    cross_check = cross_check_prescriptions(timeline)
    issue_count = sum(len(v) for v in cross_check.values() if isinstance(v, list))
    logger.info(
        "upload_documents: user=%s timeline rebuilt, cross-check found %d issue(s)",
        user_id, issue_count,
    )

    lab_trends = track_lab_trends(timeline)
    logger.info(
        "upload_documents: user=%s lab trend tracking found %d trend(s), %d test(s) with insufficient data",
        user_id, len(lab_trends["trends"]), len(lab_trends["insufficient_data"]),
    )

    indexed, index_error = True, None
    try:
        index_patient_timeline(user_id, timeline)
        logger.info("upload_documents: user=%s re-indexed for Q&A", user_id)
    except Exception as e:
        indexed, index_error = False, str(e)
        logger.error(
            "upload_documents: user=%s indexing failed: %s", user_id, e, exc_info=True,
        )

    db.insert_documents(user_id, new_docs)
    db.save_patient_snapshot(user_id, timeline, cross_check, lab_trends=lab_trends)
    logger.info(
        "upload_documents: user=%s request complete: documents_added=%d documents_total=%d indexed=%s",
        user_id, len(new_docs), len(all_docs), indexed,
    )

    response = {
        "user_id": user_id,
        "documents_added": len(new_docs),
        "documents_total": len(all_docs),
        "timeline": timeline,
        "cross_check_report": cross_check,
        "lab_trends": lab_trends,
        "indexed": indexed,
    }
    if not indexed:
        response["index_error"] = index_error
    return response


@app.get("/api/v1/timeline")
async def get_timeline(user_id: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Returns the authenticated user's merged timeline (medications, lab
    results, visits, allergies) from the most recent upload/processing run."""
    snapshot = db.load_patient_snapshot(user_id)
    if snapshot is None:
        raise HTTPException(404, "No timeline found for this user.")
    return snapshot["patient_timeline"]


@app.get("/api/v1/cross-check")
async def get_cross_check(user_id: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Returns the authenticated user's latest cross-check report
    (interactions, duplicates, dosage conflicts, allergy conflicts)."""
    snapshot = db.load_patient_snapshot(user_id)
    if snapshot is None:
        raise HTTPException(404, "No cross-check report found for this user.")
    return snapshot["cross_check_report"]


@app.get("/api/v1/lab-trends")
async def get_lab_trends(user_id: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Returns the authenticated user's lab result trends (direction of
    drift per test, reference-range crossings, plain-language explanations)
    computed from the most recent upload/processing run. Recomputed on the
    fly from the saved timeline for snapshots saved before this field
    existed."""
    snapshot = db.load_patient_snapshot(user_id)
    if snapshot is None:
        raise HTTPException(404, "No timeline found for this user.")
    if "lab_trends" in snapshot:
        return snapshot["lab_trends"]
    return track_lab_trends(snapshot["patient_timeline"])


# ---------------------------------------------------------------------------
# Single-shot Q&A (Phase 1)
# ---------------------------------------------------------------------------

@app.post("/api/v1/qa")
async def qa(body: QARequest, user_id: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Answers one question grounded in the authenticated user's indexed
    timeline, with no session/conversation state (caller manages
    chat_history, if any)."""
    try:
        return answer_question(
            patient_key=user_id,
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

@app.post("/api/v1/sessions", status_code=201)
async def create_session(user_id: str = Depends(get_current_user)) -> Dict[str, str]:
    """Starts a new conversation session for the authenticated user and
    returns its session_id, to be used in subsequent
    /sessions/{session_id}/messages calls."""
    session_id = uuid.uuid4().hex
    conversation.get_or_create_session(user_id, session_id)
    return {"user_id": user_id, "session_id": session_id}


@app.post("/api/v1/sessions/{session_id}/messages")
async def post_message(
    session_id: str, body: MessageRequest, user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """Asks one question within an existing conversation session — the
    question is rewritten into a self-contained retrieval query using prior
    turns before Chroma retrieval, then answered against the original
    question + history. 404s if the session doesn't exist yet (create it via
    POST /sessions first), or belongs to a different user."""
    session = conversation.get_session(user_id, session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    try:
        return conversation.ask(session, body.question, top_k=body.top_k)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(502, str(e))


@app.get("/api/v1/sessions/{session_id}")
async def get_session_history(
    session_id: str, user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """Returns the full, untrimmed transcript of a conversation session
    (for logging/export/debugging) — never summarized or truncated,
    regardless of how conversation.ask() compacts history for prompting."""
    session = conversation.get_session(user_id, session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    return {
        "user_id": user_id,
        "session_id": session_id,
        "turns": session.get_full_history(),
    }


@app.delete("/api/v1/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, user_id: str = Depends(get_current_user)) -> None:
    """Ends a conversation session, freeing its in-memory turn history."""
    if not conversation.delete_session(user_id, session_id):
        raise HTTPException(404, f"Session '{session_id}' not found.")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/v1/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
