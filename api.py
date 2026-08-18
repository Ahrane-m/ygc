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

MongoDB is the only store this API needs: the patient snapshot it writes on
upload is the same one Q&A reads from (retrieval.py) and conversation
sessions are persisted alongside it (conversation.py). Nothing is kept on
local disk or in process memory, so the service is safe to restart and to
run with more than one worker.

Run:
    uvicorn api:app --reload
    # then see interactive docs at http://127.0.0.1:8000/docs

Install (in addition to Phase 1/2 dependencies):
    pip install fastapi uvicorn[standard] python-multipart pymongo cloudinary pyjwt

Env:
    OPENAI_API_KEY, MONGODB_URI, CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY,
    CLOUDINARY_API_SECRET, JWT_SECRET
"""

import asyncio
import hashlib
import logging
import os
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import conversation
import db
import graph_db
import storage
from auth import get_current_user
from consult_triage import TRIAGE_OUTPUT_VERSION, triage_consultation
from document_filter import NonMedicalDocumentError, assert_medical_document
from evidence_grading import graph_backed_findings_from_antidotes
from identity_guard import partition_identity_mismatch
from language_guard import (
    UnsupportedLanguageError,
    assert_supported_language,
    assess_documents_translation_risk,
)
from lab_trends import track_lab_trends
from medical_extractor import (
    _is_demo_document,
    build_patient_timeline,
    cross_check_inputs_fingerprint,
    cross_check_prescriptions,
    process_document,
)
from poisoning_kg import (
    extract_antidote_section,
    ingest_antidote_entries,
    lookup_antidote_references,
)
from retrieval import answer_question
from risk_timeline import build_treatment_windows, concurrent_exposure, risk_calendar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api")

SUPPORTED_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".webp")

# Browsers block cross-origin calls unless the API opts in, so a web
# frontend served from anywhere other than this host needs its origin
# listed here. Set ALLOWED_ORIGINS to a comma-separated list in
# production (e.g. "https://mediscan.example.app"); the "*" default keeps
# local development and a not-yet-deployed frontend working.
# Credentials are off because auth travels in the Authorization header,
# not a cookie — and "*" is invalid alongside allow_credentials=True.
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

app = FastAPI(title="Medical Records Q&A API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    db.ensure_indexes()
    # The antidote reference graph is an enrichment, not core: a missing
    # NEO4J_* var or a paused/unreachable Aura instance must not stop the
    # API from serving documents, timelines, cross-checks or Q&A. The read
    # path in upload_documents is already fail-open, so this is the only
    # place Neo4j could take the whole service down.
    try:
        graph_db.ensure_constraints()
        logger.info("startup: antidote reference graph ready")
    except Exception as e:
        # graph_db has already logged the failing step and the (redacted) URI;
        # this line records the consequence — the service runs without the
        # graph, so every later antidote lookup will find nothing.
        logger.warning(
            "startup: antidote reference graph unavailable, continuing without it "
            "(antidote reference notes will be empty on every upload): %s", e,
        )


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class QARequest(BaseModel):
    """Body for the single-shot (Phase 1) Q&A endpoint. `top_k` is accepted
    for backwards compatibility with existing clients and ignored —
    retrieval is whole-record assembly, not top-k nearest-neighbour search
    (see retrieval.py)."""
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None
    top_k: int = Field(default=8, ge=1, le=50, deprecated=True)


class MessageRequest(BaseModel):
    """Body for posting a message into a conversation session (Phase 2).
    `top_k` is accepted and ignored, as above."""
    question: str
    top_k: int = Field(default=8, ge=1, le=50, deprecated=True)


# ---------------------------------------------------------------------------
# Documents / timeline / cross-check / lab trends
# ---------------------------------------------------------------------------

@app.post("/api/v1/documents", status_code=201)
async def upload_documents(
    files: List[UploadFile] = File(...),
    confirm_name_mismatch: bool = Form(False),
    user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Uploads one or more documents (PDF/image) for the authenticated user.
    Extracts each, merges the results with any documents previously
    uploaded by this user, rebuilds the timeline, and re-runs cross-checking
    and lab trend tracking. The saved snapshot is what Q&A answers from, so
    the new documents are queryable as soon as this call returns.

    If a document's extracted patient identity (name, age, gender) doesn't
    look consistent with this account's document history — either against
    documents already on file, or against other documents in the same
    batch when there's no history yet — it is held back rather than
    silently merged: everything else in the batch still proceeds normally
    (still 201), and the response's `identity_review_needed` field
    describes which document(s) were held and why. See identity_guard.py
    for the matching/scoring rules (fuzzy name matching, inferred-birth-year
    age comparison, corroboration across signals) and the no-history/
    first-upload tie-break behavior. Resubmit just the held file(s) with
    confirm_name_mismatch=true to add them.
    """
    logger.info("upload_documents: user=%s received %d file(s)", user_id, len(files))
    if not files:
        raise HTTPException(400, "No files were uploaded.")

    # Pass 1: extract + validate every file/page first. Nothing is uploaded
    # to Cloudinary or written to Mongo until the whole batch passes, so a
    # bad file later in the batch never leaves an orphaned upload behind
    # for a good file earlier in it.
    per_file_pages: List[Tuple[Path, str, List[Tuple[str, Dict[str, Any]]]]] = []
    new_docs: List[Dict[str, Any]] = []

    # Loaded up front (rather than after the loop, where it used to be) so an
    # already-uploaded file can be recognised BEFORE it is extracted.
    existing_docs = db.load_documents(user_id)
    seen_hashes = {
        d["content_sha256"]: d for d in existing_docs if d.get("content_sha256")
    }
    duplicate_files_skipped: List[Dict[str, Any]] = []

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

            # Re-uploading a file this user already sent used to add a SECOND
            # copy of the same document: Cloudinary public_ids carry a random
            # suffix, so nothing upstream noticed, and the timeline ended up
            # holding one physical prescription two or three times. The
            # consequences were clinical, not cosmetic — every medication on
            # it then appeared under multiple (date, source_file) pairs, which
            # is exactly the shape detect_exact_duplicate_medications() and
            # the cross-check LLM both read as "prescribed twice", so the
            # patient was told to see a pharmacist about a double-dosing risk
            # that only existed because they uploaded the same photo twice.
            #
            # Checked BEFORE extraction, so a re-upload costs no vision call.
            content_sha256 = hashlib.sha256(content).hexdigest()
            already = seen_hashes.get(content_sha256)
            if already is not None:
                first_seen = already.get("uploaded_at") or "an earlier upload"
                logger.info(
                    "upload_documents: user=%s skipping '%s' — identical file already "
                    "on file as '%s' (uploaded %s, sha256=%s)",
                    user_id, upload.filename,
                    (already.get("_source") or {}).get("file", "unknown"),
                    first_seen, content_sha256[:12],
                )
                duplicate_files_skipped.append({
                    "filename": upload.filename,
                    "reason": "identical_file_already_uploaded",
                    "previously_uploaded_as": (already.get("_source") or {}).get("file"),
                    "previously_uploaded_at": already.get("uploaded_at"),
                    "message": (
                        f"'{upload.filename}' is byte-for-byte identical to a document "
                        "already in your records, so it was not added again. Nothing was "
                        "lost — the existing copy is still there."
                    ),
                })
                continue

            logger.info(
                "upload_documents: user=%s processing '%s' (%d bytes, sha256=%s)",
                user_id, upload.filename, len(content), content_sha256[:12],
            )
            try:
                # Off the event loop. process_document() blocks for ~44s on a
                # vision extraction, and running that inline in an async
                # handler stalls every other request the worker is serving —
                # including ones doing no model work at all. asyncio.to_thread
                # also makes the concurrent extraction below possible.
                result = await asyncio.to_thread(process_document, str(tmp_path))
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
            kept_pages: List[Tuple[str, Dict[str, Any]]] = []
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

                # Reject a document whose language could not be normalized
                # into the English fields cross-document matching depends on.
                # Accepting it would leave its medications unmatchable
                # against the rest of the record — a silent gap in the very
                # cross-check the upload exists to feed, so it fails loudly
                # here instead.
                try:
                    assert_supported_language(page, label)
                except UnsupportedLanguageError as e:
                    logger.warning(
                        "upload_documents: user=%s rejected '%s' (language=%s): %s",
                        user_id, label, e.detected_language, "; ".join(e.problems),
                    )
                    raise HTTPException(422, str(e))

                # Stored so a later upload of this same file is recognised.
                # Every page of a multi-page PDF carries its parent file's
                # hash — the unit of re-upload is the file, not the page.
                page["content_sha256"] = content_sha256
                kept_pages.append((label, page))

            if kept_pages:
                per_file_pages.append((tmp_path, upload.filename, kept_pages))
                # Also guards against the same file appearing twice within a
                # SINGLE batch, which the stored-document check alone can't
                # see because nothing has been written yet.
                seen_hashes[content_sha256] = {
                    "_source": {"file": upload.filename},
                    "uploaded_at": None,
                }

        if not per_file_pages:
            # Every file being a known duplicate is a SUCCESS, not an error:
            # the user's records already contain exactly what they just sent,
            # and nothing needs to change. Returning 422 here would tell them
            # their upload failed when it was simply unnecessary.
            if duplicate_files_skipped and len(duplicate_files_skipped) == len(files):
                logger.info(
                    "upload_documents: user=%s all %d file(s) were already on file — "
                    "nothing to add", user_id, len(files),
                )
                snapshot = db.load_patient_snapshot(user_id) or {}
                return {
                    "user_id": user_id,
                    "documents_added": 0,
                    "documents_total": len(existing_docs),
                    "timeline": snapshot.get("patient_timeline", {}),
                    "cross_check_report": snapshot.get("cross_check_report", {}),
                    "lab_trends": snapshot.get("lab_trends", {}),
                    "consult_triage": _fresh_triage(snapshot),
                    "duplicate_files_skipped": duplicate_files_skipped,
                    "antidote_reference_notes": [],
                    "indexed": True,
                }
            raise HTTPException(
                422,
                "No medical content found in the uploaded file(s) (all pages were "
                "demo/placeholder documents).",
            )

        # Check the newly-extracted patient identity (name, fuzzy-matched;
        # age, via inferred birth year so it tolerates the patient aging
        # between documents; gender) against this account's DOCUMENT
        # HISTORY (previously-stored documents only — never the account
        # holder's registered profile name, which is frequently not the
        # patient's own name). A brand new account has no history to
        # compare against, so a first-ever upload is only questioned
        # against itself: if every document in the batch agrees on one
        # patient, nothing is held back, no matter whose name that is; if
        # the batch itself disagrees (e.g. one file says "Ramesh", another
        # says "Suresh"), the larger group is treated as the baseline and
        # the rest are held. Matching documents proceed immediately —
        # mismatched ones are held out of THIS request entirely (not
        # uploaded to Cloudinary, not written to Mongo) rather than
        # blocking the whole batch. A single noisy signal (e.g. one
        # OCR-garbled name, a missing gender) never holds a document back
        # by itself — see identity_guard.py for the corroboration scoring.
        # Resubmitting just the held file(s) with confirm_name_mismatch=true
        # adds them without re-litigating whatever already went through.
        labeled_new_pages = [
            (label, page) for _, _, kept in per_file_pages for label, page in kept
        ]
        held_labels = set()
        mismatch_details: Optional[Dict[str, Any]] = None
        if not confirm_name_mismatch:
            partition = partition_identity_mismatch(labeled_new_pages, existing_docs)
            held_labels = partition["held_labels"]
            mismatch_details = partition["details"]
            if mismatch_details is not None:
                logger.warning(
                    "upload_documents: user=%s identity mismatch, holding %d page(s): %s",
                    user_id, len(held_labels), mismatch_details["message"],
                )

        # A file is held back in its entirety if ANY of its kept pages was
        # held — avoids uploading a multi-page document to Cloudinary with
        # only some of its pages merged into the timeline.
        proceeding_file_pages = [
            (tmp_path, filename, kept_pages)
            for tmp_path, filename, kept_pages in per_file_pages
            if not any(label in held_labels for label, _ in kept_pages)
        ]

        # Pass 2: everything validated — archive each original file to
        # Cloudinary once, and attach the resulting URL to every page that
        # came from it. Held-back files are skipped entirely; their temp
        # files are simply discarded when this `with` block exits.
        for tmp_path, filename, kept_pages in proceeding_file_pages:
            upload_info = storage.upload_patient_document(user_id, str(tmp_path), filename)
            for label, page in kept_pages:
                page["document_url"] = upload_info["document_url"]
                page["cloudinary_public_id"] = upload_info["cloudinary_public_id"]
                new_docs.append(page)

    all_docs = existing_docs + new_docs
    logger.info(
        "upload_documents: user=%s merged documents: +%d new, %d total",
        user_id, len(new_docs), len(all_docs),
    )

    timeline = build_patient_timeline(all_docs)

    # Look the medication list up in the reference graph BEFORE cross-checking,
    # so any finding about a drug the graph actually documents can be graded as
    # evidence-backed and cite its source, instead of being capped as
    # unverifiable model recall. Fail-open as always: no graph just means every
    # finding grades as model_knowledge, which is the honest default.
    # Both the printed name and the extracted ingredients are offered. The
    # printed name is what the prescription said ("Naloxone Hydrochloride
    # 400mcg"); `ingredients` is the cleaner form the extractor already
    # separated out, and classify_medication() prefers it for exactly this
    # reason. Sending only the printed name meant a combination product, or
    # anything printed with its salt, never reached the reference data.
    med_names = sorted({
        value
        for m in timeline.get("medications_timeline", [])
        for value in [m.get("name"), *(m.get("ingredients") or [])]
        if value
    })
    antidote_references: Dict[str, Dict[str, Any]] = {}
    try:
        logger.info(
            "upload_documents: user=%s querying antidote graph for %d medication name(s)",
            user_id, len(med_names),
        )
        antidote_references = lookup_antidote_references(med_names)
    except Exception as e:
        logger.warning(
            "upload_documents: user=%s antidote reference lookup skipped, continuing "
            "without it (findings will grade as unverified model knowledge): %s",
            user_id, e,
        )
    graph_backed_findings = graph_backed_findings_from_antidotes(antidote_references)

    # The cross-check is the single most expensive step in an upload (~44s on
    # a real record, about half the request). It is a pure function of the
    # medication timeline plus allergies, so an upload that leaves both
    # untouched — a lab report, a re-upload, a prescription for drugs already
    # on file — cannot get a different answer. Comparing a hash of those exact
    # inputs against the saved one turns those uploads into a snapshot read.
    cross_check_fingerprint = cross_check_inputs_fingerprint(timeline)
    previous = db.load_patient_snapshot(user_id)
    reusable = (
        previous
        and previous.get("cross_check_report")
        and previous.get("cross_check_fingerprint") == cross_check_fingerprint
    )

    if reusable:
        cross_check = previous["cross_check_report"]
        logger.info(
            "upload_documents: user=%s reusing saved cross-check — medication and "
            "allergy inputs are byte-identical (skipped one OpenAI call)", user_id,
        )
    elif not new_docs and previous and previous.get("cross_check_report"):
        # Nothing proceeded this request (every extracted document was held
        # back pending identity confirmation), so all_docs matches what is
        # already on file.
        cross_check = previous["cross_check_report"]
        logger.info(
            "upload_documents: user=%s no documents proceeded — reusing saved "
            "cross-check", user_id,
        )
    else:
        cross_check = await asyncio.to_thread(
            cross_check_prescriptions, timeline,
            graph_backed_findings=graph_backed_findings,
        )

    issue_count = sum(len(v) for v in cross_check.values() if isinstance(v, list))
    evidence = cross_check.get("evidence_summary") or {}
    timing = cross_check.get("timing_summary") or {}
    if timing:
        logger.info(
            "upload_documents: user=%s findings in time: %d concurrent, %d possible, "
            "%d never overlapped (historical), %d undatable; %d double-dose period(s)",
            user_id, timing.get("concurrent", 0), timing.get("possible", 0),
            timing.get("not_concurrent", 0), timing.get("unknown", 0),
            len(cross_check.get("concurrent_exposure") or []),
        )
    logger.info(
        "upload_documents: user=%s timeline rebuilt, cross-check found %d issue(s) "
        "(evidence: %d deterministic, %d reference-graph, %d unverified model knowledge)",
        user_id, issue_count, evidence.get("deterministic", 0),
        evidence.get("reference_graph", 0), evidence.get("model_knowledge", 0),
    )

    lab_trends = track_lab_trends(timeline)
    logger.info(
        "upload_documents: user=%s lab trend tracking found %d trend(s), %d test(s) with insufficient data",
        user_id, len(lab_trends["trends"]), len(lab_trends["insufficient_data"]),
    )

    # Route whatever the cross-check and trend tracking found to the
    # professional who can act on it. Reads only the findings computed above,
    # so it adds no clinical judgment of its own.
    consult_triage = triage_consultation(cross_check, lab_trends, timeline)
    logger.info(
        "upload_documents: user=%s consult triage: needed=%s type=%s urgency=%s confidence=%s",
        user_id, consult_triage["consult_needed"], consult_triage["consult_type"],
        consult_triage["urgency"], consult_triage["confidence"],
    )

    # Red flag for documents whose English fields were converted from another
    # language, or that were hard to read. Assessed over the WHOLE record, not
    # just this upload, so the banner reflects everything currently on file.
    translation_risk = assess_documents_translation_risk(all_docs)
    if translation_risk["flag"] != "none":
        logger.warning(
            "upload_documents: user=%s translation red flag=%s (high=%d review=%d)",
            user_id, translation_risk["flag"],
            translation_risk["counts"]["high"], translation_risk["counts"]["review"],
        )

    # Best-effort: note when a medication the patient is already on is
    # itself WHO-EML-listed under "Antidotes and other substances used in
    # poisonings" (e.g. naloxone, activated charcoal). Never fails the
    # upload if Neo4j is unreachable -- the reference note is an enrichment,
    # not part of the patient's own record.
    antidote_notes: List[Dict[str, Any]] = []
    try:
        # Reuses the lookup already performed above for evidence grading —
        # one round trip to the graph per upload, not two.
        antidote_notes = [
            {"medication": name, **ref}
            for name, ref in sorted(antidote_references.items())
        ]
        if antidote_notes:
            logger.info(
                "upload_documents: user=%s antidote graph matched %d of %d medication(s): %s",
                user_id, len(antidote_notes), len(med_names),
                ", ".join(n["medication"] for n in antidote_notes),
            )
        else:
            logger.info(
                "upload_documents: user=%s antidote graph checked %d medication(s), "
                "none are WHO-listed antidotes",
                user_id, len(med_names),
            )
    except Exception as e:
        # poisoning_kg/graph_db have already logged which step failed. This
        # records that the upload CONTINUED without the enrichment, which is
        # the part that matters when reading the request end to end — an
        # empty antidote_reference_notes below means "not checked", not
        # "checked and found nothing".
        logger.warning(
            "upload_documents: user=%s antidote reference lookup skipped, continuing "
            "without it (antidote_reference_notes will be empty): %s", user_id, e,
        )

    # Saving the snapshot is what makes this user's records answerable —
    # retrieval.py reads the snapshot directly, so there is no separate
    # index to build, fall out of sync, or fail independently of the write.
    db.insert_documents(user_id, new_docs)
    db.save_patient_snapshot(
        user_id, timeline, cross_check,
        lab_trends=lab_trends, consult_triage=consult_triage,
        cross_check_fingerprint=cross_check_fingerprint,
    )
    logger.info(
        "upload_documents: user=%s request complete: documents_added=%d documents_total=%d",
        user_id, len(new_docs), len(all_docs),
    )

    response: Dict[str, Any] = {
        "user_id": user_id,
        "documents_added": len(new_docs),
        "documents_total": len(all_docs),
        "timeline": timeline,
        "cross_check_report": cross_check,
        "lab_trends": lab_trends,
        "consult_triage": consult_triage,
        "translation_risk": translation_risk,
        "antidote_reference_notes": antidote_notes,
        # Present (and non-empty) when a re-uploaded file was recognised and
        # not added a second time.
        "duplicate_files_skipped": duplicate_files_skipped,
        # Retained for existing clients that branch on it. Always true now:
        # a successful upload is queryable by definition, since Q&A reads the
        # same snapshot this request just wrote.
        "indexed": True,
    }
    # Present only when one or more uploaded documents were held back
    # pending identity confirmation (see identity_guard.py) — everything
    # else in this response still reflects what WAS successfully added.
    # Resubmit just the held file(s) with confirm_name_mismatch=true to
    # add them.
    if mismatch_details is not None:
        response["identity_review_needed"] = mismatch_details
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


# ---------------------------------------------------------------------------
# Reference knowledge graph (Neo4j)
# ---------------------------------------------------------------------------

@app.post("/api/v1/knowledge-graph/antidotes", status_code=201)
async def upload_antidote_reference(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Ingests the "Antidotes and other substances used in poisonings"
    section of a WHO Model List of Essential Medicines PDF into the Neo4j
    reference graph. Extraction is deterministic table parsing, so the
    graph holds only what the document literally prints -- notably NOT
    which poison each antidote treats, which the source never states.

    Accepts both WHO lists: the main EML (adults) and the EMLc (children).
    Each is stored as its own :SourceDocument, so the two coexist and a
    drug on both keeps a separate listing (and dosage form) per document.
    Which population a PDF covers is read off its own title text.

    Unlike every other route here, what this writes is shared reference
    data, not per-patient data, so it is not scoped by user_id -- a valid
    token is still required to keep the write authenticated.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(
            400,
            f"Unsupported file type '{suffix or '(no extension)'}'. "
            "Antidote reference ingestion requires a PDF (table extraction).",
        )

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / file.filename
        tmp_path.write_bytes(await file.read())
        logger.info(
            "upload_antidote_reference: user=%s parsing '%s'", user_id, file.filename,
        )
        try:
            section = extract_antidote_section(str(tmp_path))
        except Exception as e:
            logger.error(
                "upload_antidote_reference: user=%s parse failed for '%s': %s",
                user_id, file.filename, e, exc_info=True,
            )
            raise HTTPException(422, f"Could not parse '{file.filename}': {e}")

    entries = section["entries"]
    if not entries:
        raise HTTPException(
            422,
            f"No 'Antidotes and other substances used in poisonings' section "
            f"found in '{file.filename}'.",
        )

    count = ingest_antidote_entries(section, source_document=file.filename)
    categories = sorted({e["subsection"] for e in entries if e["subsection"]})
    logger.info(
        "upload_antidote_reference: user=%s ingested %d entrie(s) from '%s' (population=%s)",
        user_id, count, file.filename, section["population"],
    )
    return {
        "source_document": file.filename,
        "population": section["population"],
        "entries_ingested": count,
        "categories": categories,
    }


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


@app.get("/api/v1/risk-timeline")
async def get_risk_timeline(user_id: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Returns this user's safety findings placed in time — which risks were
    live during which dates, most recent period first, plus any period where
    two prescriptions supplied the same ingredient at once.

    Two drugs only interact if they were taken together, so findings whose
    courses never overlapped are grouped separately as history rather than
    presented as current risks. Computed from the printed prescription dates
    and durations — no model call."""
    snapshot = db.load_patient_snapshot(user_id)
    if snapshot is None:
        raise HTTPException(404, "No records found for this user.")

    timeline = snapshot.get("patient_timeline") or {}
    cross_check = snapshot.get("cross_check_report") or {}
    return {
        "calendar": risk_calendar(cross_check, timeline),
        "concurrent_exposure": concurrent_exposure(timeline),
        "treatment_windows": [
            {**w, "start": w["start"].isoformat() if w["start"] else None,
             "end": w["end"].isoformat() if w["end"] else None}
            for w in build_treatment_windows(timeline)
        ],
        "timing_summary": cross_check.get("timing_summary") or {},
    }


def _fresh_triage(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """The snapshot's cached triage if it was produced by the CURRENT version
    of consult_triage, otherwise a fresh one.

    Checking only that the key EXISTS — which both read paths used to do —
    pins a user to whatever wording was current when their snapshot was
    written. Recomputing is cheap here: the findings are already stored, and
    triage over stored findings is rule-based, so no model call is made.
    """
    cached = snapshot.get("consult_triage")
    if isinstance(cached, dict) and cached.get("output_version") == TRIAGE_OUTPUT_VERSION:
        return cached
    if cached:
        logger.info(
            "consult triage: cached output_version=%r is stale (current %r) — recomputing",
            cached.get("output_version"), TRIAGE_OUTPUT_VERSION,
        )
    return triage_consultation(
        snapshot.get("cross_check_report") or {},
        snapshot.get("lab_trends") or {},
        snapshot.get("patient_timeline") or {},
        use_llm=False,
    )


@app.get("/api/v1/consult-triage")
async def get_consult_triage(user_id: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Returns who the authenticated user should consult about what was found
    in their records — a pharmacist or a doctor, how soon, and for a doctor,
    which specialty — with a confidence score inherited from the finding that
    triggered each referral.

    `consult_needed: false` means these automated checks found no trigger; it
    is not a clean bill of health (see the `summary` and `note` fields).

    Recomputed on the fly from the saved cross-check and lab trends for
    snapshots saved before this field existed."""
    snapshot = db.load_patient_snapshot(user_id)
    if snapshot is None:
        raise HTTPException(404, "No records found for this user.")
    return _fresh_triage(snapshot)


# ---------------------------------------------------------------------------
# Single-shot Q&A (Phase 1)
# ---------------------------------------------------------------------------

@app.post("/api/v1/qa")
async def qa(body: QARequest, user_id: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Answers one question grounded in the authenticated user's processed
    records, with no session/conversation state (caller manages chat_history,
    if any). Without a session there is no entity focus to carry over, so
    prefer the /sessions endpoints for follow-up questions."""
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
    """Asks one question within an existing conversation session. The
    question is rewritten into a self-contained query using prior turns, and
    the entities the conversation is already about (medications, lab tests,
    documents) are carried over so a follow-up like "was that safe?" stays
    anchored to the right subject across documents; the answer is then
    generated against the original question + history. 404s if the session
    doesn't exist (create it via POST /sessions first) or belongs to a
    different user."""
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
