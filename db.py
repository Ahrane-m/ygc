"""
MongoDB persistence (Phase 4)
=========================================
Replaces the local patient_docs_*.json / patient_report_*.json files used
by the CLI (see medical_extractor.py) with per-user, access-controlled
storage in MongoDB, for the HTTP API only. Every read/write is scoped by
user_id — there is no query in this module that can return another user's
data.

Two collections (database name comes from MONGODB_URI, same "mediscan" DB
the existing `users` collection already lives in):

    documents          one record per extracted page/file, includes the
                        Cloudinary document_url — no raw file bytes, no
                        OpenAI request/response payloads, no access tokens.
    patient_snapshots   one record per user: the last-built patient_timeline
                        + cross_check_report + lab_trends (mirrors what the
                        CLI writes to patient_report_<name>.json).

Env:
    MONGODB_URI   connection string (database name taken from its path)
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import MongoClient
from pymongo.collection import Collection

_client: Optional[MongoClient] = None


def _get_db():
    global _client
    if _client is None:
        _client = MongoClient(os.environ["MONGODB_URI"])
    return _client.get_default_database()


def _documents() -> Collection:
    return _get_db()["documents"]


def _snapshots() -> Collection:
    return _get_db()["patient_snapshots"]


def ensure_indexes() -> None:
    """Called once at API startup. user_id is the access-control boundary
    for both collections, so both are indexed on it; patient_snapshots is
    additionally unique per user_id since it's a single materialized view."""
    _documents().create_index("user_id")
    _snapshots().create_index("user_id", unique=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_documents(user_id: str) -> List[Dict[str, Any]]:
    """Loads every previously-saved document for this user, oldest first —
    used to merge with newly-uploaded documents before rebuilding the
    timeline. Returns [] if this user has never uploaded anything."""
    cursor = _documents().find({"user_id": user_id}, {"_id": 0}).sort("uploaded_at", 1)
    return list(cursor)


def insert_documents(user_id: str, docs: List[Dict[str, Any]]) -> None:
    """Appends newly-extracted documents for this user (append-only — never
    rewrites or touches this user's existing documents). No-op on an empty
    list."""
    if not docs:
        return
    now = _now_iso()
    records = [{**d, "user_id": user_id, "uploaded_at": now} for d in docs]
    _documents().insert_many(records)


def load_patient_snapshot(user_id: str) -> Optional[Dict[str, Any]]:
    """Loads the {"patient_timeline", "cross_check_report"} snapshot last
    saved for this user, or None if they've never been processed."""
    return _snapshots().find_one({"user_id": user_id}, {"_id": 0})


def save_patient_snapshot(
    user_id: str,
    timeline: Dict[str, Any],
    cross_check: Dict[str, Any],
    lab_trends: Optional[Dict[str, Any]] = None,
) -> None:
    """Upserts the merged timeline + cross-check report (+ lab trends, if
    computed) for this user."""
    fields: Dict[str, Any] = {
        "user_id": user_id,
        "patient_timeline": timeline,
        "cross_check_report": cross_check,
        "updated_at": _now_iso(),
    }
    if lab_trends is not None:
        fields["lab_trends"] = lab_trends
    _snapshots().update_one({"user_id": user_id}, {"$set": fields}, upsert=True)
