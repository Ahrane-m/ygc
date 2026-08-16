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
                        + cross_check_report + lab_trends + consult_triage
                        (mirrors what the CLI writes to
                        patient_report_<name>.json).
    conversation_sessions  one record per (user_id, session_id): the Q&A turn
                        history plus the conversation's entity focus. Stored
                        rather than kept in process memory so follow-up
                        questions survive a restart and work across more than
                        one API worker.

Env:
    MONGODB_URI   connection string (database name taken from its path)
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import MongoClient
from pymongo.collection import Collection

_client: Optional[MongoClient] = None


# pymongo's default server-selection timeout is 30s. That's far too long to
# block on now that a conversation turn writes here (conversation.py persists
# the session on every message) — an unreachable database would hang a
# request rather than fail it. Bounded so callers that treat persistence as
# best-effort degrade in seconds instead of half a minute.
SERVER_SELECTION_TIMEOUT_MS = int(os.environ.get("MONGODB_TIMEOUT_MS", "8000"))


def _get_db():
    global _client
    if _client is None:
        _client = MongoClient(
            os.environ["MONGODB_URI"],
            serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
        )
    return _client.get_default_database()


def _documents() -> Collection:
    return _get_db()["documents"]


def _snapshots() -> Collection:
    return _get_db()["patient_snapshots"]


def _sessions() -> Collection:
    return _get_db()["conversation_sessions"]


def ensure_indexes() -> None:
    """Called once at API startup. user_id is the access-control boundary
    for every collection, so all are indexed on it; patient_snapshots is
    additionally unique per user_id since it's a single materialized view,
    and conversation_sessions is unique per (user_id, session_id) — the
    compound key also means a session_id guessed from another user can never
    match, since the lookup is always scoped by the authenticated user_id."""
    _documents().create_index("user_id")
    _snapshots().create_index("user_id", unique=True)
    _sessions().create_index([("user_id", 1), ("session_id", 1)], unique=True)


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
    consult_triage: Optional[Dict[str, Any]] = None,
) -> None:
    """Upserts the merged timeline + cross-check report (+ lab trends and
    consultation triage, if computed) for this user."""
    fields: Dict[str, Any] = {
        "user_id": user_id,
        "patient_timeline": timeline,
        "cross_check_report": cross_check,
        "updated_at": _now_iso(),
    }
    if lab_trends is not None:
        fields["lab_trends"] = lab_trends
    if consult_triage is not None:
        fields["consult_triage"] = consult_triage
    _snapshots().update_one({"user_id": user_id}, {"$set": fields}, upsert=True)


# ---------------------------------------------------------------------------
# Conversation sessions
# ---------------------------------------------------------------------------

def load_session(user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
    """Loads {"turns", "focus"} for one conversation, or None if this user
    has no such session. Scoped by user_id, so one user's session_id can
    never read another's conversation."""
    return _sessions().find_one(
        {"user_id": user_id, "session_id": session_id}, {"_id": 0}
    )


def save_session(
    user_id: str,
    session_id: str,
    turns: List[Dict[str, Any]],
    focus: Optional[Dict[str, Any]] = None,
    summary: Optional[str] = None,
    summary_covers_up_to: int = 0,
) -> None:
    """Upserts a conversation's full turn history and entity focus. The
    cached summary is stored alongside so a long conversation doesn't have
    to be re-summarized after a restart."""
    _sessions().update_one(
        {"user_id": user_id, "session_id": session_id},
        {
            "$set": {
                "user_id": user_id,
                "session_id": session_id,
                "turns": turns,
                "focus": focus or {},
                "summary": summary,
                "summary_covers_up_to": summary_covers_up_to,
                "updated_at": _now_iso(),
            },
            "$setOnInsert": {"created_at": _now_iso()},
        },
        upsert=True,
    )


def delete_session(user_id: str, session_id: str) -> bool:
    """Deletes one conversation. Returns True if a session was removed."""
    return _sessions().delete_one(
        {"user_id": user_id, "session_id": session_id}
    ).deleted_count > 0
