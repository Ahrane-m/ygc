"""
Conversational Q&A Layer (Phase 2)
=========================================
Builds on retrieval.py's single-shot answer_question() to support real
multi-turn conversations about a patient's medical timeline, without
changing retrieval.py's behavior for callers that don't use sessions.

THE PROBLEM: FOLLOW-UPS SPAN DOCUMENTS
--------------------------------------
"Was that safe?" is meaningless on its own. It's only answerable if you know
(a) what "that" refers to, and (b) which documents establish it — and the
answer usually has to cross documents: the drug was prescribed in one, the
allergy noted in another, the lab that makes it risky in a third.

Three mechanisms handle this, in increasing order of reliability:

  1. QUERY REWRITING (LLM) — turns "was that safe?" into a self-contained
     query. Good at natural language, but it's a model call that can fail or
     drop a detail, so nothing safety-critical depends on it alone.

  2. ENTITY FOCUS (deterministic) — the session tracks WHICH medications,
     lab tests and documents the conversation is actually about, resolved by
     exact matching against the patient's own record vocabulary. Focus is
     passed to retrieval so the subject of a follow-up is pinned into
     context as established fact, not re-inferred every turn. This is what
     keeps "and the other one?" or "what about in March?" anchored.

  3. WHOLE-RECORD CONTEXT (retrieval.py) — because the patient's full record
     usually fits the context window, most follow-ups can see every document
     at once regardless of how well (1) and (2) did. Focus then matters only
     for large records, where it decides what survives narrowing.

Sessions are persisted in MongoDB (db.py) rather than held in process
memory, so a follow-up still resolves after a restart or when the API runs
more than one worker. When no database is configured — the CLI path — they
fall back to an in-memory registry.

Env:
    OPENAI_API_KEY   (same key used by the rest of the pipeline)
    MONGODB_URI      (optional — sessions become in-memory without it)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAIError

from medical_extractor import client, MODEL
import retrieval

# Cheap/fast model for query rewriting and summarization — these are short,
# low-stakes generations, not the main answer synthesis.
REWRITE_MODEL = MODEL

SUMMARIZE_AFTER_TOTAL_TURNS = 20  # start summarizing once a session grows past this
KEEP_RECENT_TURNS_VERBATIM = 6    # ...but always keep this many most-recent turns as-is

# How many turns an entity stays "in focus" after it was last mentioned.
# Long enough to survive a couple of intervening clarifications, short enough
# that a conversation which has moved on isn't dragged back to an old drug.
FOCUS_TURN_MEMORY = 6

FOCUS_FIELDS = ("medications", "lab_tests", "source_files")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 1. Session management
# ---------------------------------------------------------------------------

class ConversationSession:
    """
    Holds the turn history and entity focus for one (patient_key,
    session_id) conversation. State lives in MongoDB when it's available and
    in process memory otherwise; `save()` is what reconciles the two.
    """

    def __init__(self, patient_key: str, session_id: str):
        self.patient_key = patient_key
        self.session_id = session_id
        self.turns: List[Dict[str, Any]] = []  # full, untrimmed transcript
        self._summary: Optional[str] = None
        self._summary_covers_up_to = 0  # index into self.turns the cached summary accounts for

    # -- turn recording ------------------------------------------------

    def add_user_turn(self, text: str, entities: Optional[Dict[str, List[str]]] = None) -> None:
        """Appends a user turn, tagged with the record entities it named.
        Those tags are what get_focus() later reads — resolved once, when
        the vocabulary is at hand, rather than re-derived per turn."""
        self.turns.append({
            "role": "user",
            "content": text,
            "timestamp": _now_iso(),
            "entities": entities or {},
        })

    def add_assistant_turn(self, answer: Dict[str, Any]) -> None:
        """Appends an assistant turn. Stores the "answer" text as the turn's
        content (what a human would read), plus the source files it cited —
        those files are part of what the conversation is "about", so they
        feed focus alongside entities the user named explicitly."""
        content = answer.get("answer", "") if isinstance(answer, dict) else str(answer)
        cited_files = []
        if isinstance(answer, dict):
            cited_files = [
                s.get("source_file")
                for s in answer.get("sources") or []
                if isinstance(s, dict) and s.get("source_file")
            ]
        self.turns.append({
            "role": "assistant",
            "content": content,
            "timestamp": _now_iso(),
            "entities": {"source_files": cited_files},
        })

    # -- focus ---------------------------------------------------------

    def get_focus(self, turn_memory: int = FOCUS_TURN_MEMORY) -> Dict[str, List[str]]:
        """
        The entities this conversation is currently about, gathered from the
        last `turn_memory` turns, most-recently-mentioned first.

        Deliberately deterministic: these are exact matches against the
        patient's own record vocabulary, recorded at the time each turn was
        processed. A follow-up's subject is therefore recalled, not guessed,
        which is why it doesn't degrade when the rewrite model has a bad day.
        """
        focus: Dict[str, List[str]] = {field: [] for field in FOCUS_FIELDS}
        for turn in reversed(self.turns[-turn_memory:] if turn_memory else self.turns):
            for field in FOCUS_FIELDS:
                for value in (turn.get("entities") or {}).get(field) or []:
                    if value and value not in focus[field]:
                        focus[field].append(value)
        return focus

    # -- history for prompting ----------------------------------------

    def get_history(self, max_turns: int = 6) -> List[Dict[str, str]]:
        """
        Returns the last `max_turns` turns formatted as {"role", "content"}
        dicts, ready to splice into a chat_history list for prompting.

        Once the session exceeds SUMMARIZE_AFTER_TOTAL_TURNS total turns,
        everything older than the most recent KEEP_RECENT_TURNS_VERBATIM
        turns is collapsed into a single "conversation summary so far"
        system message (see summarize_old_turns), which is prepended ahead
        of the verbatim recent turns. This keeps prompt size bounded
        regardless of how long the conversation runs.
        """
        if len(self.turns) > SUMMARIZE_AFTER_TOTAL_TURNS:
            cutoff = len(self.turns) - KEEP_RECENT_TURNS_VERBATIM
            if self._summary_covers_up_to < cutoff:
                self._summary = summarize_old_turns(self.turns[:cutoff])
                self._summary_covers_up_to = cutoff
            recent = self.turns[cutoff:]
            if max_turns < len(recent):
                recent = recent[-max_turns:]
            formatted = [{
                "role": "system",
                "content": f"Conversation summary so far: {self._summary}",
            }]
            formatted.extend({"role": t["role"], "content": t["content"]} for t in recent)
            return formatted

        recent = self.turns[-max_turns:] if max_turns else []
        return [{"role": t["role"], "content": t["content"]} for t in recent]

    def get_full_history(self) -> List[Dict[str, Any]]:
        """Returns the complete, untrimmed transcript (with timestamps and
        resolved entities) for logging/export. Unlike get_history(), this is
        never summarized or truncated — summarization only affects what's
        sent to the LLM for prompting, not what's retained."""
        return list(self.turns)

    # -- persistence ---------------------------------------------------

    def save(self) -> None:
        """Persists this session. Never raises: with no database configured
        (the CLI path) the in-memory registry is already authoritative, so a
        failed write costs nothing but durability."""
        try:
            import db

            db.save_session(
                self.patient_key,
                self.session_id,
                self.turns,
                focus=self.get_focus(),
                summary=self._summary,
                summary_covers_up_to=self._summary_covers_up_to,
            )
        except Exception as e:
            print(f"  Session not persisted (in-memory only): {e}")

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "ConversationSession":
        session = cls(record["user_id"], record["session_id"])
        session.turns = record.get("turns") or []
        session._summary = record.get("summary")
        session._summary_covers_up_to = record.get("summary_covers_up_to") or 0
        return session


# In-memory registry. Acts as the cache in front of Mongo, and as the only
# store when no database is configured.
_SESSIONS: Dict[Tuple[str, str], ConversationSession] = {}


def _load_persisted(patient_key: str, session_id: str) -> Optional[ConversationSession]:
    try:
        import db

        record = db.load_session(patient_key, session_id)
    except Exception:
        return None
    return ConversationSession.from_record(record) if record else None


def get_or_create_session(patient_key: str, session_id: str) -> ConversationSession:
    """Fetches the ConversationSession for (patient_key, session_id) —
    from memory, else from the database, else newly created and registered."""
    key = (patient_key, session_id)
    session = _SESSIONS.get(key)
    if session is None:
        session = _load_persisted(patient_key, session_id) or ConversationSession(
            patient_key, session_id
        )
        _SESSIONS[key] = session
        session.save()
    return session


def get_session(patient_key: str, session_id: str) -> Optional[ConversationSession]:
    """Fetches the ConversationSession for (patient_key, session_id) without
    creating one, checking the database if it isn't in memory (so a session
    created before a restart, or by another API worker, still resolves).
    Returns None if no such session exists — used by callers (e.g. the HTTP
    API) that need to distinguish "unknown session" (404) from "brand new
    session" (auto-create)."""
    key = (patient_key, session_id)
    session = _SESSIONS.get(key)
    if session is not None:
        return session
    session = _load_persisted(patient_key, session_id)
    if session is not None:
        _SESSIONS[key] = session
    return session


def delete_session(patient_key: str, session_id: str) -> bool:
    """Removes a session from both the in-memory registry and the database.
    Returns True if it existed in either."""
    removed = _SESSIONS.pop((patient_key, session_id), None) is not None
    try:
        import db

        removed = db.delete_session(patient_key, session_id) or removed
    except Exception:
        pass
    return removed


# ---------------------------------------------------------------------------
# 2. Query rewriting
# ---------------------------------------------------------------------------

REWRITE_SYSTEM_PROMPT = """
You rewrite a patient's follow-up question into a single, fully
self-contained search query used to select relevant parts of that patient's
medical records (medications, lab results, clinical notes, allergies) across
multiple documents.

Rules:
- Resolve pronouns and vague references ("that", "it", "the other one",
  "this medication") into the specific medication/lab/date/event they refer
  to, using the conversation history and the "currently being discussed"
  entities provided.
- Preserve ALL safety-relevant framing from the original question. If the
  patient is asking about risk, danger, safety, interactions, allergies, or
  dosage, the rewritten query must keep that risk framing explicit (e.g.
  keep words like "safe", "danger", "interact", "allergy") — never rewrite
  it into a neutral factual lookup that loses the risk framing.
- Preserve comparisons across documents or time ("since", "before", "still",
  "changed", "the newer one") — these decide which documents get selected,
  so dropping them silently narrows the answer to a single document.
- Do not answer the question. Do not add information that isn't implied by
  the conversation.
- Output ONLY the rewritten query text — no quotes, no preamble, no
  explanation.
"""


def rewrite_query_with_context(
    question: str,
    history: List[Dict[str, str]],
    focus: Optional[Dict[str, List[str]]] = None,
) -> str:
    """
    Turns an ambiguous follow-up question into a self-contained query, using
    recent conversation history and the session's entity focus. The result
    drives record selection — the original `question` is left untouched for
    the final answer-generation prompt.

    If there's no history and no focus (first turn in a session), rewriting
    is skipped and `question` is returned as-is. If the rewrite call fails
    for any reason (auth, rate limit, network), falls back to the raw
    `question` rather than raising — a rewrite failure degrades the query but
    never blocks the answer, and the session's focus is passed to retrieval
    separately, so the subject of the follow-up survives regardless.
    """
    if not question or not question.strip():
        return question
    has_focus = any((focus or {}).get(field) for field in FOCUS_FIELDS)
    if not history and not has_focus:
        return question

    messages: List[Dict[str, str]] = [{"role": "system", "content": REWRITE_SYSTEM_PROMPT}]
    messages.extend(history)
    focus_note = ""
    if has_focus:
        parts = [
            f"{field.replace('_', ' ')}: {', '.join(values)}"
            for field in FOCUS_FIELDS
            for values in [(focus or {}).get(field) or []]
            if values
        ]
        focus_note = "Currently being discussed — " + "; ".join(parts) + "\n\n"
    messages.append({
        "role": "user",
        "content": (
            f"{focus_note}New follow-up question to rewrite: {question}\n\n"
            "Output only the rewritten, self-contained search query."
        ),
    })

    try:
        response = client.chat.completions.create(model=REWRITE_MODEL, messages=messages)
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten if rewritten else question
    except OpenAIError as e:
        print(f"  Query rewrite failed, falling back to raw question for retrieval: {e}")
        return question


# ---------------------------------------------------------------------------
# 3. Summarization of older turns (context-window safety)
# ---------------------------------------------------------------------------

SUMMARIZE_SYSTEM_PROMPT = """
You compress the older portion of a patient/assistant conversation about a
patient's medical records into a short, dense summary that will be used as
background context for later turns in the same conversation.

Rules:
- Preserve every safety-relevant detail: allergies mentioned, medications
  and lab results discussed, and any risk/interaction/dosage questions
  raised — including whether a professional consult was recommended in the
  response.
- Preserve which documents/dates were being compared, since later follow-ups
  often refer back to them.
- Do not soften, omit, or generalize away risk-related content.
- Do not add new information, speculation, or medical advice.
- Output 3-6 sentences of plain prose. No headers, no bullet points, no
  markdown.
"""


def summarize_old_turns(turns: List[Dict[str, Any]]) -> str:
    """
    Collapses a list of older {"role", "content"} turns into a single
    compact summary string via a cheap LLM call, preserving safety-relevant
    details (allergies, risk questions, professional-consult flags) rather
    than softening or dropping them.

    Falls back to a crude truncated concatenation of the turns (instead of
    raising) if the summarization call fails, so older context degrades
    gracefully rather than being silently lost.
    """
    if not turns:
        return ""

    transcript = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)
    messages = [
        {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Conversation to summarize:\n\n{transcript}"},
    ]

    try:
        response = client.chat.completions.create(model=REWRITE_MODEL, messages=messages)
        return (response.choices[0].message.content or "").strip()
    except OpenAIError as e:
        print(f"  Conversation summarization failed, using raw fallback: {e}")
        return transcript[:2000]


# ---------------------------------------------------------------------------
# 4. Conversational answer function
# ---------------------------------------------------------------------------

def ask(session: ConversationSession, question: str, top_k: int = 8) -> Dict[str, Any]:
    """
    Answers one turn of a multi-turn conversation about session.patient_key,
    grounded in that patient's records across every document on file.

    1. Loads the patient's record once, and derives its entity vocabulary.
    2. Pulls recent history and the conversation's current entity focus.
    3. Rewrites `question` into a self-contained query using both — skipped
       on a first turn with no focus.
    4. Resolves which record entities the rewritten query and the raw
       question actually name (exact match against the vocabulary), and
       merges them into the focus passed to retrieval.
    5. Calls retrieval.answer_question() with the rewritten query driving
       record selection, but the ORIGINAL question + history passed through
       for answer generation, so the model responds to what was actually
       asked.
    6. Records both turns (tagged with their entities, which become the next
       turn's focus) and persists the session.

    Returns the same JSON shape as retrieval.answer_question(), plus:
        "rewritten_query" — the query actually used for record selection
        "focus"           — the entities this turn was resolved against
    both for debugging and demo transparency.

    Raises ValueError for an empty question.
    """
    if not question or not question.strip():
        raise ValueError("question is required and cannot be empty.")

    record = retrieval.load_patient_record(session.patient_key)
    vocabulary = retrieval.build_record_vocabulary(record) if record else {}

    history = session.get_history()
    prior_focus = session.get_focus()
    rewritten_query = rewrite_query_with_context(question, history, prior_focus)

    # Entities named in this turn, resolved against the patient's own record.
    # Both the raw question and the rewrite are scanned: the rewrite may
    # introduce the resolved name ("that" -> "Metformin"), while the raw
    # question may keep wording the rewrite dropped.
    turn_entities: Dict[str, List[str]] = {field: [] for field in FOCUS_FIELDS}
    if vocabulary:
        from_question = retrieval.match_vocabulary(question, vocabulary)
        from_rewrite = retrieval.match_vocabulary(rewritten_query, vocabulary)
        turn_entities = {
            field: list(dict.fromkeys(from_question[field] + from_rewrite[field]))
            for field in FOCUS_FIELDS
        }

    # Focus for THIS turn = what it names, ahead of what the conversation was
    # already about. Ordering matters: the current subject should outrank a
    # carried-over one when a narrowed record can only fit some of them.
    effective_focus = {
        field: list(dict.fromkeys(turn_entities.get(field, []) + prior_focus.get(field, [])))
        for field in FOCUS_FIELDS
    }

    result = retrieval.answer_question(
        patient_key=session.patient_key,
        question=question,
        chat_history=history,
        top_k=top_k,
        retrieval_query=rewritten_query,
        record=record,
        focus=effective_focus,
    )

    session.add_user_turn(question, entities=turn_entities)
    session.add_assistant_turn(result)
    session.save()

    result = dict(result)
    result["rewritten_query"] = rewritten_query
    result["focus"] = effective_focus
    return result
