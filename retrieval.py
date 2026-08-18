"""
Structured Retrieval + Q&A Layer (Phase 1)
=========================================
Sits on top of the ALREADY-EXTRACTED structured JSON produced by
medical_extractor.py — specifically the per-patient timeline returned by
build_patient_timeline(). It does NOT re-read raw documents.

WHY THERE IS NO VECTOR STORE HERE
---------------------------------
This module used to chunk the timeline, embed each chunk, and retrieve the
top_k nearest chunks from a per-patient Chroma collection. That was removed
deliberately, not for lack of infrastructure:

  * The retrieval unit is one patient's own record — tens to a few hundred
    entries, not a corpus. It fits in a model context window whole.
  * Every headline question this product answers is a COMPLETENESS question,
    not a similarity one. "What am I taking?" needs every medication, not the
    8 nearest. "Has my dose changed?" needs every occurrence of one drug
    across every document. "What changed since March?" needs a date slice.
    Top-k cosine similarity silently drops exactly the evidence that makes
    those answers correct — and a dropped medication in a drug-interaction
    answer is a safety failure, not a relevance miss.
  * Cross-document reasoning needs document PROVENANCE and ORDERING preserved.
    Embedding collapses both.

So retrieval here is deterministic assembly instead of approximate search:
load the patient's saved snapshot, render it into a context that is grouped
by document AND rolled up per entity across documents, and hand the whole
thing to the answering model. When a record is too large for the budget, a
planner narrows it down (see build_retrieval_plan) — but the common case
makes ZERO extra API calls, where the old path always paid for an embedding.

Pipeline:
    patient snapshot (Mongo, or the CLI's local JSON report)
      -> render document manifest + allergies + medication rollups
         + lab series w/ trends + clinical notes + safety flags
      -> (only if over budget) plan which entities matter, re-render narrowed
      -> answer strictly from that context, with a post-hoc safety guard

Env:
    OPENAI_API_KEY   (same key used by medical_extractor.py)
    MONGODB_URI      (optional here — falls back to the CLI's local report)
    QA_CONTEXT_BUDGET_CHARS  (optional, default 48000)
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from openai import OpenAIError

from medical_extractor import client, MODEL
from reference_intervals import canonical_test

CHAT_MODEL = MODEL      # reuse the same chat model configured in medical_extractor.py
PLANNER_MODEL = MODEL   # only called when a record exceeds the context budget

# Roughly 12k tokens at ~4 chars/token — comfortably inside the chat model's
# window while leaving room for the system prompt, history, and the answer.
DEFAULT_CONTEXT_BUDGET_CHARS = int(os.environ.get("QA_CONTEXT_BUDGET_CHARS", "48000"))

MAX_CLINICAL_NOTE_CHARS = 1200  # per note, when notes have to be trimmed
LOW_CONFIDENCE_THRESHOLD = 0.6  # at/below this, we always recommend a professional


# ---------------------------------------------------------------------------
# 1. Loading a patient's record
# ---------------------------------------------------------------------------

def load_patient_record(patient_key: str) -> Optional[Dict[str, Any]]:
    """
    Loads the saved snapshot for one patient as
    {"patient_timeline", "cross_check_report", "lab_trends"}.

    Two storage backends, tried in order, because two callers exist:
      1. MongoDB (db.load_patient_snapshot) — the HTTP API path, where
         patient_key IS the authenticated user_id.
      2. The local patient_report_<name>.json the CLI writes — the
         medical_extractor.py __main__ path, which has no Mongo.

    Returns None if neither backend has anything for this patient. A Mongo
    failure (unset MONGODB_URI, unreachable server) is not fatal — it falls
    through to the file backend, so the CLI works with no database at all.
    """
    try:
        import db

        snapshot = db.load_patient_snapshot(patient_key)
        if snapshot:
            return snapshot
    except Exception:
        pass  # no Mongo configured/reachable — try the CLI's local report

    try:
        from medical_extractor import load_patient_report

        report = load_patient_report(patient_key)
        if report:
            return report
    except Exception:
        pass

    return None


def _timeline_of(record: Dict[str, Any]) -> Dict[str, Any]:
    return record.get("patient_timeline") or {}


def _has_content(record: Optional[Dict[str, Any]]) -> bool:
    """True if this record has anything worth answering from."""
    if not record:
        return False
    timeline = _timeline_of(record)
    return bool(
        timeline.get("visits")
        or timeline.get("medications_timeline")
        or timeline.get("lab_results_timeline")
        or timeline.get("known_allergies")
    )


# ---------------------------------------------------------------------------
# 2. Cross-document rollups
#
# The heart of cross-document Q&A. A raw timeline lists medications and labs
# document-by-document; these functions re-key them BY ENTITY so that every
# occurrence of one drug (or one test) across every document sits together,
# in date order, with the change between occurrences computed in code rather
# than left for the model to notice.
# ---------------------------------------------------------------------------

def _med_group_key(med: Dict[str, Any]) -> Tuple[str, ...]:
    """Groups by normalized active ingredient(s) so the same drug under two
    brand names — or printed in two languages — lands in one group. Falls
    back to the printed name when extraction found no ingredients."""
    ingredients = tuple(sorted(i.strip().lower() for i in (med.get("ingredients") or []) if i and i.strip()))
    if ingredients:
        return ingredients
    return ((med.get("name") or "unknown").strip().lower(),)


def _med_display_name(meds: Sequence[Dict[str, Any]]) -> str:
    """Prefers the most recent printed name, which is what the patient will
    recognise, over the normalized ingredient list."""
    for med in reversed(list(meds)):
        if med.get("name"):
            return med["name"]
    return "unknown medication"


def _describe_dose(med: Dict[str, Any]) -> str:
    value, unit = med.get("dosage_value"), med.get("dosage_unit")
    if value is not None and unit:
        return f"{value} {unit}"
    return med.get("dosage") or "dose not recorded"


def _describe_frequency(med: Dict[str, Any]) -> str:
    if med.get("is_as_needed"):
        return "as needed (PRN)"
    per_day = med.get("frequency_per_day")
    if per_day is not None:
        return f"{per_day}x per day"
    return med.get("frequency") or "frequency not recorded"


def _sort_by_date(entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chronological, with undated entries last (same convention as
    build_patient_timeline)."""
    return sorted(entries, key=lambda e: e.get("date") or "9999-99-99")


def group_medications(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Rolls the flat medications_timeline up into one group per drug:

        {"key", "display_name", "ingredients", "occurrences" (chronological),
         "source_files", "dose_changes", "frequency_changes"}

    dose_changes / frequency_changes are computed here, in code, rather than
    inferred by the model — "did my dose change across these documents?" is
    an exact comparison of normalized numbers, and the answer should not
    depend on the model noticing it.
    """
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for med in timeline.get("medications_timeline", []):
        groups.setdefault(_med_group_key(med), []).append(med)

    rolled: List[Dict[str, Any]] = []
    for key, meds in groups.items():
        occurrences = _sort_by_date(meds)

        # Only compare readings that actually normalized — an unparsed dose
        # is missing data, not a change.
        doses = [
            f"{m['dosage_value']} {m['dosage_unit']}"
            for m in occurrences
            if m.get("dosage_value") is not None and m.get("dosage_unit")
        ]
        freqs = [
            f"{m['frequency_per_day']}x per day"
            for m in occurrences
            if m.get("frequency_per_day") is not None
        ]

        # The key IS the ingredient tuple whenever extraction found
        # ingredients; it falls back to the printed name otherwise, which is
        # not an ingredient list and must not be presented as one.
        grouped_by_ingredient = any(m.get("ingredients") for m in occurrences)

        rolled.append({
            "key": key,
            "display_name": _med_display_name(occurrences),
            "ingredients": list(key) if grouped_by_ingredient else [],
            "occurrences": occurrences,
            "source_files": [m.get("source_file") for m in occurrences if m.get("source_file")],
            "dose_changes": _distinct_progression(doses),
            "frequency_changes": _distinct_progression(freqs),
        })

    rolled.sort(key=lambda g: g["display_name"].lower())
    return rolled


def _distinct_progression(values: List[str]) -> List[str]:
    """Collapses consecutive repeats: ['500 mg','500 mg','1000 mg'] ->
    ['500 mg','1000 mg']. Returns [] when nothing ever changed, so callers
    can test truthiness to mean "this changed across documents"."""
    progression: List[str] = []
    for value in values:
        if not progression or progression[-1] != value:
            progression.append(value)
    return progression if len(progression) > 1 else []


def group_lab_results(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Rolls the flat lab_results_timeline up into one group per test,
    chronologically ordered, so a multi-document series reads as a series
    instead of scattered readings.

    Grouped on reference_intervals.canonical_test() — the SAME key
    lab_trends.py groups on — so that a test printed as "Fasting Glucose" on
    one report and "FBS" on the next forms one group here too. Keying on the
    printed name (as this used to) split it into two, and the trend computed
    over both readings would then attach to a group displaying only one of
    them, with the text saying "the 2 times this was tested" beside a single
    row. Tests outside that table still fall back to their lowercased name.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    display: Dict[str, str] = {}
    names_seen: Dict[str, List[str]] = {}
    for lab in timeline.get("lab_results_timeline", []):
        name = (lab.get("test_name") or "unknown test").strip()
        key = canonical_test(name) or name.lower()
        groups.setdefault(key, []).append(lab)
        display.setdefault(key, name)
        if name not in names_seen.setdefault(key, []):
            names_seen[key].append(name)

    rolled = [
        {
            "key": key,
            "test_name": display[key],
            # Every spelling this test appeared under. The planner selects
            # against the record vocabulary, which lists the raw printed
            # names, so a question about "FBS" has to keep matching a group
            # now displayed as "Fasting Glucose".
            "names": names_seen[key],
            "results": _sort_by_date(labs),
            "source_files": [l.get("source_file") for l in _sort_by_date(labs) if l.get("source_file")],
            "has_abnormal": any((l.get("flag") or "").lower() in ("high", "low") for l in labs),
        }
        for key, labs in groups.items()
    ]
    rolled.sort(key=lambda g: g["test_name"].lower())
    return rolled


def build_record_vocabulary(record: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Every entity name that appears anywhere in this patient's record. Used
    two ways: to give the planner a closed vocabulary to select from (so it
    can map "my sugar levels" onto the actual test name), and to resolve
    conversational focus deterministically without an LLM call.
    """
    timeline = _timeline_of(record)

    medications: Set[str] = set()
    for med in timeline.get("medications_timeline", []):
        if med.get("name"):
            medications.add(med["name"])
        for ingredient in med.get("ingredients") or []:
            if ingredient:
                medications.add(ingredient)

    lab_tests = {
        lab["test_name"] for lab in timeline.get("lab_results_timeline", []) if lab.get("test_name")
    }
    source_files = {
        visit.get("_source", {}).get("file")
        for visit in timeline.get("visits", [])
        if visit.get("_source", {}).get("file")
    }

    return {
        "medications": sorted(medications),
        "lab_tests": sorted(lab_tests),
        "source_files": sorted(f for f in source_files if f),
        "allergies": list(timeline.get("known_allergies") or []),
    }


# Words that appear inside record entity names but carry no identity on their
# own — matching a question against them alone would select half the record.
_GENERIC_TERM_WORDS = {
    "fasting", "random", "total", "free", "direct", "serum", "plasma", "blood",
    "urine", "level", "levels", "test", "tests", "count", "profile", "panel",
    "report", "ratio", "index", "pdf", "jpg", "jpeg", "png", "webp", "page",
    "tablet", "tablets", "capsule", "capsules", "oral", "injection",
}

# Below this length a term is matched whole-word only: "ALT" as a substring
# also hits "salt" and "alternative".
_SUBSTRING_SAFE_MIN_LEN = 5


def _significant_words(term: str) -> Set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", term.lower())
        if len(word) >= 4 and word not in _GENERIC_TERM_WORDS
    }


def match_vocabulary(text: str, vocabulary: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Deterministic entity spotting: which known medications / lab tests /
    source files does this text actually name? Matching is against a closed,
    patient-specific vocabulary, so an unrelated drug the patient isn't on
    can never be selected.

    Matches three ways, in decreasing strictness:
      * the full term appears verbatim ("metformin sr");
      * a short term appears as a whole word (so "ALT" doesn't match "salt");
      * a distinctive word of the term appears — this is what lets "my
        glucose" find the record's "Fasting Glucose". Generic qualifiers
        ("fasting", "serum", "total") are excluded from that last rule, so
        the match still has to be on something identifying.
    """
    lowered = (text or "").lower()
    if not lowered:
        return {"medications": [], "lab_tests": [], "source_files": []}

    words = set(re.findall(r"[a-z0-9]+", lowered))

    def matches(term: str) -> bool:
        lowered_term = term.lower()
        if len(lowered_term) >= _SUBSTRING_SAFE_MIN_LEN:
            if lowered_term in lowered:
                return True
        elif lowered_term in words:
            return True
        return bool(_significant_words(term) & words)

    return {
        field: [term for term in vocabulary.get(field, []) if term and matches(term)]
        for field in ("medications", "lab_tests", "source_files")
    }


# ---------------------------------------------------------------------------
# 3. Rendering the context
#
# Each renderer emits a labelled section. Every fact carries its date and
# source_file inline, because the answering model is required to cite them
# and cross-document answers are only trustworthy when provenance survives
# into the prompt.
# ---------------------------------------------------------------------------

def _render_document_manifest(timeline: Dict[str, Any]) -> str:
    """The backbone of cross-document reasoning: an explicit, numbered list
    of every document on file. Without this the model cannot tell "the
    record does not mention X" from "I was only shown part of the record"."""
    visits = timeline.get("visits", [])
    if not visits:
        return "DOCUMENTS ON FILE: none."

    lines = [f"DOCUMENTS ON FILE ({len(visits)} total, chronological):"]
    for i, visit in enumerate(_sort_by_date(visits), start=1):
        source_file = visit.get("_source", {}).get("file") or "unknown file"
        parts = [
            f"  [{i}] {source_file}",
            f"type: {visit.get('document_type') or 'unknown'}",
            f"date: {visit.get('date') or 'undated'}",
        ]
        if visit.get("provider_or_doctor"):
            parts.append(f"provider: {visit['provider_or_doctor']}")
        parts.append(f"{len(visit.get('medications') or [])} medication(s)")
        parts.append(f"{len(visit.get('lab_results') or [])} lab result(s)")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _render_allergies(timeline: Dict[str, Any]) -> str:
    allergies = timeline.get("known_allergies") or []
    if not allergies:
        return "KNOWN ALLERGIES: none recorded in any document on file."
    return "KNOWN ALLERGIES (across all documents): " + ", ".join(allergies) + "."


def _render_medication_group(group: Dict[str, Any]) -> str:
    source_count = len({s for s in group["source_files"]})
    header = f"- {group['display_name']}"
    if group["ingredients"]:
        header += f" (active ingredient(s): {', '.join(group['ingredients'])})"
    header += f" — appears in {source_count or 1} document(s):"

    lines = [header]
    for med in group["occurrences"]:
        # The normalized dose/frequency is what downstream comparison uses,
        # but the printed text is kept alongside it so a reader can audit
        # what the document literally said (and in what language).
        as_printed = f"{med.get('dosage') or '?'} {med.get('frequency') or ''}".strip()
        # The group is named after the most recent occurrence, so an earlier
        # document that printed a different brand for the same ingredient
        # would otherwise appear under a name it never used. Name it here, or
        # "which brand did that doctor prescribe?" becomes unanswerable.
        printed_name = med.get("name")
        alias = (
            f"printed as \"{printed_name}\" | "
            if printed_name and printed_name != group["display_name"]
            else ""
        )
        lines.append(
            f"    * {med.get('date') or 'undated'} | {alias}{_describe_dose(med)} | "
            f"{_describe_frequency(med)} | duration: {med.get('duration') or 'not specified'} | "
            f"as printed: \"{as_printed}\" | "
            f"source: {med.get('source_file') or 'unknown file'}"
        )
    # Stated explicitly because "did this change?" is the most common
    # cross-document follow-up, and it is an exact comparison, not a judgment.
    if group["dose_changes"]:
        lines.append(f"    ! DOSE CHANGED ACROSS DOCUMENTS: {' -> '.join(group['dose_changes'])}")
    if group["frequency_changes"]:
        lines.append(
            f"    ! FREQUENCY CHANGED ACROSS DOCUMENTS: {' -> '.join(group['frequency_changes'])}"
        )
    return "\n".join(lines)


def _render_medications(groups: Sequence[Dict[str, Any]], omitted: Sequence[str] = ()) -> str:
    if not groups and not omitted:
        return "MEDICATIONS: none recorded in any document on file."
    lines = [f"MEDICATIONS (rolled up across all documents, {len(groups)} distinct):"]
    lines.extend(_render_medication_group(g) for g in groups)
    if omitted:
        lines.append(
            "  (Also on file but not expanded for this question: "
            + ", ".join(omitted)
            + ". Say so if the question needs their details.)"
        )
    return "\n".join(lines)


def _render_lab_group(
    group: Dict[str, Any],
    trend: Optional[Dict[str, Any]],
    single: Optional[Dict[str, Any]] = None,
) -> str:
    results = group["results"]
    latest = results[-1]
    unit = latest.get("unit") or ""
    ref_range = latest.get("reference_range") or "not specified"

    lines = [
        f"- {group['test_name']} ({unit or 'no unit recorded'}, reference range: {ref_range}) — "
        f"{len(results)} result(s) across {len({s for s in group['source_files']}) or 1} document(s):"
    ]
    for lab in results:
        lines.append(
            f"    * {lab.get('date') or 'undated'} | {lab.get('value', '?')}"
            f"{(' ' + unit) if unit else ''} | flag: {lab.get('flag') or 'unknown'} | "
            f"source: {lab.get('source_file') or 'unknown file'}"
        )
    if trend:
        lines.append(f"    TREND: {trend.get('direction')} — {trend.get('flag_sequence')}.")
        if trend.get("crossed_into_abnormal_at"):
            crossing = trend["crossed_into_abnormal_at"]
            lines.append(
                f"    CROSSED INTO ABNORMAL on {crossing.get('date')} (became {crossing.get('flag')})."
            )
        if trend.get("approaching_threshold"):
            lines.append("    APPROACHING a reference-range boundary.")
        if trend.get("explanation"):
            lines.append(f"    PLAIN LANGUAGE: {trend['explanation']}")
    elif single:
        # One reading, so no trend — but lab_trends still worked out whether
        # the value sits low, normal or high. Without this a question like
        # "is my hemoglobin ok?" would reach the model as a bare number.
        basis = {
            "report": "against the range printed on the report",
            "general": (
                "against a general range for this patient's age and sex, NOT the "
                "range their own laboratory uses"
            ),
        }.get(single.get("range_source"), "no range was available to compare against")
        lines.append(
            f"    SINGLE RESULT (no earlier reading to compare with) — status: "
            f"{single.get('status')}, assessed {basis}."
        )
        if single.get("explanation"):
            lines.append(f"    PLAIN LANGUAGE: {single['explanation']}")
    return "\n".join(lines)


def _render_labs(
    groups: Sequence[Dict[str, Any]],
    lab_trends: Dict[str, Any],
    omitted: Sequence[str] = (),
) -> str:
    if not groups and not omitted:
        return "LAB RESULTS: none recorded in any document on file."

    # Keyed the same way group_lab_results() keys its groups -- canonical id
    # where there is one, lowercased printed name otherwise -- so a trend
    # computed over "Fasting Glucose" + "FBS" lands on the group holding both.
    def _trend_key(entry):
        return entry.get("test_id") or (entry.get("test_name") or "").lower()

    trends_by_test = {_trend_key(t): t for t in (lab_trends.get("trends") or [])}
    singles_by_test = {_trend_key(s): s for s in (lab_trends.get("single_results") or [])}
    lines = [f"LAB RESULTS (grouped by test across all documents, {len(groups)} distinct):"]
    lines.extend(
        _render_lab_group(g, trends_by_test.get(g["key"]), singles_by_test.get(g["key"]))
        for g in groups
    )

    insufficient = lab_trends.get("insufficient_data") or []
    if insufficient:
        lines.append(
            "  No trend could be computed for: "
            + ", ".join(
                f"{i.get('test_name')} ({i.get('reason')})" for i in insufficient
            )
        )
    if omitted:
        lines.append(
            "  (Also on file but not expanded for this question: " + ", ".join(omitted) + ".)"
        )
    return "\n".join(lines)


def _render_clinical_notes(
    timeline: Dict[str, Any],
    only_files: Optional[Set[str]] = None,
    max_note_chars: int = MAX_CLINICAL_NOTE_CHARS,
) -> str:
    notes = []
    for visit in _sort_by_date(timeline.get("visits", [])):
        text = visit.get("clinical_notes")
        if not text:
            continue
        source_file = visit.get("_source", {}).get("file") or "unknown file"
        if only_files is not None and source_file not in only_files:
            continue
        if len(text) > max_note_chars:
            text = text[:max_note_chars].rstrip() + " …[note truncated]"
        notes.append(
            f"- {visit.get('date') or 'undated'} | {visit.get('document_type') or 'unknown type'} | "
            f"source: {source_file}\n    {text}"
        )
    if not notes:
        return "CLINICAL NOTES: none recorded."
    return "CLINICAL NOTES (chronological):\n" + "\n".join(notes)


def _render_safety_flags(cross_check: Dict[str, Any]) -> str:
    """Surfaces the already-computed cross-check report into Q&A context, so
    a question about one drug can see that the SAME drug is already flagged
    against another one elsewhere in the record."""
    if not cross_check:
        return "SAFETY CROSS-CHECK: not yet computed for this patient."

    lines = ["SAFETY CROSS-CHECK (already computed over the full record):"]
    empty = True

    def _evidence(item: Dict[str, Any]) -> str:
        """Marks whether a finding is verified or is unconfirmed model recall,
        so the answer can carry that distinction instead of presenting both
        with equal authority."""
        source = item.get("evidence_source")
        if source == "deterministic":
            return " [VERIFIED from the patient's own records]"
        if source == "reference_graph":
            return " [BACKED BY a reference document in the knowledge graph]"
        if source == "model_knowledge":
            return (
                " [UNVERIFIED — general medical knowledge, not confirmed by any "
                "drug-interaction database in this system; say so if you rely on it]"
            )
        return ""

    def _when(item: Dict[str, Any]) -> str:
        """States whether the drugs were actually taken together, and when. A
        pair whose courses finished years apart is history, not a live risk,
        and an answer that omits that reads as a current warning."""
        timing = item.get("timing") or {}
        status = timing.get("status")
        if status == "concurrent":
            return (f" [TAKEN TOGETHER {timing.get('window_start')} to "
                    f"{timing.get('window_end')}, {timing.get('overlap_days')} days]")
        if status == "possible":
            return (f" [MAY have overlapped around {timing.get('window_start')} to "
                    f"{timing.get('window_end')} — a course has no stated duration]")
        if status == "not_concurrent":
            return (f" [NEVER TAKEN TOGETHER — courses ended about "
                    f"{timing.get('gap_days')} days apart; historical, not a current risk]")
        return ""

    for item in cross_check.get("potential_drug_interactions") or []:
        empty = False
        lines.append(
            f"- INTERACTION ({item.get('severity', 'unknown')} severity, confidence "
            f"{item.get('confidence')}){_evidence(item)}{_when(item)}: "
            f"{', '.join(item.get('medications_involved') or [])} — "
            f"{item.get('explanation')}"
        )

    for item in cross_check.get("concurrent_exposure") or []:
        empty = False
        lines.append(
            f"- DOUBLE DOSE (confidence 0.9) [VERIFIED from the patient's own records]: "
            f"{item.get('ingredient')} supplied by two prescriptions at once, "
            f"{item.get('window_start')} to {item.get('window_end')}"
            + (f", totalling {item['cumulative_daily_dose']} {item['dosage_unit']}/day"
               if item.get("cumulative_daily_dose") and item.get("dosage_unit") else "")
            + f" — {item.get('note')}"
        )
    for item in cross_check.get("duplicate_prescriptions") or []:
        empty = False
        occurrences = ", ".join(
            f"{o.get('date') or 'undated'} ({o.get('source_file') or 'unknown file'})"
            for o in item.get("occurrences") or []
        )
        lines.append(
            f"- DUPLICATE (confidence {item.get('confidence')}): {item.get('medication')} in "
            f"{occurrences} — {item.get('explanation')}"
        )
    for item in cross_check.get("conflicting_dosage_instructions") or []:
        empty = False
        conflicts = ", ".join(
            f"{c.get('date') or 'undated'}: {c.get('dosage')} {c.get('frequency')} "
            f"({c.get('source_file') or 'unknown file'})"
            for c in item.get("conflicting_instructions") or []
        )
        lines.append(
            f"- DOSAGE CONFLICT (confidence {item.get('confidence')}): {item.get('medication')} — "
            f"{conflicts} — {item.get('explanation')}"
        )
    for item in cross_check.get("allergy_conflicts") or []:
        empty = False
        lines.append(
            f"- ALLERGY CONFLICT (confidence {item.get('confidence')}){_evidence(item)}: "
            f"{item.get('medication')} vs "
            f"allergy '{item.get('allergy')}' — {item.get('explanation')}"
        )

    if empty:
        lines.append("- No interactions, duplicates, dosage conflicts or allergy conflicts were flagged.")
    if cross_check.get("overall_recommendation"):
        lines.append(f"- Overall: {cross_check['overall_recommendation']}")
    return "\n".join(lines)


def _render_reference_guidance(timeline: Dict[str, Any]) -> str:
    """Injects published clinical guidance that applies to this patient's
    medication list (see reference_library.py), with source and page.

    Kept separate from the patient's own findings on purpose: this is
    established guidance the answer may state as fact and cite, whereas a
    cross-check finding is a claim ABOUT this person. Returns "" when nothing
    applies, so a record with no opioid never carries opioid guidance."""
    from reference_library import render_reference_guidance

    return render_reference_guidance(timeline)


def _render_consult_routing(triage: Dict[str, Any]) -> str:
    """Surfaces the already-computed consultation routing (see
    consult_triage.py) so an answer about "should I see someone?" reflects
    the same pharmacist/doctor decision the rest of the pipeline reached,
    rather than the answering model improvising its own."""
    if not triage:
        return "CONSULTATION ROUTING: not yet computed for this patient."

    # Documents that scanned badly. Reported in both branches below, because
    # "was my prescription read correctly?" is a fair question whether or not
    # anything clinical was found — but always as a note about the PAPERWORK,
    # so the answering model does not turn a blurry scan into a referral. See
    # DATA_QUALITY_TRIGGERS in consult_triage.py.
    quality_lines = []
    for notice in triage.get("document_quality_notices") or []:
        quality_lines.append(
            f"- DOCUMENT QUALITY (not a clinical finding, not a reason to consult "
            f"anyone): {notice.get('subject')} — {notice.get('detail')}"
        )

    if not triage.get("consult_needed"):
        base = (
            "CONSULTATION ROUTING: no automated trigger for a consultation was found. "
            "State plainly that this is NOT a clean bill of health — it means only that "
            "these specific checks found nothing in the uploaded documents, and the "
            "patient should still raise any symptom or concern with their doctor or "
            "pharmacist."
        )
        if quality_lines:
            base += (
                "\nSome uploaded documents could not be read with full confidence. "
                "Mention this only if asked about the documents themselves, and say "
                "it means the details are worth checking against the original "
                "paperwork — not that the patient needs an appointment.\n"
                + "\n".join(quality_lines)
            )
        return base

    lines = [
        f"CONSULTATION ROUTING (already computed): consult a "
        f"{triage.get('consult_type')} — {triage.get('urgency_meaning') or triage.get('urgency')} "
        f"(confidence {triage.get('confidence')}).",
    ]

    for specialty in triage.get("recommended_specialties") or []:
        lines.append(
            f"- SPECIALTY: {specialty.get('specialty')} (confidence "
            f"{specialty.get('confidence')}) for {', '.join(specialty.get('triggered_by') or [])}"
            f" — {specialty.get('reason')}"
        )

    for item in triage.get("referral_items") or []:
        caveat = " [LOW CONFIDENCE — verify against the original document]" if item.get(
            "confidence_caveat"
        ) else ""
        lines.append(
            f"- {item.get('trigger')} -> {item.get('route')} ({item.get('urgency')}, "
            f"confidence {item.get('confidence')}): {item.get('subject')}{caveat}"
        )

    lines.extend(quality_lines)
    lines.append(f"- Why this routing: {triage.get('summary')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Context assembly
# ---------------------------------------------------------------------------

def _join_sections(sections: Sequence[str]) -> str:
    return "\n\n".join(s for s in sections if s)


# A section trimmed below this is worse than useless — a few dangling entries
# read as if they were the whole list. Replace it with a marker instead.
MIN_USEFUL_SECTION_CHARS = 240


def _fit_to_budget(
    sections: List[Dict[str, Any]], budget_chars: int
) -> Tuple[str, bool]:
    """
    Trims an assembled context to fit `budget_chars`, giving up the least
    valuable sections first.

    Sections marked mandatory are never trimmed: the document manifest, the
    allergy list, the safety cross-check, and the consultation routing.
    Cutting those is what turns a space problem into a safety problem — a
    truncated manifest makes "that isn't in your records" a lie, a truncated
    allergy list makes the consult recommendation fire on incomplete grounds,
    and a truncated routing block drops who the patient was told to see and
    how urgently. Elastic sections are
    trimmed from the bottom up (clinical notes, then labs, then medications),
    and anything reduced past usefulness is replaced by an explicit marker so
    the model states what it wasn't shown rather than implying absence.

    Returns (context, was_trimmed).
    """
    texts = [s["text"] for s in sections]
    overflow = len(_join_sections(texts)) - budget_chars
    if overflow <= 0:
        return _join_sections(texts), False

    for i in reversed(range(len(sections))):
        if overflow <= 0:
            break
        if sections[i]["mandatory"]:
            continue

        original = texts[i]
        keep = len(original) - overflow
        if keep < MIN_USEFUL_SECTION_CHARS:
            replacement = (
                f"{sections[i]['label']}: omitted for space. Ask about a specific "
                "medication, test or document to see these details."
            )
        else:
            replacement = original[:keep].rstrip() + "\n  […section truncated for space…]"
        overflow -= len(original) - len(replacement)
        texts[i] = replacement

    context = _join_sections(texts)
    if len(context) > budget_chars:
        # Mandatory sections alone exceed the budget. Cutting the tail is the
        # only option left; the marker keeps the model honest about it.
        context = (
            context[:budget_chars].rstrip()
            + "\n\n[CONTEXT TRUNCATED — some records were not shown.]"
        )
    return context, True


def build_full_context(record: Dict[str, Any]) -> str:
    """The whole record, nothing dropped. Preferred whenever it fits."""
    timeline = _timeline_of(record)
    return _join_sections([
        _render_document_manifest(timeline),
        _render_allergies(timeline),
        _render_medications(group_medications(timeline)),
        _render_labs(group_lab_results(timeline), record.get("lab_trends") or {}),
        _render_clinical_notes(timeline),
        _render_safety_flags(record.get("cross_check_report") or {}),
        _render_consult_routing(record.get("consult_triage") or {}),
        _render_reference_guidance(timeline),
    ])


RETRIEVAL_PLAN_PROMPT = """
You select which parts of a patient's medical record are needed to answer a
question. You do NOT answer the question.

You are given the question and the closed vocabulary of entities that
actually exist in this patient's record. Select only from that vocabulary —
never invent a medication or test the patient does not have.

Rules:
- Include an entity if the question refers to it directly, by synonym or
  category ("my sugar levels" -> the glucose test that exists in the
  vocabulary; "my blood pressure meds" -> the antihypertensives present).
- When the question is broad ("what am I taking?", "summarize my record",
  "is anything wrong?", "what changed?"), set wants_full_record to true
  instead of guessing a subset. Under-selecting is the dangerous error.
- Set touches_risk to true if the question is about safety, danger,
  interactions, allergies, side effects, or changing a dose.
"""

RETRIEVAL_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "medications": {"type": "array", "items": {"type": "string"}},
        "lab_tests": {"type": "array", "items": {"type": "string"}},
        "source_files": {"type": "array", "items": {"type": "string"}},
        "wants_full_record": {"type": "boolean"},
        "touches_risk": {"type": "boolean"},
    },
    "required": ["medications", "lab_tests", "source_files", "wants_full_record", "touches_risk"],
    "additionalProperties": False,
}

RETRIEVAL_PLAN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "retrieval_plan", "strict": True, "schema": RETRIEVAL_PLAN_SCHEMA},
}

# Keywords that make a question safety-relevant. Used for the deterministic
# planner fallback AND for the post-hoc consult guard, so the safety
# recommendation never depends on a model call succeeding.
RISK_PATTERN = re.compile(
    r"\b(safe|safety|danger|dangerous|risk|risky|interact\w*|allerg\w*|overdose|"
    r"side.?effect\w*|adverse|contraindicat\w*|harmful|toxic|stop taking|"
    r"increase|decrease|double|halve|adjust|change (?:my|the) dose|together|combine|mix)\b",
    re.IGNORECASE,
)

BROAD_QUESTION_PATTERN = re.compile(
    r"\b(everything|all|summar\w*|overview|what.{0,12}(?:taking|on|prescribed)|"
    r"any(?:thing)?\s+(?:wrong|concerning|issue|problem)|changed?|history|timeline)\b",
    re.IGNORECASE,
)


def _fallback_plan(question: str, vocabulary: Dict[str, List[str]]) -> Dict[str, Any]:
    """Deterministic plan used when the planner call fails, or is skipped.
    Errs toward including more: a missed entity is a wrong answer, extra
    context is only cost."""
    matched = match_vocabulary(question, vocabulary)
    nothing_matched = not any(matched.values())
    return {
        "medications": matched["medications"],
        "lab_tests": matched["lab_tests"],
        "source_files": matched["source_files"],
        "wants_full_record": nothing_matched or bool(BROAD_QUESTION_PATTERN.search(question or "")),
        "touches_risk": bool(RISK_PATTERN.search(question or "")),
    }


def build_retrieval_plan(
    question: str,
    vocabulary: Dict[str, List[str]],
    focus: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Decides which entities a question needs, when the record is too large to
    include whole. `focus` carries entities from earlier conversation turns
    so a follow-up like "and was that safe?" keeps its subject.

    Never raises: falls back to deterministic keyword matching against the
    patient's own vocabulary if the planner call fails.
    """
    payload = {
        "question": question,
        "available_medications": vocabulary.get("medications", []),
        "available_lab_tests": vocabulary.get("lab_tests", []),
        "available_documents": vocabulary.get("source_files", []),
        "entities_already_being_discussed": focus or {},
    }
    try:
        response = client.chat.completions.create(
            model=PLANNER_MODEL,
            messages=[
                {"role": "system", "content": RETRIEVAL_PLAN_PROMPT},
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
            response_format=RETRIEVAL_PLAN_RESPONSE_FORMAT,
        )
        plan = json.loads(response.choices[0].message.content)
    except (OpenAIError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"  Retrieval planning failed, falling back to keyword selection: {e}")
        plan = _fallback_plan(question, vocabulary)

    # Conversational focus is merged in regardless of what the planner said —
    # the subject of a follow-up is established fact, not something to re-infer.
    for field in ("medications", "lab_tests", "source_files"):
        merged = list(plan.get(field) or []) + list((focus or {}).get(field) or [])
        seen: Set[str] = set()
        plan[field] = [x for x in merged if not (x.lower() in seen or seen.add(x.lower()))]

    # The risk flag is a safety control, so code gets the final say on
    # setting it — the model may only ever add to it, never clear it.
    plan["touches_risk"] = bool(plan.get("touches_risk")) or bool(RISK_PATTERN.search(question or ""))
    return plan


def _selects(group_terms: Sequence[str], selected: Sequence[str]) -> bool:
    """True if any selected term names this group (either direction of
    substring match, so 'Metformin' selects 'Metformin 500' and vice versa)."""
    lowered_selected = [s.lower() for s in selected if s]
    return any(
        term and any(term.lower() in sel or sel in term.lower() for sel in lowered_selected)
        for term in group_terms
    )


def build_planned_context(
    record: Dict[str, Any],
    plan: Dict[str, Any],
    budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
) -> Tuple[str, bool]:
    """
    Narrowed assembly for records that exceed the budget. Selection happens
    twice over: first by relevance (which entities the plan asked for), then
    by budget (see _fit_to_budget), which never trims the manifest, the
    allergies, or the safety cross-check.

    Everything dropped at either stage is named explicitly in the context, so
    the model can say what it wasn't shown rather than implying the record is
    silent on it.

    Returns (context, was_trimmed).
    """
    timeline = _timeline_of(record)

    med_groups = group_medications(timeline)
    lab_groups = group_lab_results(timeline)

    if plan.get("wants_full_record"):
        kept_meds, kept_labs = med_groups, lab_groups
    else:
        selected_meds = plan.get("medications") or []
        selected_labs = plan.get("lab_tests") or []
        kept_meds = [
            g for g in med_groups
            if _selects([g["display_name"], *g["ingredients"]], selected_meds)
        ]
        kept_labs = [g for g in lab_groups if _selects(g["names"], selected_labs)]
        # A risk question about specific drugs still needs the whole
        # medication list — you cannot check an interaction against drugs
        # you were not shown.
        if plan.get("touches_risk"):
            kept_meds = med_groups
        # Abnormal results are always worth showing; a "normal" question can
        # still have an abnormal answer.
        for group in lab_groups:
            if group["has_abnormal"] and group not in kept_labs:
                kept_labs.append(group)
        kept_labs.sort(key=lambda g: g["test_name"].lower())

    omitted_meds = [g["display_name"] for g in med_groups if g not in kept_meds]
    omitted_labs = [g["test_name"] for g in lab_groups if g not in kept_labs]

    only_files = set(plan.get("source_files") or []) or None
    if plan.get("wants_full_record"):
        only_files = None

    # Ordered as the model reads them; trimming walks this list backwards, so
    # elastic sections are listed cheapest-to-lose last.
    return _fit_to_budget(
        [
            {"label": "DOCUMENTS ON FILE", "mandatory": True,
             "text": _render_document_manifest(timeline)},
            {"label": "KNOWN ALLERGIES", "mandatory": True,
             "text": _render_allergies(timeline)},
            {"label": "MEDICATIONS", "mandatory": False,
             "text": _render_medications(kept_meds, omitted_meds)},
            {"label": "LAB RESULTS", "mandatory": False,
             "text": _render_labs(kept_labs, record.get("lab_trends") or {}, omitted_labs)},
            {"label": "CLINICAL NOTES", "mandatory": False,
             "text": _render_clinical_notes(timeline, only_files=only_files)},
            {"label": "SAFETY CROSS-CHECK", "mandatory": True,
             "text": _render_safety_flags(record.get("cross_check_report") or {})},
            {"label": "CONSULTATION ROUTING", "mandatory": True,
             "text": _render_consult_routing(record.get("consult_triage") or {})},
            # Mandatory: it only appears when a genuinely high-risk combination
            # is on file, and trimming the one cited safety source to save
            # space would leave the answer relying on uncited recall instead.
            {"label": "PUBLISHED GUIDANCE", "mandatory": True,
             "text": _render_reference_guidance(timeline)},
        ],
        budget_chars,
    )


def assemble_context(
    record: Dict[str, Any],
    question: str,
    focus: Optional[Dict[str, List[str]]] = None,
    budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
) -> Tuple[str, Dict[str, Any]]:
    """
    Builds the context string for one question, plus a small diagnostics dict
    describing how it was built (useful for debugging and demo transparency).

    Fast path — and the normal one: if the entire record fits the budget, it
    is used whole and NO planning call is made. Narrowing only happens when
    a record is genuinely too large, which is where an approximate selection
    is finally worth its risk.
    """
    full_context = build_full_context(record)
    if len(full_context) <= budget_chars:
        return full_context, {
            "strategy": "full_record",
            "context_chars": len(full_context),
            "plan": None,
        }

    plan = build_retrieval_plan(question, build_record_vocabulary(record), focus)
    context, truncated = build_planned_context(record, plan, budget_chars)

    return context, {
        "strategy": "planned",
        "context_chars": len(context),
        "plan": plan,
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# 5. Answering
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = """
You are a patient-facing medical records assistant. You answer questions
using ONLY the patient record context provided below — assembled from that
patient's own extracted documents (medications, lab results, clinical notes,
allergies, and an already-computed safety cross-check).

WHAT THE CONTEXT IS:
- "DOCUMENTS ON FILE" lists every document in this patient's record. If a
  section says something is "none recorded", that means it is absent from
  the whole record, and you may say so.
- Medications and lab results are rolled up ACROSS documents: every
  occurrence of a drug or test is listed together with its date and source
  file. Lines marked "! DOSE CHANGED ACROSS DOCUMENTS" or "! FREQUENCY
  CHANGED" were computed exactly from the record — treat them as fact and
  use them when asked what changed.
- If a section names entries that were "not expanded for this question", or
  the context is marked truncated, you were shown a subset. Say so plainly
  rather than implying the record is silent on them.

RULES:
- Answer strictly from the context. If it does not cover the question, say
  "I don't have enough information" rather than guessing or using outside
  medical knowledge.
- NEVER provide a diagnosis or interpret what a result "means" clinically.
- Write for someone with NO medical background. Spell out any abbreviation
  printed on a document the first time you use it, keeping the printed form
  in brackets so they can find it on their report (e.g. "a kidney test
  (Creatinine)"). Say "the normal range" rather than "reference range", and
  "higher than the normal range" rather than "elevated" or "abnormal". The
  "PLAIN LANGUAGE" line under each lab result is already written this way —
  match its register, and prefer it over the raw numbers when explaining a
  trend. Naming what a test looks at is allowed; saying what a result implies
  about someone's health is the diagnosis rule above and is not.
- Prefer answers that span documents when the question calls for it: compare
  dates, name which document each fact came from, and state the direction of
  any change over time.
- Whenever the question touches risk, drug interactions, allergy conflicts,
  or changing/adjusting a dosage, explicitly recommend consulting a doctor or
  pharmacist and set recommend_professional_consult to true.
- A "PUBLISHED GUIDANCE" section, when present, is established clinical
  guidance from a named source — not a claim about this patient. You may state
  it as fact, and should name the source and page when you rely on it ("SAMHSA's
  overdose toolkit, page 13"). Never present it as something found in their
  records, and never extend it beyond what the quote says.
- Safety findings carry TIMING. A finding marked NEVER TAKEN TOGETHER must be
  described as history, not as a current risk — say the two courses did not
  overlap and give the dates. For one marked TAKEN TOGETHER, state the period
  it was live ("between 9 and 23 November 2025"). Never warn about a drug pair
  as if it were current when the record shows the courses were years apart.
- Safety findings are marked VERIFIED, BACKED BY a reference document, or
  UNVERIFIED. Never present an UNVERIFIED finding as established fact: say
  plainly that it comes from general medical knowledge and has not been
  checked against a drug-interaction database, and that a pharmacist can
  confirm it. Do not "upgrade" it by sounding certain, and do not dismiss it
  either — unverified means unconfirmed, not wrong.
- When asked WHO to see, how urgently, or which kind of doctor, answer from
  the "CONSULTATION ROUTING" section — it was computed over the whole record
  and is the pipeline's own routing. Do not substitute your own judgment of
  who to see or invent a specialty it does not name. If that section says no
  trigger was found, say plainly that this is not a clean bill of health and
  that any symptom or concern still warrants contacting a professional.
- Cite the date and source_file of every record you rely on in "sources".
- Set cross_document to true if your answer combined facts from more than one
  source document.
- Respond with STRICT JSON only, matching the required schema.

CONFIDENCE SCORING — "confidence" reflects how directly the record answers
the question, not how fluent your answer sounds:
- 0.90-1.00: the record states the answer directly and completely.
- 0.60-0.89: the record is relevant but partial, or you combined several
  entries to form the answer, or some entries were low-confidence extractions.
- Below 0.60: the record is only tangentially related, you were shown a
  narrowed subset that may exclude the answer, or you are largely saying
  "I don't have enough information".
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
        "cross_document": {"type": "boolean"},
        "recommend_professional_consult": {"type": "boolean"},
    },
    "required": ["answer", "confidence", "sources", "cross_document", "recommend_professional_consult"],
    "additionalProperties": False,
}

ANSWER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "patient_qa_answer", "strict": True, "schema": ANSWER_JSON_SCHEMA},
}

_NO_INFO_ANSWER = {
    "answer": (
        "I don't have enough information — no processed medical records were found for "
        "this patient yet. Upload a document first."
    ),
    "confidence": 0.0,
    "sources": [],
    "cross_document": False,
    "recommend_professional_consult": False,
}


def _enrich_sources(sources: List[Dict[str, Any]], timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Attaches document_type and the archived document_url to each cited
    source by looking it up in the timeline. Done in code rather than asked
    of the model — a URL is exactly the kind of field a model will happily
    invent."""
    by_file: Dict[str, Dict[str, Any]] = {}
    for visit in timeline.get("visits", []):
        source_file = visit.get("_source", {}).get("file")
        if source_file:
            by_file[source_file] = visit

    enriched = []
    for source in sources or []:
        visit = by_file.get(source.get("source_file") or "")
        entry = dict(source)
        if visit:
            entry["document_type"] = visit.get("document_type")
            if visit.get("document_url"):
                entry["document_url"] = visit["document_url"]
        enriched.append(entry)
    return enriched


def _apply_safety_guard(result: Dict[str, Any], question: str, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic backstop for the product's stated safety promise:
    recommend a professional for high-risk OR low-confidence answers. The
    model is already told to do this, but "already told to" is not a control
    — this makes it one, and records WHY it fired.
    """
    reasons: List[str] = []

    if RISK_PATTERN.search(question or ""):
        reasons.append("the question involves safety, interactions, allergies, or a dosage change")
    if (diagnostics.get("plan") or {}).get("touches_risk"):
        reasons.append("the records retrieved for this question are safety-relevant")

    confidence = result.get("confidence")
    low_confidence = isinstance(confidence, (int, float)) and confidence <= LOW_CONFIDENCE_THRESHOLD
    if low_confidence:
        reasons.append(
            f"the answer's confidence is low ({confidence:.2f} at or below "
            f"{LOW_CONFIDENCE_THRESHOLD:.2f})"
        )
    if diagnostics.get("truncated"):
        reasons.append("only part of the record could be shown for this question")

    result["low_confidence"] = bool(low_confidence)
    if reasons:
        result["recommend_professional_consult"] = True
        result["consult_reason"] = (
            "Please confirm this with a doctor or pharmacist, because "
            + "; ".join(dict.fromkeys(reasons))
            + "."
        )
    elif result.get("recommend_professional_consult"):
        result["consult_reason"] = (
            "Please confirm this with a doctor or pharmacist before acting on it."
        )
    return result


def answer_question(
    patient_key: str,
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 8,
    retrieval_query: Optional[str] = None,
    record: Optional[Dict[str, Any]] = None,
    focus: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Answers a natural-language question about one patient, grounded only in
    that patient's own processed records.

    1. Loads the patient's saved snapshot (Mongo, else the CLI's local
       report) unless a `record` is passed in.
    2. Assembles context — the full record when it fits, otherwise a planned
       subset (see assemble_context).
    3. Calls the chat model with a system prompt that forbids diagnosis,
       requires citing dates + source files, and forces structured JSON.
    4. Enriches cited sources with document_type/document_url from the
       timeline, and applies the deterministic consult safety guard.

    retrieval_query: used instead of `question` for entity selection when the
        record needs narrowing. Lets conversation.py resolve a follow-up like
        "was that safe?" into a fully-specified query, while `question`
        remains the literal text the answering model responds to.
    record: pre-loaded snapshot, to avoid a second database read when the
        caller already has one.
    focus: entities under discussion in this conversation, carried across
        turns so a follow-up keeps its subject.
    top_k: accepted for API compatibility and ignored. Retrieval is no longer
        top-k nearest-neighbour; it is complete-record assembly.

    Returns:
        {"answer": str, "confidence": float,
         "sources": [{"date", "source_file", "document_type"?, "document_url"?}],
         "cross_document": bool, "recommend_professional_consult": bool,
         "low_confidence": bool, "consult_reason": str?,
         "retrieval": {"strategy", "context_chars", "plan"}}

    Raises ValueError for a missing patient_key/question, RuntimeError if the
    chat call fails. Returns a graceful "no information" answer (no API call)
    when the patient has no processed records.
    """
    if not patient_key or not patient_key.strip():
        raise ValueError("patient_key is required and cannot be empty.")
    if not question or not question.strip():
        raise ValueError("question is required and cannot be empty.")

    if record is None:
        record = load_patient_record(patient_key)
    if not _has_content(record):
        return dict(_NO_INFO_ANSWER)

    selection_query = (
        retrieval_query if retrieval_query and retrieval_query.strip() else question
    )
    context, diagnostics = assemble_context(record, selection_query, focus)

    messages: List[Dict[str, str]] = [{"role": "system", "content": QA_SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({
        "role": "user",
        "content": f"PATIENT RECORD CONTEXT:\n\n{context}\n\nQuestion: {question}",
    })

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            response_format=ANSWER_RESPONSE_FORMAT,
        )
    except OpenAIError as e:
        raise RuntimeError(f"Chat completion failed while answering question: {e}") from e

    result = json.loads(response.choices[0].message.content)
    result["sources"] = _enrich_sources(result.get("sources") or [], _timeline_of(record))
    result = _apply_safety_guard(result, question, diagnostics)
    result["retrieval"] = diagnostics
    return result
