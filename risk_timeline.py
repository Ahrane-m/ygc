"""
Risk Timeline — when was each risk actually live?
=========================================
Gives every safety finding a TIME WINDOW, and separates risks the patient was
genuinely exposed to from drug pairs that merely appear in the same record.

WHY THIS MATTERS (measured on a real record)
--------------------------------------------
Two drugs only interact if the patient was taking them at the same time. The
cross-check compares a flat medication list with no notion of when each course
started or ended, so it reports any pair that co-occurs in the record. On a
real six-document record, that produced:

    Fluconazole + Montelukast        308 days apart   — never concurrent
    Cetirizine  + Chlorpheniramine   861 days apart   — never concurrent
    Fluconazole + Omeprazole         199 days apart   — never concurrent
    Paracetamol + Diclofenac         same 14 days     — genuinely concurrent

Three of four flagged interactions were between courses that had finished
months or years before the other began. Presented without dates they read as
current risks, and they drove a pharmacist referral. This is the same class of
error as counting one prescription twice because it was uploaded twice: an
alarm manufactured from the shape of the record rather than from the patient's
actual exposure — and false alarms are what teach someone to ignore the real
one.

WHAT IT DOES NOT DO
-------------------
It never deletes a finding. A pair that was concurrent in the past is real
history and stays, marked `not_concurrent` with the dates that show why. What
changes is that it stops being presented as something happening now.

Nor does it invent a maximum safe dose. `cumulative_daily_dose` is arithmetic
over concurrent prescriptions of one ingredient — a number the record
supports. Saying that number is "too much" would need reference data this
system does not have, and would be exactly the ungrounded claim
evidence_grading.py exists to cap.

DATES
-----
Prescription dates arrive in mixed formats, and "09/11/2025" is ambiguous. The
convention is inferred from the record itself rather than assumed: if any date
in it has a first component above 12 ("14/10/2023", "26/02/2026"), the record
is day-first and every date is read that way. Only when nothing disambiguates
does it fall back to the parser default — and a date that cannot be read at
all makes a window unknown, never a guess.
"""

import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dateutil import parser as dateutil_parser

# Overlap verdicts.
CONCURRENT = "concurrent"          # windows provably overlap
POSSIBLE = "possible"              # a duration is unknown — cannot rule out
NOT_CONCURRENT = "not_concurrent"  # windows provably do not overlap
UNKNOWN = "unknown"                # dates unusable on one side

_DURATION_UNITS = (
    (r"day", 1),
    (r"week", 7),
    (r"month", 30),
    (r"year", 365),
)

# "As required" / PRN has no fixed end. Treated as unknown length rather than
# as zero — a PRN course that overlaps something is still an exposure.
_OPEN_ENDED = re.compile(r"as\s+required|as\s+needed|\bprn\b|when\s+required", re.I)

# A course with no stated duration still occupied some time. This is only used
# to say "possible", never "concurrent" — it widens what gets checked, and
# never turns an unknown into a confirmed overlap.
ASSUMED_COURSE_DAYS = 30


def parse_duration_days(text: Any) -> Optional[int]:
    """Duration in days from printed text, or None if open-ended/unstated.

    "14 days" -> 14, "4 weeks" -> 28, "1 month" -> 30, "As required" -> None.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    if _OPEN_ENDED.search(text):
        return None
    for unit, multiplier in _DURATION_UNITS:
        match = re.search(rf"(\d+(?:\.\d+)?)\s*{unit}s?\b", text, re.I)
        if match:
            return max(1, int(round(float(match.group(1)) * multiplier)))
    return None


def infer_dayfirst(date_strings: Sequence[Any]) -> bool:
    """
    Whether this record writes dates day-first, inferred from the record.

    A single unambiguous date settles it for all the ambiguous ones:
    "14/10/2023" can only be day-first, so "09/11/2025" in the same record is
    9 November, not 11 September. Guessing this wrong shifts a treatment
    window by months and silently changes which risks look concurrent.
    """
    for raw in date_strings:
        if not isinstance(raw, str):
            continue
        match = re.match(r"\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*\d{2,4}", raw)
        if not match:
            continue
        first, second = int(match.group(1)), int(match.group(2))
        if first > 12 and second <= 12:
            return True
        if second > 12 and first <= 12:
            return False
    return True  # day-first is the majority convention outside the US


# YYYY-MM-DD is unambiguous, but dateutil still applies `dayfirst` to it and
# reads "2025-11-09" as 11 September. One real record contained both
# "09/11/2025" and "2025-11-09" for the same prescription, so a record-wide
# dayfirst would have shifted that window by two months and changed which
# risks looked concurrent. ISO is therefore matched first and parsed straight.
_ISO_DATE_RE = re.compile(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})")


def parse_date(raw: Any, dayfirst: bool = True) -> Optional[date]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    iso = _ISO_DATE_RE.match(raw)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return None
    try:
        return dateutil_parser.parse(raw.strip(), fuzzy=True, dayfirst=dayfirst).date()
    except (ValueError, OverflowError, TypeError):
        return None


def _daily_dose(med: Dict[str, Any]) -> Optional[float]:
    """Normalized dose x doses per day, when both are known."""
    value = med.get("dosage_value")
    per_day = med.get("frequency_per_day")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not isinstance(per_day, (int, float)) or isinstance(per_day, bool):
        return None
    return round(float(value) * float(per_day), 3)


def _ingredient_keys(med: Dict[str, Any]) -> List[str]:
    """Lowercased ingredient names, salt suffixes stripped, for matching a
    finding's drug names back to timeline entries."""
    from document_dedup import _base_ingredient

    keys = [
        _base_ingredient(i) for i in (med.get("ingredients") or []) if i and i.strip()
    ]
    name = med.get("name")
    if name:
        keys.append(_base_ingredient(name))
    return [k for k in dict.fromkeys(keys) if k]


def build_treatment_windows(
    timeline: Dict[str, Any], dayfirst: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """
    One window per medication entry:
      {ingredients, name, start, end, duration_days, duration_known,
       daily_dose, dosage_unit, date, source_file, prescription_group}

    `end` is None when the duration is unknown — the course has a start but no
    provable finish, which is what makes an overlap "possible" rather than
    "concurrent".
    """
    entries = timeline.get("medications_timeline") or []
    if dayfirst is None:
        dayfirst = infer_dayfirst([e.get("date") for e in entries])

    windows: List[Dict[str, Any]] = []
    for med in entries:
        start = parse_date(med.get("date"), dayfirst=dayfirst)
        days = parse_duration_days(med.get("duration"))
        windows.append({
            "ingredients": _ingredient_keys(med),
            "name": med.get("name"),
            "date": med.get("date"),
            "start": start,
            "end": start + timedelta(days=days) if (start and days) else None,
            "duration_days": days,
            "duration_known": days is not None,
            "daily_dose": _daily_dose(med),
            "dosage_unit": med.get("dosage_unit"),
            "source_file": med.get("source_file"),
            "prescription_group": med.get("prescription_group"),
        })
    return windows


def _effective_end(window: Dict[str, Any]) -> Optional[date]:
    """End date, or a provisional one for an unknown-length course. Used only
    to decide whether an overlap is POSSIBLE."""
    if window["end"]:
        return window["end"]
    if window["start"]:
        return window["start"] + timedelta(days=ASSUMED_COURSE_DAYS)
    return None


def overlap_of(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Whether two treatment windows overlap, and by how much.

    Returns {"status", "start", "end", "days", "gap_days"} — `gap_days` is how
    far apart they were when they don't overlap, which is the number that makes
    a stale finding obviously stale.
    """
    if not a["start"] or not b["start"]:
        return {"status": UNKNOWN, "start": None, "end": None, "days": 0, "gap_days": None}

    end_a, end_b = _effective_end(a), _effective_end(b)
    latest_start = max(a["start"], b["start"])
    earliest_end = min(end_a, end_b)

    if latest_start <= earliest_end:
        certain = a["duration_known"] and b["duration_known"]
        return {
            "status": CONCURRENT if certain else POSSIBLE,
            "start": latest_start,
            "end": earliest_end,
            "days": (earliest_end - latest_start).days + 1,
            "gap_days": 0,
        }

    return {
        "status": NOT_CONCURRENT,
        "start": None,
        "end": None,
        "days": 0,
        "gap_days": (latest_start - earliest_end).days,
    }


def _windows_for(term: str, windows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every window whose ingredients match a drug name used in a finding."""
    from document_dedup import _base_ingredient

    key = _base_ingredient(term)
    if not key:
        return []
    return [w for w in windows if any(key == k or key in k or k in key for k in w["ingredients"])]


def _best_overlap(
    groups: Sequence[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Most-exposed overlap across every combination of the named drugs.

    Takes the BEST case for the patient being exposed (the longest genuine
    overlap), because the question a finding answers is "was this ever a live
    risk?" — not "is it a risk on average".
    """
    if len(groups) < 2 or any(not g for g in groups):
        return {"status": UNKNOWN, "start": None, "end": None, "days": 0, "gap_days": None}

    best = None
    smallest_gap = None
    for a in groups[0]:
        for b in groups[1]:
            result = overlap_of(a, b)
            if result["status"] in (CONCURRENT, POSSIBLE):
                if best is None or result["days"] > best["days"] or (
                    best["status"] == POSSIBLE and result["status"] == CONCURRENT
                ):
                    best = result
            elif result["status"] == NOT_CONCURRENT:
                if smallest_gap is None or (result["gap_days"] or 0) < smallest_gap["gap_days"]:
                    smallest_gap = result
    return best or smallest_gap or {
        "status": UNKNOWN, "start": None, "end": None, "days": 0, "gap_days": None,
    }


def _fmt(value: Optional[date]) -> Optional[str]:
    return value.isoformat() if value else None


def _timing_note(result: Dict[str, Any], subjects: Sequence[str]) -> str:
    names = " and ".join(subjects) if subjects else "these medicines"
    if result["status"] == CONCURRENT:
        return (
            f"{names} overlapped for {result['days']} day(s), from "
            f"{_fmt(result['start'])} to {_fmt(result['end'])}. This was a live risk "
            "during that period."
        )
    if result["status"] == POSSIBLE:
        return (
            f"{names} may have overlapped around {_fmt(result['start'])} to "
            f"{_fmt(result['end'])}, but at least one course has no stated duration, so "
            "the dates cannot be confirmed from the documents."
        )
    if result["status"] == NOT_CONCURRENT:
        return (
            f"{names} were never taken at the same time — the courses finished about "
            f"{result['gap_days']} day(s) apart. This is a historical pairing, not a "
            "current risk."
        )
    return (
        f"The dates for {names} could not be read from the documents, so it is not "
        "possible to tell whether the courses overlapped."
    )


def _finding_subjects(finding: Dict[str, Any]) -> List[str]:
    involved = [d for d in (finding.get("medications_involved") or []) if d]
    if involved:
        return involved
    single = finding.get("medication")
    return [single] if single else []


def annotate_findings_with_timing(
    cross_check: Dict[str, Any], timeline: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Adds a `timing` block to every interaction, duplicate and dosage-conflict
    finding, in place:

      timing = {"status", "window_start", "window_end", "overlap_days",
                "gap_days", "note"}

    Findings whose drugs were never concurrent keep their place in the report
    but are marked `not_concurrent`, so a caller can present them as history
    instead of as a live risk.
    """
    windows = build_treatment_windows(timeline)

    for list_name in ("potential_drug_interactions", "duplicate_prescriptions",
                      "conflicting_dosage_instructions"):
        for finding in cross_check.get(list_name) or []:
            if not isinstance(finding, dict):
                continue
            subjects = _finding_subjects(finding)
            groups = [_windows_for(s, windows) for s in subjects]

            if len(subjects) >= 2:
                result = _best_overlap(groups)
            elif len(subjects) == 1:
                # One named drug (a duplicate or dosage conflict): the risk is
                # two of ITS OWN courses running at once.
                own = groups[0] if groups else []
                result = _best_overlap([own, own]) if len(own) > 1 else {
                    "status": UNKNOWN, "start": None, "end": None,
                    "days": 0, "gap_days": None,
                }
            else:
                result = {"status": UNKNOWN, "start": None, "end": None,
                          "days": 0, "gap_days": None}

            finding["timing"] = {
                "status": result["status"],
                "window_start": _fmt(result["start"]),
                "window_end": _fmt(result["end"]),
                "overlap_days": result["days"],
                "gap_days": result["gap_days"],
                "note": _timing_note(result, subjects),
            }

    statuses = [
        f["timing"]["status"]
        for name in ("potential_drug_interactions", "duplicate_prescriptions",
                     "conflicting_dosage_instructions")
        for f in (cross_check.get(name) or [])
        if isinstance(f, dict) and f.get("timing")
    ]
    cross_check["timing_summary"] = {
        "concurrent": statuses.count(CONCURRENT),
        "possible": statuses.count(POSSIBLE),
        "not_concurrent": statuses.count(NOT_CONCURRENT),
        "unknown": statuses.count(UNKNOWN),
        "note": (
            f"{statuses.count(CONCURRENT)} finding(s) involve medicines taken at the same "
            f"time. {statuses.count(NOT_CONCURRENT)} involve courses that never "
            "overlapped — those are historical pairings kept for the record, not current "
            "risks."
        ) if statuses else "No findings to place in time.",
    }
    return cross_check


def concurrent_exposure(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Periods where MORE THAN ONE live prescription supplied the same ingredient
    — the double-dosing exposure a patient can hit without realising, because
    each prescription looks reasonable alone.

    Reports the overlap window and the cumulative daily dose, both arithmetic
    over the record. It does NOT judge whether that total is too high: no
    maximum-dose reference data exists in this system, and asserting one would
    be an ungrounded claim. The number plus the dates is what a pharmacist
    needs to make that call.
    """
    windows = [w for w in build_treatment_windows(timeline) if w["start"]]
    by_ingredient: Dict[str, List[Dict[str, Any]]] = {}
    for window in windows:
        for key in window["ingredients"]:
            by_ingredient.setdefault(key, []).append(window)

    exposures: List[Dict[str, Any]] = []
    for ingredient, group in sorted(by_ingredient.items()):
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                # Two copies of ONE prescription (uploaded twice) are not two
                # concurrent courses — see document_dedup.py.
                if a["prescription_group"] and a["prescription_group"] == b["prescription_group"]:
                    continue
                result = overlap_of(a, b)
                if result["status"] not in (CONCURRENT, POSSIBLE):
                    continue
                doses = [w["daily_dose"] for w in (a, b) if w["daily_dose"] is not None]
                cumulative = round(sum(doses), 3) if len(doses) == 2 else None
                unit = a["dosage_unit"] or b["dosage_unit"]
                exposures.append({
                    "ingredient": ingredient,
                    "status": result["status"],
                    "window_start": _fmt(result["start"]),
                    "window_end": _fmt(result["end"]),
                    "overlap_days": result["days"],
                    "sources": [
                        {"name": w["name"], "date": w["date"],
                         "source_file": w["source_file"], "daily_dose": w["daily_dose"]}
                        for w in (a, b)
                    ],
                    "cumulative_daily_dose": cumulative,
                    "dosage_unit": unit,
                    "note": (
                        f"Between {_fmt(result['start'])} and {_fmt(result['end'])}, two "
                        f"separate prescriptions supplied {ingredient}"
                        + (f", totalling {cumulative} {unit} per day"
                           if cumulative is not None and unit else "")
                        + ". Whether that total is appropriate is for a pharmacist or "
                          "doctor to judge — this is the arithmetic, not a verdict."
                    ),
                })
    return exposures


def risk_calendar(cross_check: Dict[str, Any], timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Findings grouped into the dated periods they were live, most recent first:

      [{"window_start", "window_end", "overlap_days", "risks": [
          {"kind", "subjects", "severity", "confidence", "status"} ]}]

    This is the "in this week, this interaction was active" view. Findings that
    were never concurrent, or whose dates could not be read, are collected
    under a final undated entry rather than dropped.
    """
    annotate_findings_with_timing(cross_check, timeline)

    periods: Dict[Tuple[Optional[str], Optional[str]], Dict[str, Any]] = {}
    for kind, list_name in (
        ("drug_interaction", "potential_drug_interactions"),
        ("duplicate_prescription", "duplicate_prescriptions"),
        ("dosage_conflict", "conflicting_dosage_instructions"),
    ):
        for finding in cross_check.get(list_name) or []:
            if not isinstance(finding, dict):
                continue
            timing = finding.get("timing") or {}
            key = (timing.get("window_start"), timing.get("window_end"))
            entry = periods.setdefault(key, {
                "window_start": key[0],
                "window_end": key[1],
                "overlap_days": timing.get("overlap_days", 0),
                "risks": [],
            })
            entry["risks"].append({
                "kind": kind,
                "subjects": _finding_subjects(finding),
                "severity": finding.get("severity"),
                "confidence": finding.get("confidence"),
                "status": timing.get("status", UNKNOWN),
                "evidence_source": finding.get("evidence_source"),
            })

    dated = [p for p in periods.values() if p["window_start"]]
    undated = [p for p in periods.values() if not p["window_start"]]
    dated.sort(key=lambda p: p["window_start"], reverse=True)

    for period in undated:
        period["label"] = "Not concurrent, or dates unreadable"
    for period in dated:
        period["label"] = f"{period['window_start']} to {period['window_end']}"

    return dated + undated


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # --- Duration parsing --------------------------------------------------
    assert parse_duration_days("14 days") == 14
    assert parse_duration_days("4 weeks") == 28
    assert parse_duration_days("1 month") == 30
    assert parse_duration_days("5 days") == 5
    assert parse_duration_days("As required") is None
    assert parse_duration_days("") is None
    assert parse_duration_days(None) is None

    # --- Date convention inferred from the record --------------------------
    # "14/10/2023" can only be day-first, which settles "09/11/2025".
    assert infer_dayfirst(["09/11/2025", "14/10/2023"]) is True
    assert infer_dayfirst(["03/11/2025", "10/14/2023"]) is False
    assert parse_date("09/11/2025", dayfirst=True) == date(2025, 11, 9)
    assert parse_date("2025-11-09") == date(2025, 11, 9)
    assert parse_date("not a date") is None

    # --- The real record ---------------------------------------------------
    def med(name, ingredient, d, duration, value=None, unit=None, per_day=None, group=None):
        return {"name": name, "ingredients": [ingredient], "date": d,
                "duration": duration, "dosage_value": value, "dosage_unit": unit,
                "frequency_per_day": per_day, "source_file": f"{name}.png",
                "prescription_group": group}

    timeline = {"medications_timeline": [
        med("Paracetamol", "Paracetamol", "09/11/2025", "14 days", 1000, "mg", 3, "rx-0"),
        med("Diclofenac sodium", "Diclofenac", "09/11/2025", "14 days", None, None, 2, "rx-0"),
        med("Omeprazole", "Omeprazole", "09/11/2025", "14 days", 20, "mg", 1, "rx-0"),
        med("Fluconazole", "Fluconazole", "27/03/2025", "4 weeks", 150, "mg", 0.14, "rx-1"),
        med("Cetirizine", "Cetirizine", "26/02/2026", "14 days", 10, "mg", 1, "rx-2"),
        med("Montelukast", "Montelukast", "26/02/2026", "14 days", 10, "mg", 1, "rx-2"),
        med("Chlorpheniramine", "Chlorpheniramine", "14/10/2023", "5 days", 4, "mg", 1, "rx-3"),
    ]}

    report = {
        "potential_drug_interactions": [
            {"medications_involved": ["Paracetamol", "Diclofenac"],
             "explanation": "Additive GI/renal risk.", "severity": "moderate",
             "confidence": 0.6},
            {"medications_involved": ["Fluconazole", "Montelukast"],
             "explanation": "CYP inhibition raises montelukast levels.",
             "severity": "moderate", "confidence": 0.6},
            {"medications_involved": ["Cetirizine", "Chlorpheniramine"],
             "explanation": "Additive sedation.", "severity": "moderate",
             "confidence": 0.6},
            {"medications_involved": ["Cetirizine", "Montelukast"],
             "explanation": "Both prescribed together for rhinitis.",
             "severity": "low", "confidence": 0.5},
        ],
        "duplicate_prescriptions": [],
        "conflicting_dosage_instructions": [],
        "allergy_conflicts": [],
    }

    annotate_findings_with_timing(report, timeline)
    by_pair = {
        " + ".join(f["medications_involved"]): f["timing"]
        for f in report["potential_drug_interactions"]
    }

    # Genuinely concurrent — same prescription, same 14 days.
    assert by_pair["Paracetamol + Diclofenac"]["status"] == CONCURRENT
    assert by_pair["Paracetamol + Diclofenac"]["window_start"] == "2025-11-09"
    assert by_pair["Paracetamol + Diclofenac"]["overlap_days"] == 15

    # Never concurrent — these are the false alarms this module exists for.
    for pair, expected_gap in (("Fluconazole + Montelukast", 300),
                               ("Cetirizine + Chlorpheniramine", 800)):
        timing = by_pair[pair]
        assert timing["status"] == NOT_CONCURRENT, (pair, timing)
        assert timing["gap_days"] > expected_gap, (pair, timing["gap_days"])
        assert "never taken at the same time" in timing["note"]

    # Two drugs from the SAME prescription are concurrent.
    assert by_pair["Cetirizine + Montelukast"]["status"] == CONCURRENT

    summary = report["timing_summary"]
    assert summary["concurrent"] == 2, summary
    assert summary["not_concurrent"] == 2, summary

    # --- Unknown duration is "possible", never "concurrent" ----------------
    open_ended = {"medications_timeline": [
        med("DrugA", "druga", "01/01/2026", "As required", 10, "mg", 1, "g1"),
        med("DrugB", "drugb", "05/01/2026", "14 days", 10, "mg", 1, "g2"),
    ]}
    open_report = {"potential_drug_interactions": [
        {"medications_involved": ["DrugA", "DrugB"], "explanation": "x",
         "severity": "low", "confidence": 0.5}]}
    annotate_findings_with_timing(open_report, open_ended)
    assert open_report["potential_drug_interactions"][0]["timing"]["status"] == POSSIBLE

    # --- Concurrent double-dosing exposure ---------------------------------
    doubled = {"medications_timeline": [
        med("Panadol", "Paracetamol", "09/11/2025", "14 days", 1000, "mg", 3, "rx-A"),
        med("Calpol", "Paracetamol", "12/11/2025", "10 days", 500, "mg", 4, "rx-B"),
        # Same prescription uploaded twice must NOT count as a second course.
        med("Panadol", "Paracetamol", "09/11/2025", "14 days", 1000, "mg", 3, "rx-A"),
    ]}
    exposures = concurrent_exposure(doubled)
    assert len(exposures) == 2, [e["sources"] for e in exposures]
    assert all(e["ingredient"] == "paracetamol" for e in exposures)
    top = exposures[0]
    assert top["cumulative_daily_dose"] == 5000.0, top["cumulative_daily_dose"]
    assert top["window_start"] == "2025-11-12"
    assert "for a pharmacist or doctor to judge" in top["note"]

    # A single course produces no exposure finding.
    assert concurrent_exposure({"medications_timeline": [
        med("Panadol", "Paracetamol", "09/11/2025", "14 days", 1000, "mg", 3, "rx-A")]}) == []

    # --- Calendar ----------------------------------------------------------
    calendar = risk_calendar(report, timeline)
    assert calendar[0]["window_start"] == "2026-02-26", calendar[0]
    assert calendar[-1]["label"] == "Not concurrent, or dates unreadable"

    print("Treatment windows from the real record:")
    for w in build_treatment_windows(timeline):
        print(f"  {(w['name'] or '')[:22]:24} {w['start']} -> {w['end'] or 'open-ended'}")

    print("\nInteraction findings placed in time:")
    for f in report["potential_drug_interactions"]:
        t = f["timing"]
        mark = {CONCURRENT: "LIVE", POSSIBLE: "MAYBE", NOT_CONCURRENT: "past",
                UNKNOWN: "?"}[t["status"]]
        print(f"  [{mark:5}] {' + '.join(f['medications_involved'])}")
        print(f"          {t['note']}")

    print("\nRisk calendar:")
    for period in calendar:
        print(f"  {period['label']}: "
              + ", ".join(" + ".join(r["subjects"]) for r in period["risks"]))

    print("\nAll checks passed.")
