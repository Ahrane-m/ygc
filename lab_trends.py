"""
Lab Result Trend Tracking
=========================================
Takes a patient's `lab_results_timeline` (the flattened, already-merged
list build_patient_timeline() produces — one entry per lab value per
visit, each with test_name/value/unit/reference_range/flag/date/
source_file) and, per test, tracks how the value moved across visits:
direction of drift, whether/when it crossed out of the reference range,
and whether it's approaching a boundary even while still "normal".

Deliberately deterministic, no LLM call: direction and threshold-crossing
are arithmetic facts about the extracted numbers, not something that
benefits from probabilistic reasoning — matching the same "compute what
code can determine for certain" philosophy medical_extractor.py already
uses for detect_exact_duplicate_medications() alongside the LLM cross-
check. The "plain language" explanation is a template filled from those
computed facts, not a generated summary, so it can't say anything the
numbers don't support.

WRITTEN FOR THE PATIENT, NOT THE CLINICIAN
------------------------------------------
`explanation` is read by whoever uploaded the documents, who generally has
no medical background — so it avoids the vocabulary a lab report assumes.
Abbreviations printed on the report ("ALT", "TSH", "eGFR") are spelled out
via TEST_GLOSSARY, "reference range" becomes "the normal range", a "high"
flag becomes "higher than the normal range", and readings are listed as
"91 mg/dL on 05 Jan 2026, then 103 mg/dL on 20 Apr 2026".

The glossary describes only what each test MEASURES, never what a result
means — "ALT is one of the tests used to check how the liver is working" is
a description; "a high ALT means liver damage" is a diagnosis this module
must never make. The precise values, units, ranges and flags stay available
in the structured fields alongside the text, so nothing is lost for callers
that need them (retrieval.py renders both).

Dates in this pipeline arrive in wildly inconsistent formats (mixed
languages, "02-Jan-2020, 03:26 PM" vs "04-07-2019" vs "05 Jan 2026") —
see the varied `date` fields real extractions produce. dateutil.parser is
used with best-effort fuzzy parsing; anything unparseable is dropped from
the trend (noted in confidence) rather than mis-sorted.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dateutil_parser

# A value within this fraction of the reference range width, relative to
# the boundary it's moving toward, counts as "approaching" that boundary
# even though it hasn't crossed yet.
APPROACHING_THRESHOLD_FRACTION = 0.15

# ---------------------------------------------------------------------------
# Plain-language glossary
#
# Lab reports print abbreviations ("ALT", "TSH", "eGFR") that mean nothing to
# most people reading their own results, so `explanation` below spells out
# what each test looks at. Keyed by substrings matched against the test name.
#
# EVERY ENTRY DESCRIBES WHAT THE TEST MEASURES — NEVER WHAT A RESULT MEANS.
# "ALT is one of the tests used to check how the liver is working" is a
# description of the test. "A high ALT means liver damage" is a diagnosis, and
# would break the promise the rest of this module keeps: that the explanation
# cannot say anything the extracted numbers support. Nothing here interprets a
# value, states a cause, or implies a condition.
#
# Deliberately not exhaustive. A test that isn't listed simply keeps its
# printed name with no gloss, which is honest — a wrong gloss on a test we
# guessed at would be worse than none.
# ---------------------------------------------------------------------------

TEST_GLOSSARY: List[Tuple[Tuple[str, ...], str, str]] = [
    (("alt", "sgpt", "ast", "sgot", "ggt", "bilirubin"),
     "a liver test",
     "one of the tests used to check how the liver is working"),
    (("alkaline phosphatase", "alp"),
     "a liver and bone test",
     "a test used to check the liver and the bones"),
    (("creatinine", "egfr", "gfr", "urea", "bun"),
     "a kidney test",
     "a test used to check how well the kidneys are filtering the blood"),
    (("hba1c", "a1c"),
     "an average blood sugar test",
     "a test showing the average blood sugar level over roughly the last "
     "two to three months"),
    (("glucose",),
     "a blood sugar test",
     "a test showing the amount of sugar in the blood"),
    (("tsh", "thyroid", "t3", "t4"),
     "a thyroid test",
     "a test for the thyroid, the gland in the neck that helps control how "
     "the body uses energy"),
    (("ldl", "hdl", "cholesterol", "triglyceride", "lipid"),
     "a cholesterol test",
     "a test measuring fats in the blood, such as cholesterol"),
    (("hemoglobin", "haemoglobin", "hgb"),
     "a blood count test",
     "a test measuring the part of the red blood cells that carries oxygen "
     "around the body"),
    (("hematocrit", "haematocrit", "rbc", "red blood cell"),
     "a blood count test",
     "a test measuring the red blood cells, which carry oxygen around the body"),
    (("wbc", "white blood cell", "leukocyte"),
     "a blood count test",
     "a test measuring the white blood cells, which the body uses to fight "
     "infection"),
    (("platelet",),
     "a blood count test",
     "a test measuring platelets, the part of the blood that helps it clot"),
    (("inr", "prothrombin", "aptt", "ptt"),
     "a blood clotting test",
     "a test measuring how long the blood takes to clot"),
    (("uric acid", "urate"),
     "a uric acid test",
     "a test measuring uric acid, a waste product that the body normally "
     "passes out in urine"),
    (("psa",),
     "a prostate test",
     "a test measuring a substance made by the prostate gland"),
    (("troponin", "bnp"),
     "a heart test",
     "a test measuring a substance in the blood that comes from the heart"),
    (("crp", "c-reactive", "esr", "sedimentation"),
     "an inflammation test",
     "a test that looks for signs of inflammation somewhere in the body"),
    (("vitamin d", "vitamin b12", "b12", "folate", "ferritin", "iron"),
     "a vitamin or mineral test",
     "a test measuring the level of a vitamin or mineral in the blood"),
    (("sodium", "potassium", "chloride", "calcium", "magnesium"),
     "a mineral test",
     "a test measuring the minerals and salts in the blood"),
]

# Net change smaller than this fraction of the range width is "stable"
# rather than a real directional trend (guards against noise/rounding).
STABLE_CHANGE_FRACTION = 0.10


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        return dateutil_parser.parse(date_str, fuzzy=True)
    except (ValueError, OverflowError):
        return None


def _parse_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(\.\d+)?", str(value))
    return float(match.group()) if match else None


def _parse_range(reference_range: Optional[str]) -> Optional[Tuple[float, float]]:
    if not reference_range or not isinstance(reference_range, str):
        return None
    # Match "low - high" as an explicit pair, not a free-for-all number scan:
    # a naive findall(r"-?\d+...") on "70-99" mis-reads the separating hyphen
    # as a negative sign and parses it as [70, -99] instead of [70, 99].
    match = re.match(
        r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$", reference_range.strip()
    )
    if not match:
        return None
    low, high = float(match.group(1)), float(match.group(2))
    return (low, high) if low <= high else (high, low)


def _group_by_test(lab_results_timeline: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for entry in lab_results_timeline:
        name = (entry.get("test_name") or "").strip()
        if not name:
            continue
        groups.setdefault(name.lower(), []).append(entry)
    # Use the most common casing seen for display, keyed by lowercase for grouping.
    display_names: Dict[str, str] = {}
    for entry in lab_results_timeline:
        name = (entry.get("test_name") or "").strip()
        if name:
            display_names.setdefault(name.lower(), name)
    return {display_names[k]: v for k, v in groups.items()}


def _flag_sequence_phrase(flags: List[str]) -> str:
    return " → ".join(flags)


def _describe_test(test_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Looks a test name up in TEST_GLOSSARY, returning
    (short_plain_name, what_it_measures) or (None, None) if it isn't listed.

    Matches on word boundaries so a short key like "alt" doesn't fire on
    "Cobalt" or "Alkaline Phosphatase", and checks longer keys first so
    "hba1c" wins over "a1c" and "vitamin b12" over "b12"."""
    name = (test_name or "").lower()
    for keywords, plain_name, measures in TEST_GLOSSARY:
        for kw in sorted(keywords, key=len, reverse=True):
            if re.search(rf"\b{re.escape(kw)}\b", name):
                return plain_name, measures
    return None, None


def _plain_direction(direction: str) -> str:
    """Turns the computed direction into words rather than jargon."""
    return {
        "stable": "has stayed about the same",
        "increasing": "has been going up",
        "decreasing": "has been going down",
        "fluctuating (net increasing)": "has moved up and down, but overall it has gone up",
        "fluctuating (net decreasing)": "has moved up and down, but overall it has gone down",
    }.get(direction, "has changed")


def _plain_flag(flag: str) -> str:
    """'high'/'low' as a phrase a reader doesn't have to interpret."""
    return {
        "high": "higher than the normal range",
        "low": "lower than the normal range",
    }.get(flag, "outside the normal range")


def _direction(values: List[float], range_bounds: Optional[Tuple[float, float]]) -> str:
    net_change = values[-1] - values[0]
    width = (range_bounds[1] - range_bounds[0]) if range_bounds else max(abs(v) for v in values) or 1.0
    if width == 0:
        width = 1.0

    if abs(net_change) < STABLE_CHANGE_FRACTION * width:
        return "stable"

    deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    same_sign_as_net = all((d >= 0) == (net_change >= 0) or d == 0 for d in deltas)
    base = "increasing" if net_change > 0 else "decreasing"
    return base if same_sign_as_net else f"fluctuating (net {base})"


def _approaching_boundary(
    last_value: float, last_flag: str, range_bounds: Optional[Tuple[float, float]], direction: str
) -> bool:
    if last_flag != "normal" or not range_bounds:
        return False
    low, high = range_bounds
    width = high - low
    if width <= 0:
        return False
    if "increasing" in direction and (high - last_value) <= APPROACHING_THRESHOLD_FRACTION * width:
        return True
    if "decreasing" in direction and (last_value - low) <= APPROACHING_THRESHOLD_FRACTION * width:
        return True
    return False


def _crossing_point(points: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """First point where the flag changed from 'normal' to something else,
    scanning chronologically. Returns None if it never crossed, or if it
    was already abnormal at the first available reading (nothing to
    pinpoint — it didn't drift there during the observed window)."""
    for i in range(1, len(points)):
        if points[i - 1]["flag"] == "normal" and points[i]["flag"] != "normal":
            return points[i]
    return None


def _explain(
    test_name: str,
    unit: str,
    points: List[Dict[str, Any]],
    direction: str,
    range_bounds: Optional[Tuple[float, float]],
    crossing: Optional[Dict[str, Any]],
    approaching: bool,
) -> str:
    """
    Writes the trend out in language a reader with no medical background can
    follow: the test's abbreviation is spelled out, "reference range" becomes
    "the normal range", flags become "higher than the normal range", and the
    readings are listed as "X on <date>, then Y on <date>" rather than joined
    by arrows.

    Still assembled from the computed values only — same guarantee as before,
    that this text cannot claim anything the numbers don't support. It says
    what the numbers did; it never says what caused it or what it means for
    the person, which stays a clinician's call.
    """
    values = [p["_value"] for p in points]
    dates = [p.get("date") or "an unspecified date" for p in points]
    readings = [f"{v:g} {unit}".strip() + f" on {d}" for v, d in zip(values, dates)]
    trail = readings[0] + "".join(f", then {r}" for r in readings[1:])

    plain_name, measures = _describe_test(test_name)
    # Lead with the printed name (it's what's on their report), then explain
    # the abbreviation once, in brackets.
    if measures:
        opening = f"{test_name} is {measures}."
    else:
        opening = ""

    movement = _plain_direction(direction)
    base = (
        f"{opening} Looking at the {len(points)} times this was tested, the result "
        f"{movement}: {trail}."
    ).strip()

    if range_bounds:
        base += (
            f" The normal range for this test is {range_bounds[0]:g} to "
            f"{range_bounds[1]:g} {unit}.".rstrip()
        )

    if crossing is not None:
        base += (
            f" The result went from inside the normal range to "
            f"{_plain_flag(crossing['flag'])} at the "
            f"{crossing.get('date') or 'most recent'} test, and has stayed there since."
        )
    elif points[-1]["flag"] != "normal" and points[0]["flag"] != "normal":
        base += (
            f" The result was already {_plain_flag(points[0]['flag'])} at the earliest "
            f"test on file, and it is still {_plain_flag(points[-1]['flag'])}."
        )
    elif approaching:
        edge = "top" if "increasing" in direction else "bottom"
        base += (
            f" The result is still inside the normal range, but it has been moving "
            f"closer to the {edge} of that range. Nothing here is outside the normal "
            "range yet — it is just worth keeping an eye on."
        )
    elif direction == "stable":
        base += " There is no clear change here."

    # The "we can't tell you what this means" caution goes only on results
    # that are outside the normal range or heading that way — those are the
    # ones someone might read too much into. Repeating it on every stable,
    # in-range test would turn a readable list into boilerplate and train the
    # reader to skip it, which is where it would then be missed.
    if crossing is not None or approaching or points[-1]["flag"] != "normal":
        base += (
            " This only shows what was measured and when — not what caused it or what "
            "it means. A doctor or pharmacist can explain that."
        )
    return base


def track_lab_trends(timeline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Groups `timeline["lab_results_timeline"]` by test_name and analyzes
    each test with 2+ usable (dated, numeric) data points for directional
    drift and reference-range crossings.

    Returns:
      {
        "trends": [
          {
            "test_name": str,      # exactly as printed on the report
            "plain_name": str | None,        # e.g. "a kidney test"
            "what_it_measures": str | None,  # one plain sentence, or None if
                                             # the test isn't in TEST_GLOSSARY
            "unit": str, "reference_range": "low-high" | None,
            "data_points": [{"date", "value", "flag", "source_file"}, ...]  # chronological
            "direction": "increasing" | "decreasing" | "stable" | "fluctuating (net increasing/decreasing)",
            "flag_sequence": "normal → normal → high",
            "crossed_into_abnormal_at": {"date":..., "flag":...} | None,
            "approaching_threshold": bool,
            "confidence": float,   # lower if dates/values had to be dropped, or reference ranges disagreed
            "explanation": str,    # plain-language (no abbreviations or jargon),
                                   # template-generated from the numbers above
          }, ...
        ],
        "insufficient_data": [{"test_name": str, "reason": str}, ...],
        "note": "... not a diagnosis, consult a clinician/pharmacist ..."
      }
    """
    lab_results_timeline = timeline.get("lab_results_timeline", [])
    grouped = _group_by_test(lab_results_timeline)

    trends: List[Dict[str, Any]] = []
    insufficient: List[Dict[str, Any]] = []

    for test_name, entries in grouped.items():
        usable = []
        dropped = 0
        units_seen = set()
        ranges_seen = set()
        for e in entries:
            dt = _parse_date(e.get("date"))
            val = _parse_value(e.get("value"))
            if dt is None or val is None:
                dropped += 1
                continue
            usable.append({
                "_dt": dt, "_value": val,
                "date": e.get("date"), "value": e.get("value"),
                "flag": e.get("flag") or "unknown",
                "unit": e.get("unit") or "",
                "reference_range": e.get("reference_range"),
                "source_file": e.get("source_file"),
                "confidence": e.get("confidence", 1.0),
            })
            if e.get("unit"):
                units_seen.add(e["unit"])
            if e.get("reference_range"):
                ranges_seen.add(e["reference_range"])

        if len(usable) < 2:
            insufficient.append({
                "test_name": test_name,
                "reason": (
                    f"only {len(usable)} usable data point(s) with a parseable date and numeric value "
                    f"(need at least 2 to establish a trend); {dropped} entrie(s) were dropped."
                    if usable else
                    f"no entries had both a parseable date and a numeric value ({dropped} dropped)."
                ),
            })
            continue

        usable.sort(key=lambda p: p["_dt"])

        unit = usable[-1]["unit"]
        range_bounds = _parse_range(usable[-1]["reference_range"])

        direction = _direction([p["_value"] for p in usable], range_bounds)
        crossing = _crossing_point(usable)
        approaching = _approaching_boundary(usable[-1]["_value"], usable[-1]["flag"], range_bounds, direction)

        # Confidence: average of the source extraction confidences, discounted
        # for dropped/unusable readings and for disagreeing units or reference
        # ranges across visits (both make the trend less trustworthy).
        confidences = [p["confidence"] for p in usable if isinstance(p["confidence"], (int, float))]
        base_confidence = sum(confidences) / len(confidences) if confidences else 0.7
        if dropped:
            base_confidence *= max(0.5, 1 - 0.15 * dropped)
        if len(units_seen) > 1 or len(ranges_seen) > 1:
            base_confidence *= 0.7

        plain_name, measures = _describe_test(test_name)

        trends.append({
            "test_name": test_name,
            # Both null for a test not in TEST_GLOSSARY — a caller rendering
            # these should fall back to test_name rather than show "None".
            "plain_name": plain_name,
            "what_it_measures": measures,
            "unit": unit,
            "reference_range": usable[-1]["reference_range"],
            "data_points": [
                {"date": p["date"], "value": p["value"], "flag": p["flag"], "source_file": p["source_file"]}
                for p in usable
            ],
            "direction": direction,
            "flag_sequence": _flag_sequence_phrase([p["flag"] for p in usable]),
            "crossed_into_abnormal_at": (
                {"date": crossing["date"], "flag": crossing["flag"]} if crossing else None
            ),
            "approaching_threshold": approaching,
            "confidence": round(min(base_confidence, 0.97), 2),
            "explanation": _explain(test_name, unit, usable, direction, range_bounds, crossing, approaching),
        })

    return {
        "trends": trends,
        "insufficient_data": insufficient,
        "note": (
            "This is worked out directly from the numbers printed on the uploaded lab "
            "reports and the normal ranges printed alongside them. It shows how results "
            "have changed over time — it does not say what caused a change, and it is not "
            "a diagnosis. It also cannot see anything your reports don't show, such as how "
            "you are feeling or any other health condition. A doctor or pharmacist can "
            "explain what these results mean for you."
        ),
    }


if __name__ == "__main__":
    import sys

    # flag_sequence is joined with "→", which a cp1252 console (the Windows
    # default, and what Git Bash uses here) cannot encode — the self-test
    # would die printing its own results rather than on a failed assertion.
    # Same guard inspect_records.py and the extractor CLI use.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Self-test using John's three real lab reports from this project's
    # test data: Fasting Glucose drifts from normal into high, ALT jumps
    # into high only at the last test, Creatinine rises but stays just
    # inside the normal range (approaching the upper boundary).
    demo_timeline = {
        "lab_results_timeline": [
            {"test_name": "Fasting Glucose", "value": "91", "unit": "mg/dL", "reference_range": "70-99", "flag": "normal", "confidence": 0.95, "date": "05 Jan 2026", "source_file": "John_Lab_Report_1.pdf"},
            {"test_name": "ALT", "value": "24", "unit": "U/L", "reference_range": "7-56", "flag": "normal", "confidence": 0.95, "date": "05 Jan 2026", "source_file": "John_Lab_Report_1.pdf"},
            {"test_name": "Creatinine", "value": "0.92", "unit": "mg/dL", "reference_range": "0.74-1.35", "flag": "normal", "confidence": 0.95, "date": "05 Jan 2026", "source_file": "John_Lab_Report_1.pdf"},
            {"test_name": "Fasting Glucose", "value": "103", "unit": "mg/dL", "reference_range": "70-99", "flag": "high", "confidence": 0.95, "date": "20 Apr 2026", "source_file": "John_Lab_Report_2.pdf"},
            {"test_name": "ALT", "value": "41", "unit": "U/L", "reference_range": "7-56", "flag": "normal", "confidence": 0.95, "date": "20 Apr 2026", "source_file": "John_Lab_Report_2.pdf"},
            {"test_name": "Creatinine", "value": "1.08", "unit": "mg/dL", "reference_range": "0.74-1.35", "flag": "normal", "confidence": 0.95, "date": "20 Apr 2026", "source_file": "John_Lab_Report_2.pdf"},
            {"test_name": "Fasting Glucose", "value": "118", "unit": "mg/dL", "reference_range": "70-99", "flag": "high", "confidence": 0.95, "date": "30 Aug 2026", "source_file": "John_Lab_Report_3.pdf"},
            {"test_name": "ALT", "value": "82", "unit": "U/L", "reference_range": "7-56", "flag": "high", "confidence": 0.95, "date": "30 Aug 2026", "source_file": "John_Lab_Report_3.pdf"},
            {"test_name": "Creatinine", "value": "1.32", "unit": "mg/dL", "reference_range": "0.74-1.35", "flag": "normal", "confidence": 0.95, "date": "30 Aug 2026", "source_file": "John_Lab_Report_3.pdf"},
        ]
    }
    result = track_lab_trends(demo_timeline)
    by_name = {t["test_name"]: t for t in result["trends"]}

    assert by_name["Fasting Glucose"]["direction"] == "increasing"
    assert by_name["Fasting Glucose"]["crossed_into_abnormal_at"]["date"] == "20 Apr 2026"

    assert by_name["ALT"]["direction"] == "increasing"
    assert by_name["ALT"]["crossed_into_abnormal_at"]["date"] == "30 Aug 2026"

    assert by_name["Creatinine"]["direction"] == "increasing"
    assert by_name["Creatinine"]["crossed_into_abnormal_at"] is None
    assert by_name["Creatinine"]["approaching_threshold"] is True

    # Regression check for the reference-range parsing bug (hyphen
    # mis-read as a negative sign): must render as 70 to 99, not -99 to 70.
    assert "70 to 99 mg/dL" in by_name["Fasting Glucose"]["explanation"], \
        by_name["Fasting Glucose"]["explanation"]

    # Plain-language checks: abbreviations are spelled out, lab-report jargon
    # is gone, and the readings read as sentences rather than arrow trails.
    glucose_explanation = by_name["Fasting Glucose"]["explanation"]
    assert "amount of sugar in the blood" in glucose_explanation, glucose_explanation
    assert "91 mg/dL on 05 Jan 2026" in glucose_explanation, glucose_explanation
    assert "higher than the normal range" in glucose_explanation, glucose_explanation
    assert by_name["ALT"]["plain_name"] == "a liver test"
    assert "liver" in by_name["ALT"]["explanation"]
    assert by_name["Creatinine"]["plain_name"] == "a kidney test"

    for trend in result["trends"]:
        text = trend["explanation"]
        for jargon in ("reference range", "abnormal", "boundary", "flagged",
                       "drift", "→"):
            assert jargon not in text, f"{trend['test_name']}: leaked '{jargon}' -> {text}"

    # The glossary must not fire on unrelated names that merely contain a key
    # as a substring, and longer keys must win over shorter ones.
    assert _describe_test("Cobalt") == (None, None)
    assert _describe_test("Vitamin B12")[0] == "a vitamin or mineral test"
    assert _describe_test("HbA1c")[0] == "an average blood sugar test"
    assert _describe_test("Some Unlisted Assay") == (None, None)

    # Regression check for the "fluctuating (net increasing)" boundary-wording
    # bug: direction.startswith("increasing") used to miss this case (it starts
    # with "fluctuating"), so the explanation wrongly said "lower boundary"
    # for a value that was actually climbing toward the upper one.
    fluctuating_timeline = {
        "lab_results_timeline": [
            {"test_name": "Glucose", "value": "75", "unit": "mg/dL", "reference_range": "70-100", "flag": "normal", "confidence": 0.9, "date": "2026-01-01", "source_file": "a.pdf"},
            {"test_name": "Glucose", "value": "98", "unit": "mg/dL", "reference_range": "70-100", "flag": "normal", "confidence": 0.9, "date": "2026-02-01", "source_file": "b.pdf"},
            {"test_name": "Glucose", "value": "97", "unit": "mg/dL", "reference_range": "70-100", "flag": "normal", "confidence": 0.9, "date": "2026-03-01", "source_file": "c.pdf"},
        ]
    }
    fluctuating_result = track_lab_trends(fluctuating_timeline)
    glucose_trend = fluctuating_result["trends"][0]
    assert glucose_trend["direction"] == "fluctuating (net increasing)", glucose_trend["direction"]
    assert glucose_trend["approaching_threshold"] is True
    assert "top of that range" in glucose_trend["explanation"], glucose_trend["explanation"]
    assert "bottom of that range" not in glucose_trend["explanation"], glucose_trend["explanation"]

    for t in result["trends"]:
        print(f"--- {t['test_name']} ---")
        print(" direction:", t["direction"], "| flags:", t["flag_sequence"], "| confidence:", t["confidence"])
        print(" ", t["explanation"])
        print()

    print("All checks passed.")
