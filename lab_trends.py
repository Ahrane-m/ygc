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

TESTS WITH ONLY ONE READING
---------------------------
A trend needs two points, but a first-ever result is exactly the one a
patient most wants explained — and used to arrive here, land in
`insufficient_data`, and reach the reader as a bare number with nothing
beside it. Those now come back in `single_results` with a low/normal/high
status instead, worked out against a range in this strict order:

  1. the range PRINTED ON THEIR OWN REPORT, always preferred — reference
     intervals belong to the assay, and the lab that ran the test is more
     authoritative about its own range than any table here;
  2. failing that, a general interval from reference_intervals.py, chosen
     for the patient's sex and age because half these tests genuinely have
     different intervals for a man and a woman (hemoglobin, creatinine,
     ferritin, uric acid);
  3. failing that, no status at all — the value is shown and labelled as
     having no range available, rather than compared against a guess.

Same deterministic, no-LLM contract as the trends: a status is arithmetic
against an interval, and the explanation is a template filled from it. It
says where the number sits. It never says what sitting there means.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as dateutil_parser

from identity_guard import infer_birth_year
from reference_intervals import (
    canonical_test,
    is_main_test,
    lookup_interval,
    to_canonical_value,
)

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


# Digit grouping, e.g. the "100,000" in "> 100,000 CFU/mL". Matched only as
# whole groups of three so a decimal comma ("9,2 g/dL", how much of the world
# writes 9.2) is left alone rather than silently turned into 92.
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")


def _strip_digit_grouping(text: str) -> str:
    return _THOUSANDS_SEPARATOR.sub("", text)


def _parse_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # Without stripping the grouping first, the number regex stops at the
    # comma: "> 100,000 CFU/mL" parsed as 100, a thousandfold understatement
    # of a urine culture colony count.
    match = re.search(r"-?\d+(\.\d+)?", _strip_digit_grouping(str(value)))
    return float(match.group()) if match else None


def _parse_value_span(value: Any) -> Optional[Tuple[float, float]]:
    """A reading as (low, high). Most are a single number, so low == high.

    Microscopy results are routinely reported as a SPAN rather than a point —
    "30 - 40 /hpf" of pus cells, "3 - 5 /hpf" of red cells — because the
    technician counted a range across fields. Reading only the first number
    off those understates the result and, worse, prints a figure back to the
    patient that isn't the one on their report.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value), float(value)

    text = _strip_digit_grouping(str(value))
    span = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE
    )
    if span:
        low, high = float(span.group(1)), float(span.group(2))
        return (low, high) if low <= high else (high, low)

    single = _parse_value(value)
    return (single, single) if single is not None else None


Bounds = Tuple[Optional[float], Optional[float]]


def _parse_range(reference_range: Optional[str]) -> Optional[Bounds]:
    """Reads a printed reference range into (low, high). Either side may be
    None for a one-sided range — "<200" for total cholesterol and ">60" for
    eGFR are how labs actually print those, and returning None for the whole
    thing (as this used to) threw away a usable bound and silently disabled
    both the status and the approaching-boundary check for every lipid panel.
    """
    if not reference_range or not isinstance(reference_range, str):
        return None
    # Same grouping hazard as _parse_value: "<10,000" read as a bare 10 turns
    # a colony-count threshold into a thousandth of itself.
    text = _strip_digit_grouping(reference_range.strip())
    if not text:
        return None

    # "low - high" as an explicit PAIR, not a free-for-all number scan: a
    # naive findall(r"-?\d+...") on "70-99" mis-reads the separating hyphen
    # as a negative sign and parses it as [70, -99] instead of [70, 99].
    # Trailing text after the pair is tolerated so "4.0-11.0 x10^9/L" parses.
    pair = re.match(
        r"^\s*(-?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(-?\d+(?:\.\d+)?)(?:\b|$)",
        text, re.IGNORECASE,
    )
    if pair:
        low, high = float(pair.group(1)), float(pair.group(2))
        return (low, high) if low <= high else (high, low)

    # Alternation order matters: "<" listed before "<=" would match the "<"
    # of "<=5.7" and then choke on the "=".
    upper = re.match(
        r"^\s*(?:<=|≤|<|up\s+to|less\s+than)\s*(-?\d+(?:\.\d+)?)(?:\b|$)", text, re.IGNORECASE
    )
    if upper:
        return (None, float(upper.group(1)))

    lower = re.match(
        r"^\s*(?:>=|≥|>|at\s+least|greater\s+than)\s*(-?\d+(?:\.\d+)?)(?:\b|$)", text, re.IGNORECASE
    )
    if lower:
        return (float(lower.group(1)), None)

    return None


def _range_width(range_bounds: Optional[Bounds]) -> Optional[float]:
    """Width of a CLOSED range. None for a one-sided or degenerate one — the
    callers that want a width (the stability dead-band, the approaching-a-
    boundary check) are measuring a fraction of it, which a range open at one
    end has no meaningful value for."""
    if not range_bounds:
        return None
    low, high = range_bounds
    if low is None or high is None:
        return None
    width = high - low
    return width if width > 0 else None


def _classify_span(span: Tuple[float, float], range_bounds: Optional[Bounds]) -> str:
    """Classifies a reading reported as a span ("30 - 40 /hpf") against a
    range. Any part of the span outside the range makes the reading outside
    it — a count of 3-5 red cells against a "below 3" range is a result above
    that range, and classifying it on the low end alone would call it normal.
    """
    if not range_bounds:
        return "unknown"
    low_value, high_value = span
    low, high = range_bounds
    if high is not None and high_value > high:
        return "high"
    if low is not None and low_value < low:
        return "low"
    return "normal"


def _classify(value: float, range_bounds: Optional[Bounds]) -> str:
    """"low" / "normal" / "high" for one value against one range, or
    "unknown" with no range to compare against. Handles one-sided ranges:
    a value can only be "high" relative to a range that has an upper bound."""
    if not range_bounds:
        return "unknown"
    low, high = range_bounds
    if high is not None and value > high:
        return "high"
    if low is not None and value < low:
        return "low"
    return "normal"


def _group_by_test(lab_results_timeline: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Groups readings that are the SAME test even when the reports spelled it
    differently — "Fasting Glucose" on one visit and "FBS" on the next are one
    test with two readings, and grouping them by printed name (as this used to)
    produced two one-reading tests and therefore no trend at all.

    Keyed on reference_intervals.canonical_test(); anything outside that table
    falls back to its lowercased printed name, which still groups exact repeats
    and never merges two tests it cannot vouch for being the same.

    Returns {group_key: {"display_name", "test_id", "entries"}}. display_name
    is the first spelling seen, so the reader still sees what their own report
    printed rather than an internal id.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for entry in lab_results_timeline:
        name = (entry.get("test_name") or "").strip()
        if not name:
            continue
        test_id = canonical_test(name)
        key = test_id or name.lower()
        group = groups.setdefault(
            key, {"display_name": name, "test_id": test_id, "entries": [], "names": []}
        )
        group["entries"].append(entry)
        if name not in group["names"]:
            group["names"].append(name)
    return groups


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


def _range_phrase(range_bounds: Bounds, unit: str) -> str:
    """A range written the way it would be said out loud. One-sided ranges
    become "anything below 200 mg/dL" rather than being rendered with an
    invented opposite bound the report never printed."""
    # Digit grouping on the way back out: a colony-count threshold reads as
    # "10,000", the way the report printed it, not "10000".
    low, high = range_bounds
    if low is not None and high is not None:
        return f"{low:,g} to {high:,g} {unit}".strip()
    if high is not None:
        return f"anything below {high:,g} {unit}".strip()
    return f"anything above {low:,g} {unit}".strip()


def _direction(values: List[float], range_bounds: Optional[Bounds]) -> str:
    net_change = values[-1] - values[0]
    # A one-sided range gives no width to take the dead-band from, so those
    # fall back to the values' own magnitude exactly as a rangeless test does.
    width = _range_width(range_bounds) or max(abs(v) for v in values) or 1.0

    if abs(net_change) < STABLE_CHANGE_FRACTION * width:
        return "stable"

    deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    same_sign_as_net = all((d >= 0) == (net_change >= 0) or d == 0 for d in deltas)
    base = "increasing" if net_change > 0 else "decreasing"
    return base if same_sign_as_net else f"fluctuating (net {base})"


def _approaching_boundary(
    last_value: float, last_flag: str, range_bounds: Optional[Bounds], direction: str
) -> bool:
    if last_flag != "normal" or not range_bounds:
        return False
    # None for a one-sided range: "within 15% of the range width" has nothing
    # to be 15% of, so no claim is made rather than an arbitrary one.
    width = _range_width(range_bounds)
    if width is None:
        return False
    low, high = range_bounds
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
        base += f" The normal range for this test is {_range_phrase(range_bounds, unit)}."

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


def _visits_newest_first(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Visits most recent first, undated ones last. Demographics are read off
    the newest document that states them, so a document with no date must not
    outrank a dated one just because of where it landed in the list."""
    visits = timeline.get("visits") or []
    dated = [(v, _parse_date(v.get("date"))) for v in visits]
    with_dates = sorted(
        [(v, d) for v, d in dated if d is not None], key=lambda p: p[1], reverse=True
    )
    return [v for v, _ in with_dates] + [v for v, d in dated if d is None]


def extract_patient_demographics(
    timeline: Dict[str, Any],
) -> Tuple[Optional[str], Optional[float], str]:
    """The patient's sex and CURRENT age, read off the documents on file.

    Both fields are already extracted per document (`patient_gender` and
    `patient_age` in medical_extractor's schema), so this only has to pick
    which document to believe: the most recent one that states each.

    Age is carried forward to today via the birth year rather than used as
    printed. A report from 2019 saying "Age: 45" describes a 52-year-old now,
    and reference_intervals bands several tests by age — reading the printed
    number straight off an old report would pick a band off a stale figure.
    identity_guard.infer_birth_year() already does exactly this arithmetic for
    identity matching, so it is reused rather than repeated.

    Returns (sex, age, source) — source describing what was found, for logging
    and so callers can say plainly when nothing was. Either value may be None,
    which reference_intervals treats as a refusal to guess rather than a
    licence to fall back on the other sex's interval.
    """
    visits = _visits_newest_first(timeline)

    sex: Optional[str] = None
    for visit in visits:
        gender = visit.get("patient_gender")
        if gender in ("male", "female"):
            sex = gender
            break

    age: Optional[float] = None
    for visit in visits:
        raw_age = visit.get("patient_age")
        if raw_age is None or not isinstance(raw_age, (int, float)):
            continue
        birth_year = infer_birth_year(raw_age, visit.get("date"))
        if birth_year is not None:
            age = float(datetime.now().year - birth_year)
        else:
            # No usable document date to age the number forward from, so the
            # printed age is the best available and stands as-is.
            age = float(raw_age)
        break
    if age is not None and not (0 <= age <= 130):
        age = None

    if sex and age is not None:
        source = f"from documents on file (sex: {sex}, age: {age:g})"
    elif sex:
        source = f"from documents on file (sex: {sex}, age not stated)"
    elif age is not None:
        source = f"from documents on file (age: {age:g}, sex not stated)"
    else:
        source = "not stated on any document on file"
    return sex, age, source


def _demographic_interval(
    test_id: Optional[str],
    value: float,
    unit: Optional[str],
    sex: Optional[str],
    age: Optional[float],
) -> Optional[Dict[str, Any]]:
    """The general interval for this test at this patient's sex and age, with
    the value converted into the interval's unit.

    Returns None — meaning "no status for this one" — if the test isn't in the
    table, the reported unit can't be reconciled with the table's, or the
    matching rule needs a demographic that wasn't stated. All three are cases
    where the honest output is the raw value and nothing more.
    """
    if not test_id:
        return None
    converted = to_canonical_value(test_id, value, unit)
    if converted is None:
        return None
    interval = lookup_interval(test_id, sex, age)
    if interval is None:
        return None
    canonical_value, canonical_unit = converted
    return {
        "value": canonical_value,
        "bounds": (interval["low"], interval["high"]),
        "unit": canonical_unit,
        "basis": interval["basis"],
        "source": interval["source"],
        "unit_assumed": not (unit or "").strip(),
    }


def _demographic_phrase(
    sex: Optional[str], age: Optional[float], basis: Dict[str, bool]
) -> str:
    """Names the demographics the matched interval ACTUALLY depended on.
    Saying "for a woman aged 34" about an interval that is the same for
    everyone would imply a precision the table doesn't have."""
    who = None
    if basis.get("sex_specific") and sex in ("male", "female"):
        who = "a man" if sex == "male" else "a woman"
    if basis.get("age_specific") and age is not None:
        who = f"{who} aged {age:g}" if who else f"someone aged {age:g}"
    return f"for {who}" if who else ""


def _explain_single(
    test_name: str,
    point: Dict[str, Any],
    status: str,
    bounds: Optional[Bounds],
    range_unit: str,
    range_source: Optional[str],
    demographic_note: str,
) -> str:
    """The one-reading counterpart to _explain(): same rule, that every
    sentence is assembled from a computed value and none of them interpret it.

    It states what was measured, when, what it was compared against, and which
    side of that it fell on — then says outright that there is nothing to
    compare it with yet, because a single reading is the case where a reader is
    most likely to assume more was known than actually was.
    """
    # The value EXACTLY as the report printed it. Re-rendering the parsed
    # float instead would quietly rewrite the reader's own result: a pus cell
    # count printed "30 - 40 /hpf" came back as "30 /hpf", and a colony count
    # printed "> 100,000 CFU/mL" lost both the ">" and three orders of
    # magnitude. The parsed number is for comparing; this is for showing.
    raw_value = point.get("value")
    value_text = (
        f"{raw_value} {point.get('unit') or ''}".strip()
        if raw_value not in (None, "")
        else f"{point['_value']:g} {point.get('unit') or ''}".strip()
    )
    date_text = point.get("date") or "an unspecified date"

    plain_name, measures = _describe_test(test_name)
    opening = f"{test_name} is {measures}." if measures else ""

    base = (
        f"{opening} This test appears once in your records: {value_text} on {date_text}."
    ).strip()

    if range_source == "report":
        base += (
            f" The normal range printed on the report is "
            f"{_range_phrase(bounds, range_unit)}."
        )
    elif range_source == "general":
        who = f" {demographic_note}" if demographic_note else ""
        base += (
            f" The report did not print a normal range for this test, so it has been "
            f"compared against a general normal range{who}: "
            f"{_range_phrase(bounds, range_unit)}. The laboratory that ran your test may "
            f"use a slightly different one."
        )
    else:
        base += (
            " No normal range was printed on the report and none could be looked up "
            "for this test, so this shows the measured value only — there is nothing "
            "here saying whether it is inside or outside a normal range."
        )

    if bounds:
        if status == "normal":
            base += " The result is inside that range."
        elif status in ("high", "low"):
            base += f" The result is {_plain_flag(status)}."
    elif status in ("high", "low", "normal"):
        # There is no range here, so the status can only have come from a
        # marking the report printed beside the value. Saying so is the only
        # honest phrasing: "the result is inside that range" pointed at a
        # range the preceding sentence had just said did not exist.
        base += f" The report itself marked this result as {status}."

    base += (
        " This is the only result on file for this test, so there is nothing earlier "
        "to compare it with — if it is tested again, the two results together will "
        "show whether it is changing."
    )

    # Same rule _explain() follows: the caution rides on the results someone
    # might read too much into, not on every line. A single reading judged
    # against a GENERAL range earns it too, even when normal, because the
    # range itself is an approximation of their lab's.
    if status in ("high", "low") or range_source == "general":
        base += (
            " This only shows what was measured and when — not what caused it or what "
            "it means. A doctor or pharmacist can explain that."
        )
    return base


def _assess_single_result(
    point: Dict[str, Any],
    display_name: str,
    test_id: Optional[str],
    test_names: List[str],
    sex: Optional[str],
    age: Optional[float],
    dropped: int,
) -> Dict[str, Any]:
    """Interprets one reading that has no earlier result to trend against.

    Range precedence is the whole point of this function, and it runs printed
    range first, general interval second, nothing third — see the module
    docstring for why the lab's own paper outranks the table.
    """
    printed_bounds = _parse_range(point.get("reference_range"))
    demographic = None
    unit_assumed = False

    # Readings reported as a span keep both ends for the comparison; a plain
    # number becomes a zero-width span so one code path handles both.
    span = _parse_value_span(point.get("value")) or (point["_value"], point["_value"])

    if printed_bounds:
        bounds: Optional[Bounds] = printed_bounds
        compare_span = span
        range_unit = point.get("unit") or ""
        range_source: Optional[str] = "report"
    else:
        demographic = _demographic_interval(
            test_id, point["_value"], point.get("unit"), sex, age
        )
        if demographic:
            bounds = demographic["bounds"]
            range_unit = demographic["unit"]
            range_source = "general"
            unit_assumed = demographic["unit_assumed"]
            # The interval is in the table's unit, so both ends of the span
            # have to cross into it too — comparing a converted bound against
            # an unconverted reading is the exact error the unit table exists
            # to prevent.
            converted_span = [
                to_canonical_value(test_id, end, point.get("unit")) for end in span
            ]
            compare_span = (
                (converted_span[0][0], converted_span[1][0])
                if all(c is not None for c in converted_span)
                else (demographic["value"], demographic["value"])
            )
        else:
            bounds, compare_span, range_source = None, span, None
            range_unit = point.get("unit") or ""

    # The report's own H/L marking wins over anything computed here: the lab
    # marked it against the exact interval it ran the assay on. Only when the
    # extractor found no marking is the status worked out arithmetically.
    reported_flag = point.get("reported_flag")
    if reported_flag in ("high", "low", "normal"):
        status = reported_flag
    else:
        status = _classify_span(compare_span, bounds)

    confidence = point.get("confidence")
    confidence = confidence if isinstance(confidence, (int, float)) else 0.7
    if range_source == "general":
        # A general interval is a stand-in for their lab's, and says so.
        confidence *= 0.85
    if unit_assumed:
        confidence *= 0.9
    if dropped:
        confidence *= max(0.5, 1 - 0.15 * dropped)

    plain_name, measures = _describe_test(display_name)
    demographic_note = (
        _demographic_phrase(sex, age, demographic["basis"]) if demographic else ""
    )

    return {
        "test_name": display_name,
        "test_id": test_id,
        # Every spelling this test was printed under, so a caller holding raw
        # readings can match each one back to this assessment without
        # reimplementing the synonym table.
        "test_names": test_names,
        "plain_name": plain_name,
        "what_it_measures": measures,
        "is_main_test": is_main_test(display_name),
        "date": point.get("date"),
        "value": point.get("value"),
        "unit": point.get("unit") or "",
        "source_file": point.get("source_file"),
        "reference_range": point.get("reference_range"),
        "status": status,
        # Which range produced `status`: "report" (printed on their own
        # document), "general" (reference_intervals.py), or None (no status).
        "range_source": range_source,
        "range_used": (
            {"low": bounds[0], "high": bounds[1], "unit": range_unit} if bounds else None
        ),
        "compared_against": demographic_note or None,
        "confidence": round(min(confidence, 0.97), 2),
        "explanation": _explain_single(
            display_name, point, status, bounds, range_unit, range_source, demographic_note
        ),
    }


def track_lab_trends(
    timeline: Dict[str, Any],
    patient_sex: Optional[str] = None,
    patient_age: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Groups `timeline["lab_results_timeline"]` by test and analyzes each test
    with 2+ usable (dated, numeric) data points for directional drift and
    reference-range crossings. Tests with exactly ONE usable reading get a
    low/normal/high status instead, in `single_results` — see the module
    docstring for the range precedence that produces it.

    `patient_sex` / `patient_age` are only needed to override what the
    documents say; left as None they are read off the timeline itself by
    extract_patient_demographics(), so every caller gets the demographic
    intervals without having to plumb them through.

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
        "single_results": [
          {
            "test_name": str, "test_id": str | None,
            "plain_name": str | None, "what_it_measures": str | None,
            "is_main_test": bool,   # one of the tests surfaced without asking
            "date", "value", "unit", "source_file", "reference_range",
            "status": "low" | "normal" | "high" | "unknown",
            "range_source": "report" | "general" | None,
            "range_used": {"low": float|None, "high": float|None, "unit": str} | None,
            "compared_against": str | None,  # e.g. "for a woman aged 34"
            "confidence": float,
            "explanation": str,
          }, ...
        ],
        "patient_context": {"sex":..., "age":..., "source":...},
        "insufficient_data": [{"test_name": str, "reason": str}, ...],
        "note": "... not a diagnosis, consult a clinician/pharmacist ..."
      }
    """
    lab_results_timeline = timeline.get("lab_results_timeline", [])
    grouped = _group_by_test(lab_results_timeline)

    derived_sex, derived_age, demographics_source = extract_patient_demographics(timeline)
    sex = patient_sex if patient_sex is not None else derived_sex
    age = patient_age if patient_age is not None else derived_age

    trends: List[Dict[str, Any]] = []
    single_results: List[Dict[str, Any]] = []
    insufficient: List[Dict[str, Any]] = []

    for group in grouped.values():
        test_name = group["display_name"]
        test_id = group["test_id"]
        test_names = group["names"]
        entries = group["entries"]
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
            # The extractor only records a flag when the report PRINTED one
            # (an H/L marker beside the value). Plenty of reports print a
            # reference range and leave the reader to do the comparison, and
            # an unflagged value used to make a crossing undetectable even
            # when the number was plainly outside the printed range — so it
            # is computed here when, and only when, the report gave no flag.
            reported_flag = (e.get("flag") or "unknown").lower()
            if reported_flag not in ("high", "low", "normal"):
                reported_flag = None

            flag = reported_flag or "unknown"
            if reported_flag is None:
                value_span = _parse_value_span(e.get("value")) or (val, val)
                point_bounds = _parse_range(e.get("reference_range"))
                if point_bounds:
                    flag = _classify_span(value_span, point_bounds)
                else:
                    demographic = _demographic_interval(
                        test_id, val, e.get("unit"), sex, age
                    )
                    if demographic:
                        flag = _classify(demographic["value"], demographic["bounds"])

            usable.append({
                "_dt": dt, "_value": val,
                "date": e.get("date"), "value": e.get("value"),
                "flag": flag,
                # Kept apart from `flag` above, which may be the value this
                # module computed. Downstream, "the report marked this high"
                # outranks a computed status — but only when the report
                # really did mark it, and collapsing the two into one field
                # let a computed status masquerade as the lab's own.
                "reported_flag": reported_flag,
                "unit": e.get("unit") or "",
                "reference_range": e.get("reference_range"),
                "source_file": e.get("source_file"),
                "confidence": e.get("confidence", 1.0),
            })
            if e.get("unit"):
                units_seen.add(e["unit"])
            if e.get("reference_range"):
                ranges_seen.add(e["reference_range"])

        if len(usable) == 1:
            # One reading is not a trend, but it IS the result a patient most
            # wants explained. It gets a status instead of being written off
            # as insufficient data and reaching them as a bare number.
            single_results.append(
                _assess_single_result(
                    usable[0], test_name, test_id, test_names, sex, age, dropped
                )
            )
            continue

        if not usable:
            insufficient.append({
                "test_name": test_name,
                "reason": (
                    f"no entries had both a parseable date and a numeric value "
                    f"({dropped} dropped)."
                ),
            })
            continue

        usable.sort(key=lambda p: p["_dt"])

        unit = usable[-1]["unit"]
        # Trends still read the range off the patient's own report only. A
        # general interval is a reasonable stand-in for judging one value, but
        # anchoring a multi-visit crossing to it would let the table, rather
        # than the lab, decide the date a result "went abnormal".
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
            # Canonical id, so a caller can match this trend to a reading
            # whose report spelled the test differently. Null for a test
            # outside reference_intervals' table.
            "test_id": test_id,
            # Every spelling this test was printed under across the documents.
            "test_names": test_names,
            # Both null for a test not in TEST_GLOSSARY — a caller rendering
            # these should fall back to test_name rather than show "None".
            "plain_name": plain_name,
            "what_it_measures": measures,
            "is_main_test": is_main_test(test_name),
            "unit": unit,
            "reference_range": usable[-1]["reference_range"],
            "data_points": [
                {
                    "date": p["date"],
                    # Normalised alongside the printed date, for callers that
                    # need to place these on a time axis. The printed form is
                    # ambiguous across locales — "19/12/2023" is parsed as a
                    # date in December here and as an invalid month by
                    # JavaScript's Date, which would silently mis-plot it — so
                    # the parse is done once, here, where dateutil already
                    # resolved it, rather than repeated by every consumer.
                    "date_iso": p["_dt"].date().isoformat(),
                    "value": p["value"],
                    # The parsed number behind the printed value, so a chart
                    # does not have to re-extract it from "> 100,000".
                    "value_numeric": p["_value"],
                    "flag": p["flag"],
                    "source_file": p["source_file"],
                }
                for p in usable
            ],
            # The parsed reference bounds, for drawing the normal band. None
            # on either side for a one-sided range, None entirely when the
            # report printed no parseable range.
            "range_used": (
                {"low": range_bounds[0], "high": range_bounds[1], "unit": unit}
                if range_bounds else None
            ),
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
        "single_results": single_results,
        # What the demographic intervals were chosen for, so a caller can show
        # it and a reader can tell us we have the wrong age or sex on file.
        "patient_context": {"sex": sex, "age": age, "source": demographics_source},
        "insufficient_data": insufficient,
        "note": (
            "This is worked out directly from the numbers printed on the uploaded lab "
            "reports and the normal ranges printed alongside them. Where a report did not "
            "print a normal range, a general reference range for your age and sex was used "
            "instead and is labelled as such — your own laboratory may use a slightly "
            "different one. It shows how results have changed over time, or where a single "
            "result sits — it does not say what caused a change, and it is not a diagnosis. "
            "It also cannot see anything your reports don't show, such as how you are "
            "feeling or any other health condition. A doctor or pharmacist can explain what "
            "these results mean for you."
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

    # -----------------------------------------------------------------
    # One-sided reference ranges. "<200" and ">60" are how labs actually
    # print lipid and eGFR ranges; these used to parse to None, which threw
    # away the bound and left the test with no range sentence at all.
    # -----------------------------------------------------------------
    assert _parse_range("70-99") == (70.0, 99.0)
    assert _parse_range("0.74 - 1.35") == (0.74, 1.35)
    assert _parse_range("70 to 99") == (70.0, 99.0)
    assert _parse_range("4.0-11.0 x10^9/L") == (4.0, 11.0)
    assert _parse_range("<200") == (None, 200.0)
    assert _parse_range("<= 5.7") == (None, 5.7)
    assert _parse_range("up to 150") == (None, 150.0)
    assert _parse_range(">60") == (60.0, None)
    assert _parse_range(">= 40") == (40.0, None)
    assert _parse_range("Negative") is None
    # A one-sided range has no width, so nothing may be claimed about a value
    # "approaching" a boundary of it.
    assert _range_width((None, 200.0)) is None
    assert _approaching_boundary(190.0, "normal", (None, 200.0), "increasing") is False

    # -----------------------------------------------------------------
    # Demographics: current age, not the age printed on an old report.
    # -----------------------------------------------------------------
    demo_sex, demo_age, demo_source = extract_patient_demographics({
        "visits": [
            {"date": "2019-06-01", "patient_age": 40, "patient_gender": "female"},
            {"date": "2024-06-01", "patient_age": 45, "patient_gender": "female"},
        ]
    })
    assert demo_sex == "female"
    expected_age = datetime.now().year - 2024 + 45
    assert demo_age == expected_age, (demo_age, expected_age)
    # An undated document must not outrank a dated one.
    assert extract_patient_demographics({
        "visits": [
            {"date": None, "patient_gender": "male"},
            {"date": "2024-06-01", "patient_gender": "female"},
        ]
    })[0] == "female"
    assert extract_patient_demographics({"visits": []}) == (None, None,
                                                            "not stated on any document on file")

    # -----------------------------------------------------------------
    # Single reading, no printed range -> general interval chosen for sex/age.
    # Hemoglobin 11.2 is normal for a man and low for a woman, which is the
    # whole reason the interval has to be sex-specific.
    # -----------------------------------------------------------------
    single_timeline = {
        "visits": [{"date": "2026-01-05", "patient_age": 34, "patient_gender": "female"}],
        "lab_results_timeline": [
            {"test_name": "Hemoglobin", "value": "11.2", "unit": "g/dL",
             "reference_range": None, "flag": None, "confidence": 0.95,
             "date": "05 Jan 2026", "source_file": "Report.pdf"},
            {"test_name": "Platelet Count", "value": "310", "unit": "x10^9/L",
             "reference_range": None, "flag": None, "confidence": 0.95,
             "date": "05 Jan 2026", "source_file": "Report.pdf"},
            {"test_name": "Some Unlisted Assay", "value": "7.4", "unit": "arb",
             "reference_range": None, "flag": None, "confidence": 0.9,
             "date": "05 Jan 2026", "source_file": "Report.pdf"},
        ],
    }
    single = track_lab_trends(single_timeline)
    by_single = {s["test_name"]: s for s in single["single_results"]}
    assert single["trends"] == []
    assert single["patient_context"]["sex"] == "female"

    hb = by_single["Hemoglobin"]
    assert hb["status"] == "low", hb
    assert hb["range_source"] == "general"
    assert hb["range_used"] == {"low": 12.0, "high": 15.5, "unit": "g/dL"}
    # The adult hemoglobin rule is banded on BOTH sex and age (18+), so both
    # get named. A rule that depended on neither must say neither — see the
    # platelet assertion below.
    assert hb["compared_against"] == "for a woman aged 34", hb["compared_against"]
    assert hb["is_main_test"] is True
    # The reader must be told this range was not their lab's.
    assert "general normal range" in hb["explanation"], hb["explanation"]
    assert "lower than the normal range" in hb["explanation"], hb["explanation"]
    assert "nothing earlier to compare it with" in hb["explanation"]

    # The same hemoglobin number reaching a DIFFERENT verdict by sex is the
    # whole reason the interval is banded: 12.5 g/dL sits inside the female
    # interval (12.0-15.5) and below the male one (13.5-17.5). If sex were
    # being ignored, both of these would agree.
    def _hb_status(sex_arg: str) -> str:
        return track_lab_trends(
            {
                "visits": [],
                "lab_results_timeline": [
                    {"test_name": "Hemoglobin", "value": "12.5", "unit": "g/dL",
                     "reference_range": None, "flag": None, "confidence": 0.95,
                     "date": "05 Jan 2026", "source_file": "Report.pdf"},
                ],
            },
            patient_sex=sex_arg, patient_age=34,
        )["single_results"][0]["status"]

    assert _hb_status("female") == "normal", _hb_status("female")
    assert _hb_status("male") == "low", _hb_status("male")

    # An age-independent interval must not claim to be age-specific.
    plt_single = by_single["Platelet Count"]
    assert plt_single["status"] == "normal"
    assert plt_single["compared_against"] is None, plt_single["compared_against"]

    # A test outside the table gets no invented status.
    unlisted = by_single["Some Unlisted Assay"]
    assert unlisted["status"] == "unknown", unlisted
    assert unlisted["range_source"] is None
    assert unlisted["range_used"] is None
    assert unlisted["is_main_test"] is False
    assert "shows the measured value only" in unlisted["explanation"]

    # With no demographics on file, a sex-specific interval must be declined
    # rather than resolved to one sex or the other.
    anonymous = track_lab_trends({
        "visits": [],
        "lab_results_timeline": [single_timeline["lab_results_timeline"][0]],
    })
    assert anonymous["single_results"][0]["status"] == "unknown"
    assert anonymous["single_results"][0]["range_source"] is None

    # A printed range always outranks the table: this report prints 13-17 for
    # a female patient, so 11.2 is judged against 13-17, not against 12-15.5.
    printed_wins = track_lab_trends({
        "visits": [{"date": "2026-01-05", "patient_age": 34, "patient_gender": "female"}],
        "lab_results_timeline": [
            {"test_name": "Hemoglobin", "value": "11.2", "unit": "g/dL",
             "reference_range": "13-17", "flag": None, "confidence": 0.95,
             "date": "05 Jan 2026", "source_file": "Report.pdf"},
        ],
    })["single_results"][0]
    assert printed_wins["range_source"] == "report"
    assert printed_wins["range_used"] == {"low": 13.0, "high": 17.0, "unit": "g/dL"}
    assert "printed on the report" in printed_wins["explanation"]

    # Unit conversion: 112 g/L is the same 11.2 g/dL, and must reach the same
    # verdict rather than being compared as the number 112.
    converted = track_lab_trends({
        "visits": [{"date": "2026-01-05", "patient_age": 34, "patient_gender": "female"}],
        "lab_results_timeline": [
            {"test_name": "Hemoglobin", "value": "112", "unit": "g/L",
             "reference_range": None, "flag": None, "confidence": 0.95,
             "date": "05 Jan 2026", "source_file": "Report.pdf"},
        ],
    })["single_results"][0]
    assert converted["status"] == "low", converted
    # An unrecognised unit must refuse rather than compare raw numbers.
    unknown_unit = track_lab_trends({
        "visits": [{"date": "2026-01-05", "patient_age": 34, "patient_gender": "female"}],
        "lab_results_timeline": [
            {"test_name": "Hemoglobin", "value": "11.2", "unit": "furlongs",
             "reference_range": None, "flag": None, "confidence": 0.95,
             "date": "05 Jan 2026", "source_file": "Report.pdf"},
        ],
    })["single_results"][0]
    assert unknown_unit["status"] == "unknown", unknown_unit

    # -----------------------------------------------------------------
    # Two readings printed under DIFFERENT names for the same test must group
    # into one trend, not two orphaned single results.
    # -----------------------------------------------------------------
    synonym_result = track_lab_trends({
        "visits": [{"date": "2026-01-05", "patient_age": 50, "patient_gender": "male"}],
        "lab_results_timeline": [
            {"test_name": "Fasting Glucose", "value": "91", "unit": "mg/dL",
             "reference_range": "70-99", "flag": "normal", "confidence": 0.95,
             "date": "05 Jan 2026", "source_file": "a.pdf"},
            {"test_name": "FBS", "value": "118", "unit": "mg/dL",
             "reference_range": "70-99", "flag": "high", "confidence": 0.95,
             "date": "30 Aug 2026", "source_file": "b.pdf"},
        ],
    })
    assert len(synonym_result["trends"]) == 1, synonym_result["trends"]
    assert synonym_result["single_results"] == []
    assert synonym_result["trends"][0]["direction"] == "increasing"
    assert synonym_result["trends"][0]["test_id"] == "glucose_fasting"

    # -----------------------------------------------------------------
    # An unflagged reading outside its printed range must still register the
    # crossing — the report printed the range and left the comparison to the
    # reader, which used to make the crossing invisible.
    # -----------------------------------------------------------------
    unflagged = track_lab_trends({
        "visits": [],
        "lab_results_timeline": [
            {"test_name": "ALT", "value": "24", "unit": "U/L", "reference_range": "7-56",
             "flag": None, "confidence": 0.95, "date": "05 Jan 2026", "source_file": "a.pdf"},
            {"test_name": "ALT", "value": "82", "unit": "U/L", "reference_range": "7-56",
             "flag": None, "confidence": 0.95, "date": "30 Aug 2026", "source_file": "b.pdf"},
        ],
    })["trends"][0]
    assert unflagged["flag_sequence"] == "normal → high", unflagged["flag_sequence"]
    assert unflagged["crossed_into_abnormal_at"]["date"] == "30 Aug 2026"

    # Plain-language guarantee holds for the single-result explanations too.
    for single_result in single["single_results"] + [printed_wins, converted]:
        text = single_result["explanation"]
        for jargon in ("reference range", "abnormal", "boundary", "flagged", "drift", "→"):
            assert jargon not in text, f"{single_result['test_name']}: leaked '{jargon}' -> {text}"

    # -----------------------------------------------------------------
    # Regression: a real urine culture + microscopy report, which broke four
    # separate assumptions this module had made about how values are printed.
    # -----------------------------------------------------------------
    # 1. Digit grouping. "> 100,000" parsed to 100 and "<10,000" to 10 —
    #    both a thousandth of the real figure.
    assert _parse_value("> 100,000") == 100000.0
    assert _parse_range("<10,000") == (None, 10000.0)
    # ...without breaking a decimal comma, which is how much of the world
    # writes 9.2 and which must NOT become 92.
    assert _parse_value("9,2") == 9.0

    # 2. Values printed as a SPAN. Microscopy counts a range across fields,
    #    so "30 - 40 /hpf" is the whole reading, not 30.
    assert _parse_value_span("30 - 40") == (30.0, 40.0)
    assert _parse_value_span("3 - 5") == (3.0, 5.0)
    assert _parse_value_span("6.5") == (6.5, 6.5)
    assert _parse_value_span("> 100,000") == (100000.0, 100000.0)
    assert _parse_value_span("SENSITIVE") is None
    # A span whose TOP end clears the range is above the range. Classifying
    # on its bottom end alone called 3-5 red cells "normal" against "<3".
    assert _classify_span((3.0, 5.0), (None, 3.0)) == "high"
    assert _classify_span((2.0, 4.0), (None, 5.0)) == "normal"

    urine = track_lab_trends({
        "visits": [{"date": "2023-12-19", "patient_age": 49, "patient_gender": "female"}],
        "lab_results_timeline": [
            {"test_name": "Colony Count", "value": "> 100,000", "unit": "CFU/mL",
             "reference_range": "<10,000", "flag": None, "confidence": 0.9,
             "date": "2023-09-11", "source_file": "u.pdf"},
            {"test_name": "Pus Cells", "value": "30 - 40", "unit": "/hpf",
             "reference_range": "<5", "flag": None, "confidence": 0.9,
             "date": "2023-09-11", "source_file": "u.pdf"},
            {"test_name": "Red Cells", "value": "3 - 5", "unit": "/hpf",
             "reference_range": "<3", "flag": None, "confidence": 0.9,
             "date": "2023-09-11", "source_file": "u.pdf"},
            {"test_name": "Epithelial Cells", "value": "2 - 4", "unit": "/hpf",
             "reference_range": None, "flag": "normal", "confidence": 0.9,
             "date": "2023-09-11", "source_file": "u.pdf"},
        ],
    })
    by_urine = {s["test_name"]: s for s in urine["single_results"]}

    colony = by_urine["Colony Count"]
    assert colony["status"] == "high"
    # 3. The value is shown as PRINTED. Re-rendering the parsed float dropped
    #    the ">" and the magnitude, handing the reader a different result
    #    from the one on their report.
    assert "> 100,000 CFU/mL" in colony["explanation"], colony["explanation"]
    assert "anything below 10,000 CFU/mL" in colony["explanation"], colony["explanation"]

    assert "30 - 40 /hpf" in by_urine["Pus Cells"]["explanation"]
    assert by_urine["Pus Cells"]["status"] == "high"
    assert by_urine["Red Cells"]["status"] == "high", by_urine["Red Cells"]
    assert "3 - 5 /hpf" in by_urine["Red Cells"]["explanation"]

    # 4. A status with no range behind it must not be phrased as a comparison
    #    against one. This said "there is nothing here saying whether it is
    #    inside or outside a normal range" and then, in the next sentence,
    #    "the result is inside that range".
    epithelial = by_urine["Epithelial Cells"]
    assert epithelial["status"] == "normal"
    assert epithelial["range_source"] is None
    assert "inside that range" not in epithelial["explanation"], epithelial["explanation"]
    assert "The report itself marked this result as normal." in epithelial["explanation"]

    # A flag this module COMPUTED must never be mistaken for one the lab
    # printed — that is what let the wrong "normal" above win over the
    # span-aware comparison that would have caught it.
    assert by_urine["Red Cells"]["status"] == "high"
    assert by_urine["Pus Cells"]["explanation"].count("marked this result") == 0

    for t in result["trends"]:
        print(f"--- {t['test_name']} ---")
        print(" direction:", t["direction"], "| flags:", t["flag_sequence"], "| confidence:", t["confidence"])
        print(" ", t["explanation"])
        print()

    for s in single["single_results"]:
        print(f"--- {s['test_name']} (single reading) ---")
        print(" status:", s["status"], "| range from:", s["range_source"],
              "| confidence:", s["confidence"])
        print(" ", s["explanation"])
        print()

    print("All checks passed.")
