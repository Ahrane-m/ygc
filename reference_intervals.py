"""
Reference Intervals for Common Lab Tests
=========================================
A lookup table of typical reference intervals for the lab tests a patient
most often wants explained, banded by sex and age where that genuinely
changes the interval (hemoglobin, creatinine, ferritin, uric acid, the
transaminases). Used by lab_trends.py to say whether a SINGLE reading --
one with no earlier result to trend against -- sits low, normal or high.

WHY THIS TABLE IS THE FALLBACK, NEVER THE OVERRIDE
--------------------------------------------------
Reference intervals are a property of the assay and the instrument, not of
medicine in general: two labs measuring the same analyte can legitimately
print different ranges. The range printed on the patient's own report is
therefore always more authoritative than anything here, and lab_trends.py
consults this table ONLY when the report printed no range, or printed one
that could not be parsed. Every result interpreted from this table is
labelled as using a general reference interval so nobody reads it as their
own lab's verdict.

WHAT A LOOKUP RESULT MAY AND MAY NOT SAY
----------------------------------------
"Low", "normal" or "high" is a comparison between two numbers, and that is
the entire claim being made. It is not a finding, a cause, or a condition.
The same rule TEST_GLOSSARY in lab_trends.py already follows applies here:
this module describes where a value sits relative to an interval, and never
what sitting there means for the person. That stays a clinician's call.

WHEN NO INTERVAL IS RETURNED
----------------------------
lookup_interval() returns None -- and the caller shows the value with no
status -- whenever:
  - the test is not in this table,
  - the reported unit is not one this module knows how to reconcile with the
    table's unit (a wrong conversion is far worse than no answer),
  - the patient's sex or age is unknown and the matching rule needs it.
Guessing a demographic to force an interval out of this table would produce
a confident status derived from an assumption, which is the one failure
mode this design exists to avoid.

UNITS
-----
Each test declares one canonical unit and the aliases it accepts, with the
factor converting a reported value INTO the canonical unit. Factors are
per-test because molar conversions depend on molecular weight -- glucose
mmol/L -> mg/dL is x18.02, cholesterol is x38.67, triglycerides is x88.57.
A unit not listed for that test is refused rather than assumed.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Test name -> canonical id
#
# Lab reports print the same analyte a dozen ways ("Fasting Glucose", "FBS",
# "Glucose, Fasting", "Blood Sugar - Fasting"). Collapsing them to one id is
# what lets a value uploaded today be compared against the "same" test from a
# report six months ago, and what lets this table be keyed once per analyte.
#
# Matched on WORD BOUNDARIES, longest key first, so "hba1c" wins over "a1c"
# and a short key like "alt" never fires inside "Cobalt" or "Alkaline".
# ---------------------------------------------------------------------------

TEST_SYNONYMS: List[Tuple[str, Tuple[str, ...]]] = [
    ("hba1c", ("hba1c", "hb a1c", "a1c", "glycated hemoglobin", "glycosylated hemoglobin")),
    ("glucose_fasting", ("fasting glucose", "glucose fasting", "fbs", "fbg",
                         "fasting blood sugar", "fasting blood glucose")),
    ("glucose_random", ("random glucose", "rbs", "random blood sugar", "glucose random")),
    # Bare "glucose"/"blood sugar" with no fasting/random qualifier. Listed
    # AFTER the qualified forms so those win when both could match.
    ("glucose_fasting", ("glucose", "blood sugar")),
    ("hemoglobin", ("hemoglobin", "haemoglobin", "hgb", "hb")),
    ("wbc", ("wbc", "white blood cell", "white blood cells", "leukocyte",
             "leucocyte", "total leukocyte count", "tlc")),
    ("platelets", ("platelet", "platelets", "platelet count", "plt")),
    ("creatinine", ("creatinine", "serum creatinine")),
    ("egfr", ("egfr", "gfr", "estimated gfr")),
    ("alt", ("alt", "sgpt", "alanine aminotransferase", "alanine transaminase")),
    ("ast", ("ast", "sgot", "aspartate aminotransferase", "aspartate transaminase")),
    # ORDER IS LOad-BEARING FOR THE LIPID PANEL. Every one of these names
    # contains the word "cholesterol", so a bare "cholesterol" synonym listed
    # first swallows the whole panel: Total Cholesterol 232, HDL 38, LDL 157
    # and a ratio of 6.1 all collapsed into one "test" that then reported a
    # fluctuating trend across four values that were never the same analyte.
    # The specific names must therefore be matched BEFORE the generic one,
    # and bare "cholesterol" stays last.
    ("non_hdl_cholesterol", ("non-hdl cholesterol", "non hdl cholesterol",
                             "non-hdl-c", "non-hdl")),
    ("ldl", ("ldl", "ldl cholesterol", "ldl-c", "low density lipoprotein")),
    ("hdl", ("hdl", "hdl cholesterol", "hdl-c", "high density lipoprotein")),
    ("triglycerides", ("triglycerides", "triglyceride", "tg")),
    ("cholesterol_total", ("total cholesterol", "cholesterol total", "cholesterol")),
    ("tsh", ("tsh", "thyroid stimulating hormone", "thyrotropin")),
    ("sodium", ("sodium", "na+", "serum sodium")),
    ("potassium", ("potassium", "k+", "serum potassium")),
    ("ferritin", ("ferritin", "serum ferritin")),
    ("vitamin_d", ("vitamin d", "25-oh vitamin d", "25 oh vitamin d",
                   "25-hydroxyvitamin d", "vit d")),
    ("vitamin_b12", ("vitamin b12", "b12", "cobalamin", "vit b12")),
    ("uric_acid", ("uric acid", "urate", "serum uric acid")),
    ("crp", ("crp", "c-reactive protein", "c reactive protein")),
]

# The tests a patient is shown up front. Everything else on a panel is still
# stored, still returned, and still available behind "show all results" -- this
# list only decides what surfaces without being asked for. Kept to the tests a
# non-clinical reader is likely to have heard of and to have a question about.
MAIN_TESTS = {
    "hemoglobin", "wbc", "platelets", "glucose_fasting", "hba1c",
    "creatinine", "egfr", "alt", "ast", "cholesterol_total", "ldl", "hdl",
    "triglycerides", "tsh", "sodium", "potassium", "ferritin", "vitamin_d",
    "vitamin_b12", "uric_acid", "crp",
}

# ---------------------------------------------------------------------------
# The intervals themselves
#
# Each rule matches on sex ("male"/"female"/"any") and an inclusive age band.
# `low`/`high` of None means the interval is open on that side: total
# cholesterol has a desirable upper limit and no meaningful lower one, eGFR
# the reverse. Rules are scanned in order and the first match wins, so the
# more specific band goes first.
#
# Pediatric bands are deliberately sparse. They exist only for the three
# counts whose childhood intervals are stable and widely agreed; for every
# other test a rule starts at 18 and a younger patient simply gets no
# interval rather than an adult one applied to a child.
# ---------------------------------------------------------------------------

REFERENCE_INTERVALS: Dict[str, Dict[str, Any]] = {
    "hemoglobin": {
        "label": "Hemoglobin",
        "unit": "g/dL",
        "units": {"g/dl": 1.0, "gm/dl": 1.0, "g%": 1.0, "g/l": 0.1, "gm%": 1.0},
        "rules": [
            {"sex": "male", "age_min": 18, "age_max": None, "low": 13.5, "high": 17.5},
            {"sex": "female", "age_min": 18, "age_max": None, "low": 12.0, "high": 15.5},
            {"sex": "any", "age_min": 12, "age_max": 17, "low": 11.5, "high": 16.0},
            {"sex": "any", "age_min": 1, "age_max": 11, "low": 11.0, "high": 14.5},
        ],
    },
    "wbc": {
        "label": "White blood cell count",
        "unit": "x10^9/L",
        # 10^9/L, 10^3/uL and K/uL are the same number; /uL and /mm3 are
        # 1000x, hence the 0.001 factor rather than an alias.
        "units": {"x10^9/l": 1.0, "10^9/l": 1.0, "10*9/l": 1.0, "k/ul": 1.0,
                  "10^3/ul": 1.0, "x10^3/ul": 1.0, "thou/ul": 1.0, "10e9/l": 1.0,
                  "/ul": 0.001, "cells/ul": 0.001, "/mm3": 0.001, "cells/mm3": 0.001},
        "rules": [
            {"sex": "any", "age_min": 18, "age_max": None, "low": 4.0, "high": 11.0},
            {"sex": "any", "age_min": 12, "age_max": 17, "low": 4.5, "high": 13.0},
            {"sex": "any", "age_min": 2, "age_max": 11, "low": 5.0, "high": 15.0},
        ],
    },
    "platelets": {
        "label": "Platelet count",
        "unit": "x10^9/L",
        "units": {"x10^9/l": 1.0, "10^9/l": 1.0, "10*9/l": 1.0, "k/ul": 1.0,
                  "10^3/ul": 1.0, "x10^3/ul": 1.0, "thou/ul": 1.0, "10e9/l": 1.0,
                  "/ul": 0.001, "cells/ul": 0.001, "/mm3": 0.001, "cells/mm3": 0.001},
        "rules": [
            # Stable from about a year old onward, so this one rule covers any
            # age -- which also means it still applies when age is unknown.
            {"sex": "any", "age_min": None, "age_max": None, "low": 150.0, "high": 450.0},
        ],
    },
    "glucose_fasting": {
        "label": "Fasting glucose",
        "unit": "mg/dL",
        "units": {"mg/dl": 1.0, "mg%": 1.0, "mmol/l": 18.0182},
        "rules": [
            {"sex": "any", "age_min": 18, "age_max": None, "low": 70.0, "high": 99.0},
        ],
    },
    "glucose_random": {
        "label": "Random glucose",
        "unit": "mg/dL",
        "units": {"mg/dl": 1.0, "mg%": 1.0, "mmol/l": 18.0182},
        "rules": [
            {"sex": "any", "age_min": 18, "age_max": None, "low": 70.0, "high": 140.0},
        ],
    },
    "hba1c": {
        "label": "HbA1c",
        "unit": "%",
        "units": {"%": 1.0, "percent": 1.0},
        "rules": [
            {"sex": "any", "age_min": 18, "age_max": None, "low": 4.0, "high": 5.6},
        ],
    },
    "creatinine": {
        "label": "Creatinine",
        "unit": "mg/dL",
        "units": {"mg/dl": 1.0, "mg%": 1.0, "umol/l": 0.0113, "µmol/l": 0.0113},
        "rules": [
            {"sex": "male", "age_min": 18, "age_max": None, "low": 0.74, "high": 1.35},
            {"sex": "female", "age_min": 18, "age_max": None, "low": 0.59, "high": 1.04},
        ],
    },
    "egfr": {
        "label": "eGFR",
        "unit": "mL/min/1.73m2",
        "units": {"ml/min/1.73m2": 1.0, "ml/min/1.73m^2": 1.0, "ml/min": 1.0},
        "rules": [
            {"sex": "any", "age_min": 18, "age_max": None, "low": 60.0, "high": None},
        ],
    },
    "alt": {
        "label": "ALT",
        "unit": "U/L",
        "units": {"u/l": 1.0, "iu/l": 1.0, "units/l": 1.0},
        "rules": [
            {"sex": "male", "age_min": 18, "age_max": None, "low": 7.0, "high": 55.0},
            {"sex": "female", "age_min": 18, "age_max": None, "low": 7.0, "high": 45.0},
        ],
    },
    "ast": {
        "label": "AST",
        "unit": "U/L",
        "units": {"u/l": 1.0, "iu/l": 1.0, "units/l": 1.0},
        "rules": [
            {"sex": "male", "age_min": 18, "age_max": None, "low": 8.0, "high": 48.0},
            {"sex": "female", "age_min": 18, "age_max": None, "low": 8.0, "high": 43.0},
        ],
    },
    "cholesterol_total": {
        "label": "Total cholesterol",
        "unit": "mg/dL",
        "units": {"mg/dl": 1.0, "mg%": 1.0, "mmol/l": 38.67},
        "rules": [
            {"sex": "any", "age_min": 18, "age_max": None, "low": None, "high": 200.0},
        ],
    },
    "ldl": {
        "label": "LDL cholesterol",
        "unit": "mg/dL",
        "units": {"mg/dl": 1.0, "mg%": 1.0, "mmol/l": 38.67},
        "rules": [
            {"sex": "any", "age_min": 18, "age_max": None, "low": None, "high": 100.0},
        ],
    },
    "hdl": {
        "label": "HDL cholesterol",
        "unit": "mg/dL",
        "units": {"mg/dl": 1.0, "mg%": 1.0, "mmol/l": 38.67},
        "rules": [
            {"sex": "male", "age_min": 18, "age_max": None, "low": 40.0, "high": None},
            {"sex": "female", "age_min": 18, "age_max": None, "low": 50.0, "high": None},
        ],
    },
    "triglycerides": {
        "label": "Triglycerides",
        "unit": "mg/dL",
        "units": {"mg/dl": 1.0, "mg%": 1.0, "mmol/l": 88.57},
        "rules": [
            {"sex": "any", "age_min": 18, "age_max": None, "low": None, "high": 150.0},
        ],
    },
    "tsh": {
        "label": "TSH",
        "unit": "mIU/L",
        "units": {"miu/l": 1.0, "uiu/ml": 1.0, "µiu/ml": 1.0, "mu/l": 1.0},
        "rules": [
            {"sex": "any", "age_min": 18, "age_max": None, "low": 0.4, "high": 4.0},
        ],
    },
    "sodium": {
        "label": "Sodium",
        "unit": "mmol/L",
        "units": {"mmol/l": 1.0, "meq/l": 1.0},
        "rules": [
            {"sex": "any", "age_min": None, "age_max": None, "low": 135.0, "high": 145.0},
        ],
    },
    "potassium": {
        "label": "Potassium",
        "unit": "mmol/L",
        "units": {"mmol/l": 1.0, "meq/l": 1.0},
        "rules": [
            {"sex": "any", "age_min": None, "age_max": None, "low": 3.5, "high": 5.1},
        ],
    },
    "ferritin": {
        "label": "Ferritin",
        "unit": "ng/mL",
        "units": {"ng/ml": 1.0, "ug/l": 1.0, "µg/l": 1.0, "mcg/l": 1.0},
        "rules": [
            {"sex": "male", "age_min": 18, "age_max": None, "low": 24.0, "high": 336.0},
            {"sex": "female", "age_min": 18, "age_max": None, "low": 11.0, "high": 307.0},
        ],
    },
    "vitamin_d": {
        "label": "Vitamin D",
        "unit": "ng/mL",
        "units": {"ng/ml": 1.0, "nmol/l": 0.4006},
        "rules": [
            {"sex": "any", "age_min": 18, "age_max": None, "low": 30.0, "high": 100.0},
        ],
    },
    "vitamin_b12": {
        "label": "Vitamin B12",
        "unit": "pg/mL",
        "units": {"pg/ml": 1.0, "ng/l": 1.0, "pmol/l": 1.355},
        "rules": [
            {"sex": "any", "age_min": None, "age_max": None, "low": 200.0, "high": 900.0},
        ],
    },
    "uric_acid": {
        "label": "Uric acid",
        "unit": "mg/dL",
        "units": {"mg/dl": 1.0, "mg%": 1.0, "umol/l": 0.0168, "µmol/l": 0.0168},
        "rules": [
            {"sex": "male", "age_min": 18, "age_max": None, "low": 3.4, "high": 7.0},
            {"sex": "female", "age_min": 18, "age_max": None, "low": 2.4, "high": 6.0},
        ],
    },
    "crp": {
        "label": "C-reactive protein",
        "unit": "mg/L",
        "units": {"mg/l": 1.0, "mg/dl": 10.0},
        "rules": [
            {"sex": "any", "age_min": 18, "age_max": None, "low": None, "high": 5.0},
        ],
    },
}

# How a lookup describes where its numbers came from. Shown to the reader so
# a general interval is never mistaken for the one their own lab printed.
GENERAL_INTERVAL_SOURCE = "a general normal range, not the one from your lab"


def normalize_unit(unit: Optional[str]) -> str:
    """Lowercases and strips whitespace so 'x10^9 / L' and 'X10^9/L' compare
    equal. Deliberately does NOT strip punctuation: 'mg/dl' and 'mg/l' differ
    by one character and a tenfold factor."""
    if not unit or not isinstance(unit, str):
        return ""
    return re.sub(r"\s+", "", unit).lower()


# A ratio is a measurement in its own right, not the analyte whose name it
# borrows -- and its name usually contains two of them.
_RATIO_NAME = re.compile(r"\bratios?\b", re.IGNORECASE)
_LIPID_NAME = re.compile(r"\b(chol|cholesterol|hdl|ldl|tc)\b", re.IGNORECASE)


def canonical_test(test_name: Optional[str]) -> Optional[str]:
    """Maps a printed test name onto a canonical id, or None if it is not one
    this module knows. Word-boundary matched, longest synonym first."""
    name = (test_name or "").lower()
    if not name.strip():
        return None

    # Checked BEFORE the synonym table, because a ratio's name contains the
    # names of the analytes it is a ratio OF, and would otherwise resolve to
    # whichever of them the table happens to match first. "Total Cholesterol /
    # HDL Ratio" did exactly that -- it landed on `hdl` and merged a ratio of
    # 5.1 into the HDL series beside readings of 38 and 41 mg/dL, producing a
    # four-point "trend" across two different measurements. Matching on the
    # word "ratio" rather than on spelled-out forms means the spacing and
    # punctuation around the slash ("Chol/HDL", "Cholesterol : HDL Ratio")
    # stop mattering, which is what defeated the earlier synonym list.
    if _RATIO_NAME.search(name):
        # A non-lipid ratio gets no id at all, so it groups under its own
        # printed name rather than being folded into a test it merely
        # mentions.
        return "cholesterol_ratio" if _LIPID_NAME.search(name) else None

    for test_id, synonyms in TEST_SYNONYMS:
        for kw in sorted(synonyms, key=len, reverse=True):
            if re.search(rf"\b{re.escape(kw)}\b", name):
                return test_id
    return None


def is_main_test(test_name: Optional[str]) -> bool:
    """Whether this test is one of the ones surfaced without being asked for."""
    return canonical_test(test_name) in MAIN_TESTS


def test_label(test_id: Optional[str]) -> Optional[str]:
    entry = REFERENCE_INTERVALS.get(test_id or "")
    return entry["label"] if entry else None


def to_canonical_value(
    test_id: str, value: float, unit: Optional[str]
) -> Optional[Tuple[float, str]]:
    """Converts `value` into the table's unit for `test_id`, returning
    (converted_value, canonical_unit).

    Returns None when the reported unit is not one this test accepts -- a
    silently wrong conversion would produce a confident status off the wrong
    number, which is worse than declining to interpret.

    A MISSING unit is treated as already canonical. Plenty of reports print
    the range and value without repeating the unit on every row, and refusing
    those would lose most of the table's usefulness; the caller records that
    the unit was assumed so it can be said out loud.
    """
    entry = REFERENCE_INTERVALS.get(test_id)
    if entry is None:
        return None
    normalized = normalize_unit(unit)
    if not normalized:
        return value, entry["unit"]
    factor = entry["units"].get(normalized)
    if factor is None:
        return None
    return value * factor, entry["unit"]


def lookup_interval(
    test_id: Optional[str], sex: Optional[str], age: Optional[float]
) -> Optional[Dict[str, Any]]:
    """The interval for this test given the patient's sex and age, as
    {"low", "high", "unit", "basis", "source"} in the table's canonical unit,
    or None if no rule applies.

    `low`/`high` may individually be None for an open-ended interval.
    `basis` names which demographics the matched rule actually depended on,
    so the explanation can say "for a woman of your age" only when the rule
    really was sex- and age-specific.
    """
    entry = REFERENCE_INTERVALS.get(test_id or "")
    if entry is None:
        return None

    normalized_sex = (sex or "").strip().lower()
    if normalized_sex not in ("male", "female"):
        normalized_sex = None

    for rule in entry["rules"]:
        if rule["sex"] != "any":
            # A sex-specific rule cannot be applied to an unknown sex.
            if normalized_sex is None or rule["sex"] != normalized_sex:
                continue
        if rule["age_min"] is not None or rule["age_max"] is not None:
            if age is None:
                continue
            if rule["age_min"] is not None and age < rule["age_min"]:
                continue
            if rule["age_max"] is not None and age > rule["age_max"]:
                continue
        return {
            "low": rule["low"],
            "high": rule["high"],
            "unit": entry["unit"],
            "basis": {
                "sex_specific": rule["sex"] != "any",
                "age_specific": rule["age_min"] is not None or rule["age_max"] is not None,
            },
            "source": GENERAL_INTERVAL_SOURCE,
        }
    return None


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Name resolution, including the traps the word-boundary matching exists
    # to avoid.
    assert canonical_test("Fasting Glucose") == "glucose_fasting"
    assert canonical_test("FBS") == "glucose_fasting"
    assert canonical_test("Glucose") == "glucose_fasting"
    assert canonical_test("HbA1c") == "hba1c"
    assert canonical_test("Cobalt") is None
    assert canonical_test("Alkaline Phosphatase") is None
    assert canonical_test("Some Unlisted Assay") is None
    assert canonical_test("SGPT") == "alt"
    assert canonical_test("Vitamin B12") == "vitamin_b12"

    # The lipid panel: every name below contains "cholesterol", and a generic
    # match on that word merged the whole panel into one test. Each must
    # resolve to its own analyte -- a Total Cholesterol of 232, an HDL of 38,
    # an LDL of 157 and a ratio of 6.1 are four results, not four readings of
    # one test.
    assert canonical_test("Total Cholesterol") == "cholesterol_total"
    assert canonical_test("Cholesterol") == "cholesterol_total"
    assert canonical_test("HDL Cholesterol") == "hdl"
    assert canonical_test("LDL Cholesterol") == "ldl"
    assert canonical_test("Non-HDL Cholesterol") == "non_hdl_cholesterol"
    # Every spacing and punctuation variant of a ratio must stay off the
    # analyte it names. The spelled-out synonym list handled "Cholesterol/HDL
    # Ratio" and missed "Total Cholesterol / HDL Ratio", which then merged
    # into HDL.
    for ratio_name in (
        "Cholesterol/HDL Ratio", "TC/HDL Ratio", "Total Cholesterol / HDL Ratio",
        "Cholesterol : HDL Ratio", "Chol/HDL", "LDL/HDL Ratio", "CHOL/HDL RATIO",
    ):
        if "ratio" in ratio_name.lower():
            assert canonical_test(ratio_name) == "cholesterol_ratio", ratio_name
    # A ratio of something this module knows nothing about gets no id, so it
    # groups under its own name instead of joining a test it merely mentions.
    assert canonical_test("Albumin/Creatinine Ratio") is None
    assert canonical_test("Neutrophil Lymphocyte Ratio") is None
    assert len({
        canonical_test(n) for n in
        ("Total Cholesterol", "HDL Cholesterol", "LDL Cholesterol", "Cholesterol/HDL Ratio")
    }) == 4
    # A ratio and a non-HDL have no interval in the table, which is correct:
    # they group as their own test and simply get no status.
    assert lookup_interval("cholesterol_ratio", "female", 40) is None
    assert lookup_interval("non_hdl_cholesterol", "female", 40) is None

    # Sex-specific rules must refuse to answer for an unknown sex rather than
    # falling through to the other sex's interval.
    male_hb = lookup_interval("hemoglobin", "male", 40)
    female_hb = lookup_interval("hemoglobin", "female", 40)
    assert male_hb["low"] == 13.5 and female_hb["low"] == 12.0
    assert lookup_interval("hemoglobin", None, 40) is None
    assert male_hb["basis"]["sex_specific"] is True

    # Age-banded rules must refuse an unknown age...
    assert lookup_interval("glucose_fasting", None, None) is None
    # ...but a rule with no age band still answers without one.
    plt = lookup_interval("platelets", None, None)
    assert plt is not None and plt["low"] == 150.0
    assert plt["basis"]["age_specific"] is False

    # A child must not be handed the adult interval.
    assert lookup_interval("glucose_fasting", "male", 8) is None
    child_hb = lookup_interval("hemoglobin", "male", 8)
    assert child_hb["low"] == 11.0, child_hb

    # Open-ended intervals keep their open side as None.
    assert lookup_interval("cholesterol_total", "any", 40)["low"] is None
    assert lookup_interval("egfr", None, 40)["high"] is None

    # Unit handling: aliases convert, unknown units refuse.
    assert to_canonical_value("glucose_fasting", 5.5, "mmol/L")[0] == 5.5 * 18.0182
    assert to_canonical_value("hemoglobin", 130, "g/L")[0] == 13.0
    assert to_canonical_value("wbc", 7500, "/uL")[0] == 7.5
    assert to_canonical_value("glucose_fasting", 91, "mg/dL")[0] == 91
    assert to_canonical_value("glucose_fasting", 91, None)[0] == 91  # assumed canonical
    assert to_canonical_value("glucose_fasting", 91, "furlongs") is None

    assert is_main_test("Fasting Glucose") is True
    assert is_main_test("Some Unlisted Assay") is False

    print(f"{len(REFERENCE_INTERVALS)} tests in table, {len(MAIN_TESTS)} marked main.")
    print("All checks passed.")
