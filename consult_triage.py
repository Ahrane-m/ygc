"""
Consultation Triage / Referral Routing
=========================================
Runs AFTER extraction (medical_extractor.py), the safety cross-check
(cross_check_prescriptions) and lab trend tracking (lab_trends.py). It reads
those already-computed findings — each of which already carries its own
confidence score — and answers the one question the rest of the pipeline
deliberately stops short of:

    "Given what was found, does this patient need to talk to someone —
     a pharmacist or a doctor — and if a doctor, what kind?"

Nothing here re-analyzes documents or re-derives clinical findings. It is a
ROUTING layer over findings that already exist.

WHY THE ROUTING IS DETERMINISTIC
--------------------------------
Who resolves a finding is a question about professional scope of practice,
not about medicine, and scope of practice is stable enough to encode:

  * A therapeutic duplication or an ambiguous "which dose is current?" is a
    medication-reconciliation problem — a pharmacist's core competency, and
    reachable same-day without an appointment.
  * A drug prescribed against a documented allergy needs the prescription
    CHANGED, which is a prescribing decision only a doctor can make (a
    pharmacist can and should intercept it, but cannot substitute it).
  * A lab value that has drifted out of its reference range needs
    INTERPRETATION in clinical context — a diagnostic act, so a doctor.

Encoding that as a table (see ROUTING_RULES below) means the answer is the
same every run and can be explained line by line, matching the philosophy
lab_trends.py and detect_exact_duplicate_medications() already follow: compute
what code can determine for certain, and spend an LLM call only where actual
medical knowledge is required.

The ONE thing code cannot do is name a specialty: mapping "ALT has been
climbing" to hepatology, or a specific drug class to the specialist who
manages it, is medical knowledge. So specialty selection is a small LLM pass
(_suggest_specialties_llm) sitting behind a deterministic map of the common
cases (LAB_SPECIALTY_RULES). If the model call fails, or no key is set, the
module still returns a complete answer — it just falls back to the rule map
and, failing that, to a general practitioner. A referral is never lost to an
API error.

TWO SAFETY PROPERTIES THIS MODULE MAINTAINS
-------------------------------------------
1. It never de-escalates. `consult_needed: false` means "these specific
   automated checks found no trigger", NOT "you are fine" and NOT "you do not
   need to see anyone". The summary text says so explicitly, because a
   patient reading a green light is the one failure mode here that could
   cause real harm.
2. Low confidence never lowers urgency. `confidence` describes how sure the
   pipeline is that the finding is REAL (it is inherited from the finding
   that triggered the referral); it is not a measure of how safely the
   patient can ignore it. A half-legible handwritten allergy conflict is
   still worth a phone call — so a low-confidence trigger is reported with a
   low confidence score and its full urgency, and the wording tells the
   reader to verify the source document rather than to discount it.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAIError

from language_guard import assess_translation_risk
from medical_extractor import client, MODEL

# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------

# Deliberately tops out at "urgent" — there is no "emergency" level. Every
# finding here is derived from documents that were uploaded, i.e. from the
# past; calling an emergency off stale paperwork would be both unreliable and
# harmful. Anything genuinely happening right now is covered by the standing
# advice in EMERGENCY_ADVICE, which is attached to every result regardless of
# what was found.
URGENCY_ORDER = {"routine": 0, "soon": 1, "urgent": 2}

URGENCY_MEANING = {
    "routine": "raise it at the next scheduled appointment",
    "soon": "make contact within the next few days, ahead of any routine appointment",
    "urgent": "make contact today or tomorrow, and do not wait for a scheduled appointment",
}

# The same three levels as one short phrase, for the headline. `summary` says
# everything; a person scanning their results reads one line and stops, so
# there needs to be a line worth stopping at.
URGENCY_WHEN = {
    "routine": "at your next appointment",
    "soon": "in the next few days",
    "urgent": "today or tomorrow",
}

# A pharmacist is reachable without an appointment and can resolve
# medication-supply questions outright, so it is the lower rung — but "lower"
# means faster and cheaper, never less safe. Findings that need a prescribing
# or diagnostic decision are routed past it.
ROUTE_ORDER = {"pharmacist": 0, "doctor": 1}

EMERGENCY_ADVICE = (
    "This routing is based on uploaded documents, which describe the past. It "
    "cannot see how the patient is right now. Anyone with severe or sudden "
    "symptoms — trouble breathing, chest pain, swelling of the face or throat, "
    "a spreading rash, fainting, or confusion — should seek emergency care "
    "immediately rather than follow anything suggested here."
)

DISCLAIMER = (
    "This is an automated routing suggestion derived from the extracted "
    "documents, not a diagnosis or a triage decision. It says who is likely "
    "the right person to ask, not what is wrong. A licensed clinician decides "
    "what any finding actually means. Where no trigger was found, that means "
    "only that these specific automated checks found nothing — it is not a "
    "clean bill of health and is never a reason to skip or delay care."
)

# Matches retrieval.py's LOW_CONFIDENCE_THRESHOLD. At or below this, the
# finding is reported with an explicit "verify against the source document"
# caveat rather than being dropped — dropping a low-confidence allergy
# conflict is the more dangerous error.
LOW_CONFIDENCE_THRESHOLD = 0.6

# Triggers that describe how well a DOCUMENT WAS READ, not what was found in
# the patient's record. They are still reported — see `document_quality_notices`
# in the result — but they never set consult_needed, consult_type or urgency.
#
# The distinction matters because the two answer different questions. "Two of
# your medicines interact" is a fact about the patient and is a reason to speak
# to someone. "One field on your scan was hard to read" is a fact about the
# paperwork, and telling someone to go and see a pharmacist over it trains them
# to ignore the times it says something real.
DATA_QUALITY_TRIGGERS = {"low_extraction_confidence", "translation_uncertain"}

# Above this, the document was read well enough that a note about one field is
# a transcription caveat rather than a reason to distrust the medication list.
#
# This threshold exists because `illegible_or_low_confidence_fields` is not
# purely a list of unreadable fields: the extractor also uses it to record
# interpretation choices it made on text it read perfectly well ("this
# combination product was recorded as its mass component"). On a document that
# scored 0.85 overall, such an entry is the extractor being transparent, not
# the extractor struggling — so it must not by itself produce a referral.
TRUSTED_EXTRACTION_THRESHOLD = 0.8

# Which unreadable fields are worth raising at all. An unreadable clinic
# footer, letterhead or signature line changes nothing about what the patient
# is taking; an unreadable dose changes everything.
MATERIAL_FIELD_PATTERN = re.compile(
    r"\b(medication|medicine|drug|ingredient|dosage|dose|strength|frequency|"
    r"lab.?result|test.?name|reference.?range|value|unit|allerg)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 1. Routing rules — cross-check findings
#
# Keyed by (finding kind, severity or None). Each rule says who resolves it,
# how fast, and — importantly — WHY that route, since the "why" is what makes
# the recommendation auditable rather than an opaque verdict.
# ---------------------------------------------------------------------------

ROUTING_RULES: Dict[Tuple[str, Optional[str]], Dict[str, str]] = {
    ("allergy_conflict", None): {
        "route": "doctor",
        "urgency": "urgent",
        "why_this_route": (
            "A medication on file conflicts with an allergy recorded in these same "
            "documents. Resolving it means changing or substituting the prescription, "
            "which is a prescribing decision. A pharmacist can flag it immediately and "
            "should be told, but the prescriber has to make the change."
        ),
    },
    ("drug_interaction", "high"): {
        "route": "doctor",
        "urgency": "urgent",
        "why_this_route": (
            "A high-severity interaction generally means the combination itself needs "
            "to be reconsidered — a change of drug or of the treatment plan, which only "
            "a prescriber can authorize."
        ),
    },
    ("drug_interaction", "moderate"): {
        "route": "pharmacist",
        "urgency": "soon",
        "why_this_route": (
            "Moderate interactions are usually managed by adjusting timing, spacing "
            "doses, or watching for specific side effects — advice a pharmacist can give "
            "directly, without an appointment. They will refer on to the prescriber if "
            "the regimen itself needs changing."
        ),
    },
    ("drug_interaction", "low"): {
        "route": "pharmacist",
        "urgency": "routine",
        "why_this_route": (
            "A low-severity interaction is worth mentioning to a pharmacist next time "
            "the prescription is collected; it rarely requires a change on its own."
        ),
    },
    ("guideline_flagged_combination", None): {
        "route": "doctor",
        "urgency": "urgent",
        # The one finding type in this table backed by a named, quotable
        # source rather than model recall — published guidance says this
        # specific combination "can greatly increase the risk of overdose".
        # Resolving it means changing a prescription, which is a doctor's call.
        "why_this_route": (
            "A strong painkiller and a sedative were prescribed for the same period. "
            "Published national guidance warns that this combination can seriously "
            "slow breathing. Changing either prescription is a doctor's decision, and "
            "this is worth raising quickly — a pharmacist can also advise today while "
            "you arrange it."
        ),
    },
    ("concurrent_double_dose", None): {
        "route": "pharmacist",
        "urgency": "urgent",
        # Ranked above a plain duplicate: this is not "the same drug appears
        # twice in your records", it is "two live prescriptions were supplying
        # it over the same dates". The dose the patient actually took is the
        # sum, and each prescription looks reasonable alone — which is exactly
        # why nobody notices.
        "why_this_route": (
            "Two prescriptions that were active at the same time both supplied the same "
            "medicine, so the amount actually taken was the two added together. A "
            "pharmacist can check the dispensing record and say whether that total was "
            "safe — they can do this today, without an appointment."
        ),
    },
    ("duplicate_prescription", None): {
        "route": "pharmacist",
        "urgency": "soon",
        "why_this_route": (
            "The same active ingredient appearing on more than one prescription risks "
            "double-dosing without the patient realising, often because two prescribers "
            "used different brand names. Reconciling a medication list is exactly what "
            "a pharmacist does, and they can do it same-day."
        ),
    },
    ("dosage_conflict", None): {
        "route": "pharmacist",
        "urgency": "soon",
        "why_this_route": (
            "Two documents give different instructions for the same drug, so it is "
            "unclear which one is current. A pharmacist can check the dispensing record "
            "and establish which instruction stands — and will contact the prescriber if "
            "the records genuinely disagree."
        ),
    },
    ("lab_crossed_abnormal", None): {
        "route": "doctor",
        "urgency": "soon",
        "why_this_route": (
            "A test result that moved from inside the normal range to outside it needs "
            "someone to say what it means for this particular person, whether the test "
            "should be repeated, and whether anything needs to happen next. Only a doctor "
            "can make that call — a pharmacist is not able to."
        ),
    },
    ("lab_persistently_abnormal", None): {
        "route": "doctor",
        "urgency": "soon",
        "why_this_route": (
            "This test has been outside the normal range in every result on file. Whether "
            "that is already known about and being looked after, or has never been "
            "followed up, is something only the treating doctor can confirm."
        ),
    },
    ("lab_approaching_threshold", None): {
        "route": "doctor",
        "urgency": "routine",
        "why_this_route": (
            "The result is still inside the normal range, but it has been moving closer "
            "to the edge of it. Nothing is outside the normal range yet — this is worth "
            "mentioning at the next scheduled appointment so it is on the record and can "
            "be kept an eye on."
        ),
    },
    ("lab_single_abnormal", None): {
        "route": "doctor",
        "urgency": "routine",
        "why_this_route": (
            "This test appears only once in the records on file, and that one result is "
            "outside the normal range. With nothing earlier to compare it against there "
            "is no way to tell from these documents whether it has changed or has always "
            "been at this level, which is exactly the question a doctor can settle — "
            "usually by repeating the test."
        ),
    },
    ("low_extraction_confidence", None): {
        "route": "pharmacist",
        "urgency": "routine",
        "why_this_route": (
            "Parts of this document could not be read reliably, so the medication list "
            "built from it may be wrong. A pharmacist can check the extracted list "
            "against the actual dispensing record — this is a data-quality check, not a "
            "clinical finding."
        ),
    },
    ("translation_uncertain", None): {
        "route": "pharmacist",
        "urgency": "soon",
        # Ranked above low_extraction_confidence: an unreadable field is
        # visibly missing, whereas a mistranslated drug name looks completely
        # normal in the record and is wrong. A pharmacist holds both the
        # original document and the dispensing record, so they can settle it
        # in one conversation.
        "why_this_route": (
            "This document was not written in English, so the medicine names in these "
            "records were converted from another language rather than printed on the "
            "page. If that conversion was wrong, the record would still look perfectly "
            "normal. A pharmacist can compare the original document against what was "
            "actually dispensed and confirm the names."
        ),
    },
}


# ---------------------------------------------------------------------------
# 2. Specialty rules — the common lab tests, resolved without an LLM call
#
# Keyed by a substring matched case-insensitively against the test name. Only
# well-established, unambiguous test-to-discipline mappings belong here;
# anything requiring actual judgment is left to the LLM pass, which is why
# this table is deliberately short rather than exhaustive.
# ---------------------------------------------------------------------------

# (keywords, what to call it to a patient, the clinical name, one short line)
#
# The patient-facing name comes first because it is the one that gets shown.
# Someone reading their own results does not know what hepatology is, and a
# label you have to look up is a label you skip. The clinical name is kept
# beside it — a receptionist needs it to book, a clinician expects it — but it
# is never the thing on screen.
LAB_SPECIALTY_RULES: List[Tuple[Tuple[str, ...], str, str, str]] = [
    (("alt", "ast", "sgpt", "sgot", "bilirubin", "alkaline phosphatase", "ggt",
      "alp"),
     "Liver specialist", "Hepatology / Gastroenterology", "checks your liver"),
    (("creatinine", "egfr", "gfr", "urea", "bun"),
     "Kidney specialist", "Nephrology", "checks your kidneys"),
    (("glucose", "hba1c", "a1c", "insulin"),
     "Diabetes specialist", "Endocrinology", "checks your blood sugar"),
    (("tsh", "t3", "t4", "thyroid"),
     "Thyroid specialist", "Endocrinology", "checks your thyroid"),
    (("cholesterol", "ldl", "hdl", "triglyceride", "lipid"),
     "Heart specialist", "Cardiology / lipid clinic", "checks the fats in your blood"),
    (("hemoglobin", "haemoglobin", "hematocrit", "haematocrit", "platelet",
      "wbc", "rbc", "mcv", "white blood cell", "red blood cell"),
     "Blood specialist", "Haematology", "checks your blood count"),
    (("inr", "prothrombin", "aptt", "ptt"),
     "Blood specialist", "Haematology", "checks how your blood clots"),
    (("uric acid", "urate"),
     "Joint specialist", "Rheumatology", "is linked to joint problems like gout"),
    (("psa",),
     "Prostate specialist", "Urology", "checks your prostate"),
    (("troponin", "bnp", "nt-probnp"),
     "Heart specialist", "Cardiology", "checks your heart"),
]

# Triggers that represent a genuine safety finding — something is actually
# wrong with the medications or results on file. Everything outside this set
# is either a data-quality note (an unreadable scan, an uncertain translation)
# or a watch-item that is not yet abnormal.
ALERT_TRIGGERS = frozenset({
    "guideline_flagged_combination",
    "drug_interaction",
    "allergy_conflict",
    "duplicate_prescription",
    "concurrent_double_dose",
    "dosage_conflict",
    "lab_crossed_abnormal",
    "lab_persistently_abnormal",
})


def _warrants_specialty(item: Dict[str, Any]) -> bool:
    """
    Whether this finding is worth naming a KIND of doctor for.

    Naming a specialty is a strong signal — it reads as "you need to see a
    specialist about this". Attaching one to every doctor-routed item spends
    that signal on things that don't warrant it: a lab value still inside its
    normal range that has merely drifted a little, or a drug pairing from a
    course that ended two years ago. Both are worth a mention at a routine
    appointment; neither is worth telling someone to find a hepatologist.

    So a specialty is named only when there is a real safety finding, or when
    the finding is uncertain enough that someone needs to confirm it. A
    historical pairing is neither, whatever its confidence — courses that
    never overlapped were not a risk, so there is nothing for a specialist to
    act on.
    """
    if item.get("is_historical"):
        return False
    if item["trigger"] in ALERT_TRIGGERS:
        return True
    return item["confidence"] <= LOW_CONFIDENCE_THRESHOLD


GENERAL_PRACTITIONER = "Your regular doctor"
GENERAL_PRACTITIONER_CLINICAL = "General practitioner"

GP_FIRST_REASON = "They know your history and can refer you on if a specialist is needed."


def _match_lab_specialty(test_name: str) -> Optional[Tuple[str, str, str]]:
    """Returns (plain_name, clinical_name, reason_fragment) for a test name the
    rule table covers, else None — in which case the LLM pass gets a chance."""
    name = (test_name or "").lower()
    for keywords, plain, clinical, reason in LAB_SPECIALTY_RULES:
        for kw in keywords:
            # Word-boundary match so a short key like "alt" doesn't fire on
            # "Alkaline Phosphatase" or "Cobalt".
            if re.search(rf"\b{re.escape(kw)}\b", name):
                return plain, clinical, reason
    return None


# ---------------------------------------------------------------------------
# 3. Building referral items from the already-computed findings
# ---------------------------------------------------------------------------

def _clamp_confidence(value: Any, default: float = 0.5) -> float:
    """Findings arrive with a model-assigned confidence. Anything missing or
    non-numeric becomes `default` rather than silently becoming 1.0."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return default
    return round(max(0.0, min(1.0, float(value))), 2)


def _make_item(
    trigger: str,
    subject: str,
    detail: str,
    confidence: float,
    severity: Optional[str] = None,
    lab_test: Optional[str] = None,
) -> Dict[str, Any]:
    """Applies the routing table to one finding, producing a referral item.
    The item carries the source finding's own confidence forward — it is never
    recomputed here, because this module adds no new evidence."""
    rule = ROUTING_RULES.get((trigger, severity)) or ROUTING_RULES[(trigger, None)]
    item: Dict[str, Any] = {
        "trigger": trigger,
        "subject": subject,
        "detail": detail,
        "route": rule["route"],
        "urgency": rule["urgency"],
        "why_this_route": rule["why_this_route"],
        "confidence": confidence,
        # "clinical" == a finding about the patient, and therefore a reason to
        # consult someone. "data_quality" == a finding about the document it
        # came from, reported separately and never counted as a referral.
        "category": "data_quality" if trigger in DATA_QUALITY_TRIGGERS else "clinical",
    }
    if severity:
        item["severity"] = severity
    if lab_test:
        item["lab_test"] = lab_test
    if confidence <= LOW_CONFIDENCE_THRESHOLD:
        item["confidence_caveat"] = (
            f"This came from a finding the pipeline is only {confidence:.0%} confident "
            "in — most often because the source document was hard to read. Check it "
            "against the original document or the dispensing record. Treat it as "
            "unverified, not as unimportant: the urgency above is unchanged, because a "
            "finding being uncertain is a reason to confirm it, not to ignore it."
        )
    return item


def _timing_block(dated: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the same `timing` shape the cross-check findings carry, from a
    finding that computed its own window (a guideline-flagged combination, a
    double-dose period).

    These are the most firmly dated findings in the report — their windows come
    from arithmetic over treatment dates, not from a model — yet they used to
    carry those dates only inside the prose `detail`. A client filtering or
    sorting by `timing` therefore skipped exactly the findings it should have
    surfaced first. Same field on every finding, so date handling can be
    uniform.
    """
    return {
        "status": dated.get("status") or "concurrent",
        "window_start": dated.get("window_start"),
        "window_end": dated.get("window_end"),
        "overlap_days": dated.get("overlap_days", 0),
        "gap_days": 0,
        "note": (
            f"Both were active {dated.get('window_start')} to {dated.get('window_end')}."
            if dated.get("window_start") else
            "The dates could not be established from the documents."
        ),
    }


def _apply_timing(item: Dict[str, Any], finding: Dict[str, Any]) -> Dict[str, Any]:
    """
    Folds a finding's treatment-window timing into the routed item.

    A pair of drugs whose courses finished years apart is not a live risk, and
    routing it at the same urgency as a current one is how a referral list
    fills with noise. Such findings are NOT dropped — they stay, de-escalated
    to `routine` and labelled as history, because "you were once on these
    together" is still worth a mention, just not worth a same-week phone call.

    Note this is the one thing allowed to lower urgency, and it is not a
    confidence judgment: it is an arithmetic fact about dates, which is exactly
    the kind of evidence the low-confidence rule was protecting against being
    overridden by.
    """
    timing = finding.get("timing")
    if not timing:
        return item

    item["timing"] = timing
    status = timing.get("status")

    if status == "not_concurrent":
        item["urgency"] = "routine"
        item["is_historical"] = True
        item["why_this_route"] = (
            "These medicines were never taken at the same time — the courses finished "
            f"about {timing.get('gap_days')} day(s) apart, so this was not a live risk. "
            "It is kept on the record because it is worth mentioning at a routine "
            "appointment, not because anything needs doing now."
        )
    elif status in ("concurrent", "possible"):
        item["is_historical"] = False
        window = f"{timing.get('window_start')} to {timing.get('window_end')}"
        item["detail"] = f"{item['detail']} Active {window}.".strip()
    return item


def _items_from_cross_check(cross_check: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Maps each cross_check_prescriptions() finding onto a routed item."""
    items: List[Dict[str, Any]] = []

    for finding in cross_check.get("allergy_conflicts") or []:
        items.append(_make_item(
            "allergy_conflict",
            subject=f"{finding.get('medication') or 'unnamed medication'} "
                    f"vs allergy '{finding.get('allergy') or 'unspecified'}'",
            detail=finding.get("explanation") or "",
            confidence=_clamp_confidence(finding.get("confidence")),
        ))

    for finding in cross_check.get("potential_drug_interactions") or []:
        severity = (finding.get("severity") or "moderate").lower()
        if (("drug_interaction", severity)) not in ROUTING_RULES:
            # An unrecognised severity is routed as moderate rather than
            # dropped — an unroutable interaction must not become an invisible
            # one.
            severity = "moderate"
        involved = finding.get("medications_involved") or []
        items.append(_apply_timing(_make_item(
            "drug_interaction",
            subject=" + ".join(involved) if involved else "unnamed medications",
            detail=finding.get("explanation") or "",
            confidence=_clamp_confidence(finding.get("confidence")),
            severity=severity,
        ), finding))

    for finding in cross_check.get("duplicate_prescriptions") or []:
        items.append(_apply_timing(_make_item(
            "duplicate_prescription",
            subject=finding.get("medication") or "unnamed medication",
            detail=finding.get("explanation") or "",
            confidence=_clamp_confidence(finding.get("confidence")),
        ), finding))

    for finding in cross_check.get("conflicting_dosage_instructions") or []:
        items.append(_apply_timing(_make_item(
            "dosage_conflict",
            subject=finding.get("medication") or "unnamed medication",
            detail=finding.get("explanation") or "",
            confidence=_clamp_confidence(finding.get("confidence")),
        ), finding))

    # Opioid + sedative prescribed over the same dates, cited to published
    # guidance. Listed before the double-dose check because it is the one
    # finding here that carries a real source behind it.
    for combo in cross_check.get("guideline_flagged_combinations") or []:
        window = ""
        if combo.get("window_start"):
            window = f" Both were active {combo['window_start']} to {combo['window_end']}."
        citation = combo.get("citation") or {}
        item = _make_item(
            "guideline_flagged_combination",
            subject=f"{combo.get('opioid')} + {combo.get('depressant')}",
            detail=(combo.get("plain") or "") + window,
            confidence=0.9,  # dated arithmetic plus a cited source
        )
        item["reference"] = citation
        item["evidence_source"] = "reference_graph"
        item["timing"] = _timing_block(combo)
        items.append(item)

    # Periods where two live prescriptions supplied the same ingredient — the
    # double-dosing exposure the patient can hit without realising, since each
    # prescription looks reasonable on its own. Dated, so it says WHEN.
    for exposure in cross_check.get("concurrent_exposure") or []:
        dose = ""
        if exposure.get("cumulative_daily_dose") is not None and exposure.get("dosage_unit"):
            dose = (f" Combined that is {exposure['cumulative_daily_dose']} "
                    f"{exposure['dosage_unit']} per day.")
        double_dose = _make_item(
            "concurrent_double_dose",
            subject=exposure.get("ingredient") or "unnamed ingredient",
            detail=(exposure.get("note") or "") + dose,
            confidence=0.9,  # arithmetic over dated records, not model inference
        )
        double_dose["timing"] = _timing_block(exposure)
        items.append(double_dose)

    return items


def _items_from_lab_trends(lab_trends: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Maps lab trend results onto routed items. Only three states here are
    actionable — crossing out of range, having always been out of range, and
    drifting toward a boundary. A stable in-range test produces nothing."""
    items: List[Dict[str, Any]] = []

    for trend in lab_trends.get("trends") or []:
        test_name = trend.get("test_name") or "unnamed test"
        confidence = _clamp_confidence(trend.get("confidence"))
        explanation = trend.get("explanation") or ""
        points = trend.get("data_points") or []
        last_flag = (points[-1].get("flag") if points else None) or "unknown"

        if trend.get("crossed_into_abnormal_at"):
            # lab_trends' `explanation` is already written in plain language
            # for a reader with no medical background, and already names the
            # crossing and its date — so it is used verbatim. Prefixing it
            # with the raw flag value would both repeat that sentence and
            # drop the register back to jargon.
            items.append(_make_item(
                "lab_crossed_abnormal",
                subject=test_name,
                detail=explanation,
                confidence=confidence,
                lab_test=test_name,
            ))
        elif last_flag in ("high", "low") and points and (points[0].get("flag") in ("high", "low")):
            items.append(_make_item(
                "lab_persistently_abnormal",
                subject=test_name,
                detail=explanation,
                confidence=confidence,
                lab_test=test_name,
            ))
        elif trend.get("approaching_threshold"):
            items.append(_make_item(
                "lab_approaching_threshold",
                subject=test_name,
                detail=explanation,
                confidence=confidence,
                lab_test=test_name,
            ))

    # Tests with a single reading and no history to trend against. Routed
    # more cautiously than a crossing: one out-of-range value with nothing
    # before it could be a real change or could be this patient's normal, and
    # nothing in the record distinguishes those. A normal or un-assessable
    # single result produces nothing, same as a stable in-range trend does.
    for single in lab_trends.get("single_results") or []:
        if single.get("status") not in ("high", "low"):
            continue
        items.append(_make_item(
            "lab_single_abnormal",
            subject=single.get("test_name") or "unnamed test",
            detail=single.get("explanation") or "",
            confidence=_clamp_confidence(single.get("confidence")),
            lab_test=single.get("test_name"),
        ))

    return items


def _has_material_illegible_field(entries: List[Any]) -> bool:
    """True when at least one unreadable field could change what the patient
    is understood to be taking.

    Everything in `illegible_or_low_confidence_fields` is free text written by
    the extractor, so this matches on keywords rather than parsing paths. It
    errs toward matching: the cost of one extra note is a line of text, while
    the cost of missing an unreadable dose is a wrong medication list."""
    return any(
        MATERIAL_FIELD_PATTERN.search(str(entry))
        for entry in entries
        if entry
    )


def _items_from_extraction_quality(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One item per document whose extraction was poor enough that the
    medication list built from it may not be trustworthy.

    These are DOCUMENT findings, not patient findings, so they are marked
    `category: "data_quality"` and reported separately from the referral —
    with one deliberate exception, noted inline below.

    Two distinct problems are handled separately, because they have different
    fixes: a document that could not be READ (rescan it) versus one whose
    English drug names were CONVERTED from another language (have a pharmacist
    confirm them). Collapsing them would tell the user to do the wrong thing
    half the time."""
    items: List[Dict[str, Any]] = []

    for visit in timeline.get("visits") or []:
        source = (visit.get("_source") or {}).get("file") or "an uploaded document"

        # Translation risk first — a mistranslated drug name is invisible in
        # the record, where an unreadable field at least announces itself.
        risk = assess_translation_risk(visit)
        if risk["flag"] in ("high", "review") and not risk["is_english_only"]:
            translation_confidence = risk["translation_confidence"]
            translation_item = _make_item(
                "translation_uncertain",
                subject=source,
                detail=risk["message"] or "",
                confidence=_clamp_confidence(
                    risk["effective_confidence"]
                    if risk["effective_confidence"] is not None
                    else translation_confidence,
                    default=0.7,
                ),
            )
            if risk["flag"] == "high":
                # The one document-quality problem that IS a referral. A "high"
                # flag means the drug names in the record may not be the drugs
                # on the page — and unlike a blurry scan, that failure is
                # invisible: the record reads as complete and correct. Only
                # someone holding both the original document and the dispensing
                # record can settle it, which is a pharmacist. A "review" flag
                # (nothing individually low, the two axes merely compound)
                # stays a note about the document.
                translation_item["category"] = "clinical"
            items.append(translation_item)

        overall = visit.get("overall_confidence")
        illegible = visit.get("illegible_or_low_confidence_fields") or []

        low_overall = isinstance(overall, (int, float)) and overall <= LOW_CONFIDENCE_THRESHOLD

        # A note about an unreadable field only counts on its own if the field
        # was one that matters AND the document did not otherwise read well.
        # Without the second half, a clean 0.85-confidence extraction that
        # transparently recorded one interpretation choice produced the same
        # output as a barely legible scan.
        trusted = (
            isinstance(overall, (int, float)) and overall >= TRUSTED_EXTRACTION_THRESHOLD
        )
        material_illegible = (
            bool(illegible) and not trusted and _has_material_illegible_field(illegible)
        )

        if not low_overall and not material_illegible:
            continue

        reasons = []
        if low_overall:
            reasons.append(f"overall extraction confidence was {overall:.0%}")
        if illegible:
            reasons.append(
                f"these fields could not be read reliably: {', '.join(str(f) for f in illegible)}"
            )

        items.append(_make_item(
            "low_extraction_confidence",
            subject=source,
            detail=(
                f"{source}: " + "; ".join(reasons) + ". Any medication or result taken "
                "from this document should be confirmed against the original."
            ),
            confidence=_clamp_confidence(overall, default=0.5),
        ))

    return items


# ---------------------------------------------------------------------------
# 4. Specialty selection — rule map first, LLM only for what it can't cover
# ---------------------------------------------------------------------------

SPECIALTY_PROMPT = """
You assign medical specialties to findings that have ALREADY been triaged as
needing a doctor. The decision that a doctor is needed has already been made
and is not yours to revisit — you are only naming which kind of doctor.

You are given a list of findings, each with an id, a short subject (a
medication or lab test), and a description. For each one, name the single
specialty best suited to it.

THE PATIENT READS THIS. Write for someone with no medical training who wants
one short, clear line — not a paragraph.
- `specialty` must be what you would SAY to them: "Kidney specialist", "Heart
  specialist", "Skin specialist", "Your regular doctor". Never the textbook
  term — not "Nephrology", not "Dermatology". If the right answer is just
  their usual doctor, say "Your regular doctor".
- `clinical_name` is the matching professional term ("Nephrology",
  "Dermatology", "General practitioner"), used for booking. Give both.
- `reason` is ONE short sentence, maximum about 15 words, addressed to them
  as "you" / "your". "Your creatinine test checks your kidneys." Not two
  sentences, no hedging clauses, no restating the finding.
- No abbreviations or clinical shorthand in `specialty` or `reason` — not
  "INR", "CYP", "eGFR", "PRN", "monitoring", "medication review". If a term
  would send someone to a search engine, replace it with what it means:
  "your blood needs checking more often", not "you need INR monitoring".

Rules:
- Prefer their regular doctor unless the finding clearly belongs to one
  specialty. Most findings do not need a specialist, and a regular doctor is
  faster to reach, holds the whole record, and can refer onward. Naming a
  specialist unnecessarily sends the patient down a slower path.
- Never name an individual, a hospital, or a clinic.
- Do not explain what the finding means clinically and do not suggest
  treatment — you are routing, not diagnosing. "Your ALT test checks your
  liver" is routing; "this suggests liver disease" is a diagnosis and is not
  allowed.

CONFIDENCE SCORING — anchor every confidence value to these bands:
- 0.90-1.00: the finding maps to exactly one discipline by long-standing
  convention (e.g. kidney-function tests -> nephrology).
- 0.60-0.89: the mapping is reasonable but a GP could equally handle it, or
  more than one specialty plausibly fits.
- Below 0.60: you are genuinely unsure which specialty fits. Prefer naming a
  general practitioner with a higher confidence over guessing a specialist
  with a low one.
"""

SPECIALTY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "specialty": {"type": "string"},
                    "clinical_name": {"type": "string"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["id", "specialty", "clinical_name", "reason", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["assignments"],
    "additionalProperties": False,
}

SPECIALTY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "specialty_assignment",
        "strict": True,
        "schema": SPECIALTY_JSON_SCHEMA,
    },
}


def _suggest_specialties_llm(
    pending: List[Tuple[int, Dict[str, Any]]], model: str = MODEL
) -> Dict[int, Dict[str, Any]]:
    """
    Asks the model to name a specialty for each doctor-routed item the rule
    map could not resolve. Returns {item_index: {specialty, reason,
    confidence}}.

    Returns {} on any failure. That is the whole point of the split: the
    referral itself is already decided deterministically, so losing this call
    costs a specialty name (which falls back to a GP), never the referral.
    """
    if not pending:
        return {}

    payload = [
        {"id": idx, "subject": item["subject"], "description": item["detail"][:600]}
        for idx, item in pending
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SPECIALTY_PROMPT},
                {
                    "role": "user",
                    "content": "Findings needing a specialty:\n\n"
                               + json.dumps(payload, indent=2),
                },
            ],
            response_format=SPECIALTY_RESPONSE_FORMAT,
        )
        parsed = json.loads(response.choices[0].message.content)
    except (OpenAIError, json.JSONDecodeError, KeyError, IndexError, TypeError):
        return {}

    valid_ids = {idx for idx, _ in pending}
    out: Dict[int, Dict[str, Any]] = {}
    for assignment in parsed.get("assignments") or []:
        idx = assignment.get("id")
        specialty = (assignment.get("specialty") or "").strip()
        if idx not in valid_ids or not specialty:
            continue
        out[idx] = {
            "specialty": specialty,
            "clinical_name": (assignment.get("clinical_name") or "").strip() or None,
            "reason": (assignment.get("reason") or "").strip(),
            "confidence": _clamp_confidence(assignment.get("confidence"), default=0.5),
        }
    return out


def _assign_specialties(
    items: List[Dict[str, Any]], model: str = MODEL, use_llm: bool = True
) -> None:
    """
    Attaches a `specialty` block to every doctor-routed item, in place.

    Resolution order — cheapest and most certain first:
      1. LAB_SPECIALTY_RULES, for lab findings on the common tests.
      2. One LLM call for whatever is left (medication findings, uncommon
         tests), if enabled and reachable.
      3. A general practitioner, which is both the safe default and, for most
         findings, the genuinely correct answer.
    """
    pending: List[Tuple[int, Dict[str, Any]]] = []

    for idx, item in enumerate(items):
        if item["route"] != "doctor":
            continue

        if not _warrants_specialty(item):
            # Still routed to a doctor — just not worth naming a KIND of
            # doctor for. Recorded so the omission is explainable rather than
            # looking like the specialty step silently failed.
            item["specialty"] = None
            item["specialty_omitted_because"] = (
                "this finding was never a live risk, so there is nothing for a "
                "specialist to act on"
                if item.get("is_historical") else
                "nothing here is outside the normal range and no safety issue was "
                "found — a specialist is not indicated by this alone"
            )
            continue

        if item.get("lab_test"):
            matched = _match_lab_specialty(item["lab_test"])
            if matched:
                plain, clinical, reason = matched
                item["specialty"] = {
                    "specialty": plain,
                    "clinical_name": clinical,
                    # One line, in the second person, naming the test the
                    # patient can see on their own report.
                    "reason": f"Your {item['lab_test']} test {reason}.",
                    "confidence": 0.9,
                    "basis": "rule",
                }
                continue
        pending.append((idx, item))

    # No qualifying finding means no specialty to name, and therefore no
    # reason to spend an API call deciding one. Skipped explicitly rather than
    # relying on the helper's own empty-input guard, so the saving is visible
    # here where the decision is made.
    assigned = (
        _suggest_specialties_llm(pending, model=model) if (use_llm and pending) else {}
    )

    for idx, item in pending:
        suggestion = assigned.get(idx)
        if suggestion:
            item["specialty"] = {**suggestion, "basis": "model"}
        else:
            item["specialty"] = {
                "specialty": GENERAL_PRACTITIONER,
                "clinical_name": GENERAL_PRACTITIONER_CLINICAL,
                "reason": GP_FIRST_REASON,
                "confidence": 0.7,
                "basis": "default",
            }


def _collect_specialties(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rolls per-item specialties up into a deduplicated list, most urgent
    first, so a caller can render "who to see" without walking every item.
    Each entry keeps the highest confidence and the strongest urgency seen for
    that specialty, plus which findings drove it."""
    by_specialty: Dict[str, Dict[str, Any]] = {}

    for item in items:
        specialty_block = item.get("specialty")
        if item["route"] != "doctor" or not specialty_block:
            continue
        name = specialty_block["specialty"]
        entry = by_specialty.setdefault(name, {
            "specialty": name,
            "clinical_name": specialty_block.get("clinical_name"),
            "reason": specialty_block["reason"],
            "confidence": specialty_block["confidence"],
            "basis": specialty_block["basis"],
            "urgency": item["urgency"],
            "triggered_by": [],
        })
        entry["triggered_by"].append(item["subject"])
        entry["confidence"] = max(entry["confidence"], specialty_block["confidence"])
        if URGENCY_ORDER[item["urgency"]] > URGENCY_ORDER[entry["urgency"]]:
            entry["urgency"] = item["urgency"]
            entry["reason"] = specialty_block["reason"]

    return sorted(
        by_specialty.values(),
        key=lambda e: (-URGENCY_ORDER[e["urgency"]], -e["confidence"], e["specialty"]),
    )


# ---------------------------------------------------------------------------
# 5. Summary text — templated from the computed fields, never generated
# ---------------------------------------------------------------------------

def _summarize(
    items: List[Dict[str, Any]],
    consult_type: Optional[str],
    urgency: Optional[str],
    specialties: List[Dict[str, Any]],
    quality_items: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Plain-language summary, filled from the fields computed above the same
    way lab_trends._explain() does — so it cannot say anything the routing
    does not already support.

    `items` is the CLINICAL findings only. Document-quality notices are passed
    separately and described as what they are, so the summary never presents a
    hard-to-read scan as a reason to consult someone."""
    quality_items = quality_items or []

    if not items:
        summary = (
            "No automated trigger for a consultation was found: the safety cross-check "
            "reported no interactions, duplicates, dosage conflicts or allergy conflicts, "
            "and no lab test on file has drifted out of its reference range. This is NOT "
            "a clean bill of health — it only means these specific checks found nothing "
            "in the documents provided. Anyone with symptoms, questions, or a scheduled "
            "review should still see their doctor or pharmacist as normal."
        )
        if quality_items:
            summary += (
                f" Separately, {len(quality_items)} uploaded document(s) could not be read "
                "with full confidence, so anything taken from them is worth checking "
                "against the original paperwork. That is a note about the scan, not a "
                "finding about the patient, and on its own it is not a reason to book "
                "anything."
            )
        return summary

    pharmacist_items = [i for i in items if i["route"] == "pharmacist"]
    doctor_items = [i for i in items if i["route"] == "doctor"]
    timing = URGENCY_MEANING[urgency]

    if consult_type == "doctor":
        lead = (
            f"A doctor should be consulted — {timing}. "
            f"{len(doctor_items)} finding(s) need a prescribing or diagnostic decision"
        )
        if specialties:
            names = ", ".join(s["specialty"] for s in specialties)
            lead += f", best suited to: {names}"
        lead += "."
        if pharmacist_items:
            lead += (
                f" A pharmacist can separately resolve {len(pharmacist_items)} other "
                "finding(s) without an appointment, and is worth contacting first since "
                "they are usually reachable the same day."
            )
    else:
        lead = (
            f"A pharmacist should be consulted — {timing}. "
            f"{len(pharmacist_items)} finding(s) concern the medicines on file — how they "
            "combine, overlap, or were recorded — which a pharmacist can resolve directly, "
            "without an appointment. Nothing found here requires a prescribing or "
            "diagnostic decision, so a doctor's appointment is not indicated by these "
            "checks alone."
        )

    low_confidence = [i for i in items if i["confidence"] <= LOW_CONFIDENCE_THRESHOLD]
    if low_confidence:
        lead += (
            f" Note that {len(low_confidence)} of these finding(s) came from documents "
            "that were hard to read — bring the original documents along so they can be "
            "checked directly."
        )
    if quality_items:
        lead += (
            f" Separately, {len(quality_items)} document(s) could not be read with full "
            "confidence — worth confirming against the original paperwork while you are "
            "there, though not a reason to consult anyone on its own."
        )
    return lead


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------

def triage_consultation(
    cross_check: Dict[str, Any],
    lab_trends: Optional[Dict[str, Any]] = None,
    timeline: Optional[Dict[str, Any]] = None,
    model: str = MODEL,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Routes the already-computed safety findings to the professional who can
    act on them.

    Args:
        cross_check: output of medical_extractor.cross_check_prescriptions().
        lab_trends:  output of lab_trends.track_lab_trends(). Optional — omit
                     it and lab-driven referrals are simply not produced.
        timeline:    output of medical_extractor.build_patient_timeline(),
                     used only to spot documents whose extraction was too poor
                     to trust. Optional.
        use_llm:     set False to run fully deterministically (no API call);
                     specialties then come from the rule map or default to a
                     general practitioner.

    Returns:
      {
        "consult_needed": bool,           # false == no trigger found, NOT "you are fine"
        "consult_type": "doctor" | "pharmacist" | None,
        "urgency": "routine" | "soon" | "urgent" | None,
        "confidence": float,              # in the referral, not in any diagnosis
        "recommended_specialties": [
            {"specialty", "reason", "confidence", "basis", "urgency", "triggered_by"}
        ],
        "pharmacist_actions": [item, ...],  # resolvable without an appointment
        "doctor_actions": [item, ...],      # need a prescribing/diagnostic decision
        "referral_items": [item, ...],      # all of the above, most urgent first
        "document_quality_notices": [item, ...],  # about the SCANS, not the patient
        "document_quality_note": str | None,      # one line covering the above
        "summary": str,
        "emergency_advice": str,
        "note": str,
      }

    Note that `consult_needed` is decided by CLINICAL findings alone. A
    document that scanned badly is reported in `document_quality_notices` and
    never sets `consult_needed`, `consult_type` or `urgency` — see
    DATA_QUALITY_TRIGGERS for why.
    """
    all_items = _items_from_cross_check(cross_check or {})
    all_items += _items_from_lab_trends(lab_trends or {})
    all_items += _items_from_extraction_quality(timeline or {})

    # The split that keeps "your scan was blurry" out of "you need to see
    # someone". Everything below this line, up to the returned dict, reasons
    # about `items` — the clinical findings — only.
    items = [i for i in all_items if i.get("category") != "data_quality"]
    quality_items = [i for i in all_items if i.get("category") == "data_quality"]

    _assign_specialties(items, model=model, use_llm=use_llm)

    # Most urgent first, then most confident, so the first line a reader sees
    # is the one that matters most.
    def sort_key(item: Dict[str, Any]) -> Tuple[int, float]:
        return (-URGENCY_ORDER[item["urgency"]], -item["confidence"])

    items.sort(key=sort_key)
    quality_items.sort(key=sort_key)

    doctor_items = [i for i in items if i["route"] == "doctor"]
    pharmacist_items = [i for i in items if i["route"] == "pharmacist"]

    consult_type: Optional[str] = None
    urgency: Optional[str] = None
    confidence = 0.0

    if items:
        consult_type = max((i["route"] for i in items), key=lambda r: ROUTE_ORDER[r])
        urgency = max((i["urgency"] for i in items), key=lambda u: URGENCY_ORDER[u])
        # Confidence in the REFERRAL is the confidence of the single strongest
        # reason to make it — deliberately max(), not an average. Averaging
        # would let a pile of weak findings dilute one certain allergy
        # conflict, which is exactly backwards: a referral is justified by its
        # best reason, not by its typical one.
        confidence = max(i["confidence"] for i in items)

    specialties = _collect_specialties(items)

    # One line, plain, imperative — what to do and by when. Everything else in
    # this result is detail behind it.
    if not items and quality_items:
        # Distinct from the fully clean headline below, because there IS
        # something to do — just not with a clinician.
        headline = "Nothing to act on — a document needs re-checking."
    elif not items:
        headline = "Nothing to act on right now."
    elif consult_type == "doctor":
        who = specialties[0]["specialty"].lower() if specialties else "a doctor"
        if specialties and specialties[0]["specialty"] == GENERAL_PRACTITIONER:
            who = "your regular doctor"
        headline = f"See {who} {URGENCY_WHEN[urgency]}."
    else:
        headline = f"Speak to a pharmacist {URGENCY_WHEN[urgency]}."

    # Explains an empty specialty list when a doctor IS still recommended, so
    # a caller can tell "we didn't name one because none is warranted" apart
    # from "the specialty step failed".
    specialty_note = None
    if doctor_items and not specialties:
        specialty_note = (
            "No particular kind of doctor is suggested: nothing here is a safety "
            "problem or an out-of-range result, so a general appointment is the right "
            "place to raise it."
        )

    # One line for the document-quality stream, phrased so it cannot be read
    # as a clinical instruction. Kept out of `summary` when there is nothing
    # clinical to say, so a caller can render it in its own, quieter place.
    document_quality_note = None
    if quality_items:
        document_quality_note = (
            f"{len(quality_items)} uploaded document(s) could not be read with full "
            "confidence, so the medicines and results taken from them are worth "
            "checking against the original paperwork. This describes the document, "
            "not the patient — on its own it is not a reason to see anyone."
        )

    return {
        "consult_needed": bool(items),
        "headline": headline,
        "consult_type": consult_type,
        "urgency": urgency,
        "urgency_meaning": URGENCY_MEANING[urgency] if urgency else None,
        "confidence": round(confidence, 2),
        "recommended_specialties": specialties,
        "specialty_note": specialty_note,
        "pharmacist_actions": pharmacist_items,
        "doctor_actions": doctor_items,
        "referral_items": items,
        "document_quality_notices": quality_items,
        "document_quality_note": document_quality_note,
        "summary": _summarize(items, consult_type, urgency, specialties, quality_items),
        "emergency_advice": EMERGENCY_ADVICE,
        "note": DISCLAIMER,
    }


if __name__ == "__main__":
    # Self-test. use_llm=False throughout so this runs offline and for free —
    # every assertion below is about the deterministic routing, which is the
    # part that decides whether someone is told to see a doctor.

    # --- Case 1: pharmacist-only findings must NOT escalate to a doctor -----
    pharmacist_only = triage_consultation(
        cross_check={
            "potential_drug_interactions": [
                {"medications_involved": ["Ibuprofen", "Aspirin"],
                 "explanation": "Both are NSAIDs; combined use raises GI bleeding risk.",
                 "severity": "moderate", "confidence": 0.8},
            ],
            "duplicate_prescriptions": [
                {"medication": "Paracetamol", "occurrences": [], "confidence": 0.95,
                 "explanation": "Same ingredient on two prescriptions under different brands."},
            ],
            "conflicting_dosage_instructions": [],
            "allergy_conflicts": [],
            "overall_recommendation": "Speak to a pharmacist.",
        },
        use_llm=False,
    )
    assert pharmacist_only["consult_needed"] is True
    assert pharmacist_only["consult_type"] == "pharmacist", pharmacist_only["consult_type"]
    assert pharmacist_only["urgency"] == "soon", pharmacist_only["urgency"]
    assert pharmacist_only["doctor_actions"] == []
    assert pharmacist_only["recommended_specialties"] == []
    assert len(pharmacist_only["pharmacist_actions"]) == 2

    # --- Case 2: an allergy conflict must escalate to an urgent doctor -----
    allergy = triage_consultation(
        cross_check={
            "potential_drug_interactions": [],
            "duplicate_prescriptions": [],
            "conflicting_dosage_instructions": [],
            "allergy_conflicts": [
                {"medication": "Amoxicillin", "allergy": "Penicillin",
                 "explanation": "Amoxicillin is a penicillin-class antibiotic.",
                 "confidence": 0.93},
            ],
        },
        use_llm=False,
    )
    assert allergy["consult_type"] == "doctor"
    assert allergy["urgency"] == "urgent"
    assert allergy["confidence"] == 0.93
    # With no rule match and the LLM disabled, it must still name someone.
    assert allergy["recommended_specialties"][0]["specialty"] == GENERAL_PRACTITIONER
    assert allergy["recommended_specialties"][0]["basis"] == "default"

    # --- Case 3: lab specialties resolve from the rule map, no LLM ---------
    labs = triage_consultation(
        cross_check={},
        lab_trends={
            "trends": [
                {"test_name": "ALT", "confidence": 0.95,
                 "explanation": "ALT has risen across 3 tests.",
                 "crossed_into_abnormal_at": {"date": "30 Aug 2026", "flag": "high"},
                 "approaching_threshold": False,
                 "data_points": [{"flag": "normal"}, {"flag": "normal"}, {"flag": "high"}]},
                {"test_name": "Creatinine", "confidence": 0.9,
                 "explanation": "Creatinine is climbing but still in range.",
                 "crossed_into_abnormal_at": None, "approaching_threshold": True,
                 "data_points": [{"flag": "normal"}, {"flag": "normal"}]},
                {"test_name": "Vitamin B12", "confidence": 0.9,
                 "explanation": "Stable and in range.",
                 "crossed_into_abnormal_at": None, "approaching_threshold": False,
                 "data_points": [{"flag": "normal"}, {"flag": "normal"}]},
            ],
            "insufficient_data": [],
        },
        use_llm=False,
    )
    assert labs["consult_type"] == "doctor"
    assert labs["urgency"] == "soon"          # the ALT crossing, not the Creatinine drift
    by_specialty = {s["specialty"]: s for s in labs["recommended_specialties"]}
    # ALT actually crossed out of range — a real finding, so a specialty is named.
    assert "Liver specialist" in by_specialty, by_specialty
    liver = by_specialty["Liver specialist"]
    assert liver["basis"] == "rule"
    # Plain name on screen, clinical term kept alongside for booking.
    assert liver["clinical_name"] == "Hepatology / Gastroenterology"
    # One short line, addressed to the patient.
    assert liver["reason"] == "Your ALT test checks your liver.", liver["reason"]
    assert len(liver["reason"].split()) <= 12
    # Creatinine is still INSIDE its range, merely drifting toward the edge. It
    # stays on the referral list, but naming a nephrologist for a normal result
    # overstates it — see _warrants_specialty().
    assert "Kidney specialist" not in by_specialty, by_specialty
    drift_item = next(i for i in labs["referral_items"]
                      if i["trigger"] == "lab_approaching_threshold")
    assert drift_item["specialty"] is None
    assert drift_item["specialty_omitted_because"]
    # The stable in-range test must produce no referral at all.
    assert all(i["subject"] != "Vitamin B12" for i in labs["referral_items"])
    # Most urgent first: the crossing outranks the drift.
    assert labs["referral_items"][0]["lab_test"] == "ALT"

    # Word-boundary guard: "ALP"/"ALT" keys must not fire on unrelated names.
    assert _match_lab_specialty("Cobalt") is None
    assert _match_lab_specialty("Alkaline Phosphatase")[0] == "Liver specialist"

    # --- Case 4: low confidence lowers the score but NEVER the urgency -----
    low_conf = triage_consultation(
        cross_check={
            "allergy_conflicts": [
                {"medication": "Amoxicillin", "allergy": "Penicillin",
                 "explanation": "Handwriting was barely legible.", "confidence": 0.35},
            ],
        },
        use_llm=False,
    )
    assert low_conf["urgency"] == "urgent", "low confidence must not de-escalate urgency"
    assert low_conf["confidence"] == 0.35
    assert "confidence_caveat" in low_conf["referral_items"][0]

    # --- Case 5: confidence is the strongest reason, not the average -------
    mixed = triage_consultation(
        cross_check={
            "duplicate_prescriptions": [
                {"medication": "Paracetamol", "occurrences": [], "confidence": 0.95,
                 "explanation": "Exact ingredient + dose match across two documents."},
                {"medication": "Vitamin C", "occurrences": [], "confidence": 0.2,
                 "explanation": "Weak signal."},
                {"medication": "Zinc", "occurrences": [], "confidence": 0.2,
                 "explanation": "Weak signal."},
            ],
        },
        use_llm=False,
    )
    assert mixed["confidence"] == 0.95, "a pile of weak findings must not dilute a certain one"

    # --- Case 6: nothing found must not read as a clean bill of health -----
    clean = triage_consultation(cross_check={}, lab_trends={}, timeline={}, use_llm=False)
    assert clean["consult_needed"] is False
    assert clean["consult_type"] is None
    assert clean["confidence"] == 0.0
    assert "NOT a clean bill of health" in clean["summary"]
    assert clean["emergency_advice"]

    # --- Case 7: unreadable documents are a DOCUMENT notice, not a referral -
    # They are reported in full, but they must not tell the patient to go and
    # see someone: nothing was found about the patient, only about the scan.
    poor_scan = triage_consultation(
        cross_check={},
        timeline={"visits": [
            {"_source": {"file": "scan1.jpg"}, "overall_confidence": 0.4,
             "illegible_or_low_confidence_fields": ["medications[0].dosage"]},
            {"_source": {"file": "clean.pdf"}, "overall_confidence": 0.95,
             "illegible_or_low_confidence_fields": []},
        ]},
        use_llm=False,
    )
    assert poor_scan["referral_items"] == [], poor_scan["referral_items"]
    assert poor_scan["pharmacist_actions"] == []
    assert poor_scan["consult_needed"] is False
    assert poor_scan["consult_type"] is None
    assert poor_scan["urgency"] is None
    assert len(poor_scan["document_quality_notices"]) == 1
    notice = poor_scan["document_quality_notices"][0]
    assert notice["subject"] == "scan1.jpg"
    assert notice["trigger"] == "low_extraction_confidence"
    assert notice["category"] == "data_quality"
    assert poor_scan["document_quality_note"]
    assert poor_scan["headline"] == "Nothing to act on — a document needs re-checking."
    # Still not a clean bill of health, and the note explains itself.
    assert "NOT a clean bill of health" in poor_scan["summary"]
    assert "not a finding about the patient" in poor_scan["summary"]

    # --- Case 7a: a well-read document must produce nothing at all ----------
    # This is the regression that motivated the split. The extractor records
    # interpretation choices in the same field it records unreadable ones, so
    # a confident extraction can still arrive with a non-empty list. That is
    # transparency, not a problem, and it used to produce "Speak to a
    # pharmacist" on a document with no clinical findings whatsoever.
    well_read = triage_consultation(
        cross_check={},
        timeline={"visits": [
            {"_source": {"file": "clear_prescription.png"}, "overall_confidence": 0.85,
             "illegible_or_low_confidence_fields": [
                 "dosage_value/dosage_unit for 'Calcium carbonate + Vitamin D3' reduced "
                 "to 500 mg (document shows '500 mg / 250 IU' combination)",
                 "quantity fields were read from printed table but not repeated "
                 "individually in this extraction (visible on document)",
             ]},
        ]},
        use_llm=False,
    )
    assert well_read["consult_needed"] is False, well_read["referral_items"]
    assert well_read["consult_type"] is None
    assert well_read["referral_items"] == []
    assert well_read["document_quality_notices"] == []
    assert well_read["headline"] == "Nothing to act on right now."

    # A material field that genuinely could not be read on a middling document
    # still surfaces — the trust threshold must not swallow real problems.
    borderline = triage_consultation(
        cross_check={},
        timeline={"visits": [
            {"_source": {"file": "smudged.jpg"}, "overall_confidence": 0.7,
             "illegible_or_low_confidence_fields": ["dosage for Metformin was smudged"]},
        ]},
        use_llm=False,
    )
    assert len(borderline["document_quality_notices"]) == 1, borderline
    assert borderline["consult_needed"] is False

    # An unreadable field that changes nothing about the medication list is
    # not worth reporting at all.
    immaterial = triage_consultation(
        cross_check={},
        timeline={"visits": [
            {"_source": {"file": "cropped.jpg"}, "overall_confidence": 0.7,
             "illegible_or_low_confidence_fields": ["clinic footer", "signature line"]},
        ]},
        use_llm=False,
    )
    assert immaterial["document_quality_notices"] == [], immaterial
    assert immaterial["consult_needed"] is False

    # --- Case 7b: a translated document is flagged separately from an
    #     unreadable one — they have different fixes ------------------------
    translated = triage_consultation(
        cross_check={},
        timeline={"visits": [
            {"_source": {"file": "japanese_rx.pdf"},
             "document_language": "Japanese", "additional_languages": [],
             "ocr_confidence": 0.9, "translation_confidence": 0.5,
             "overall_confidence": 0.9, "illegible_or_low_confidence_fields": []},
            # Clean English document: must produce nothing at all.
            {"_source": {"file": "clean_english.pdf"},
             "document_language": "English", "additional_languages": [],
             "ocr_confidence": 0.97, "translation_confidence": 1.0,
             "overall_confidence": 0.96, "illegible_or_low_confidence_fields": []},
        ]},
        use_llm=False,
    )
    # A "high" translation flag IS a referral: unlike a blurry scan, a
    # mistranslated drug name leaves a record that looks perfectly correct.
    assert len(translated["referral_items"]) == 1, translated["referral_items"]
    item = translated["referral_items"][0]
    assert item["trigger"] == "translation_uncertain", item
    assert item["category"] == "clinical", item
    assert item["subject"] == "japanese_rx.pdf"
    assert item["route"] == "pharmacist" and item["urgency"] == "soon"
    assert translated["consult_needed"] is True
    assert translated["consult_type"] == "pharmacist"
    # ocr 0.9 * translation 0.5 -> the combined figure, not either alone.
    assert item["confidence"] == 0.45, item["confidence"]

    # A "review" flag — neither axis individually low, the two merely compound
    # — stays a document note rather than sending anyone to a pharmacist.
    mild_translation = triage_consultation(
        cross_check={},
        timeline={"visits": [
            {"_source": {"file": "spanish_rx.pdf"},
             "document_language": "Spanish", "additional_languages": [],
             "ocr_confidence": 0.8, "translation_confidence": 0.8,
             "overall_confidence": 0.85, "illegible_or_low_confidence_fields": []},
        ]},
        use_llm=False,
    )
    mild_triggers = {i["trigger"] for i in mild_translation["document_quality_notices"]}
    assert mild_triggers == {"translation_uncertain"}, mild_translation
    assert mild_translation["consult_needed"] is False, mild_translation
    assert mild_translation["referral_items"] == []
    assert mild_translation["document_quality_notices"][0]["category"] == "data_quality"

    # A blurry ENGLISH document must be routed as unreadable, never as a
    # translation problem — telling someone to confirm a translation that
    # never happened would be useless advice.
    blurry_english = triage_consultation(
        cross_check={},
        timeline={"visits": [
            {"_source": {"file": "blurry.jpg"},
             "document_language": "English", "additional_languages": [],
             "ocr_confidence": 0.35, "translation_confidence": 1.0,
             "overall_confidence": 0.4, "illegible_or_low_confidence_fields": ["dosage"]},
        ]},
        use_llm=False,
    )
    triggers = {i["trigger"] for i in blurry_english["document_quality_notices"]}
    assert triggers == {"low_extraction_confidence"}, blurry_english
    assert blurry_english["referral_items"] == [], blurry_english["referral_items"]

    # --- Case 7c: timing separates live risks from historical pairings -----
    timed = triage_consultation(
        cross_check={
            "potential_drug_interactions": [
                {"medications_involved": ["Paracetamol", "Diclofenac"],
                 "explanation": "Additive GI risk.", "severity": "moderate",
                 "confidence": 0.6,
                 "timing": {"status": "concurrent", "window_start": "2025-11-09",
                            "window_end": "2025-11-23", "overlap_days": 15,
                            "gap_days": 0, "note": "…"}},
                {"medications_involved": ["Cetirizine", "Chlorpheniramine"],
                 "explanation": "Additive sedation.", "severity": "moderate",
                 "confidence": 0.6,
                 "timing": {"status": "not_concurrent", "window_start": None,
                            "window_end": None, "overlap_days": 0,
                            "gap_days": 861, "note": "…"}},
            ],
        },
        use_llm=False,
    )
    live = [i for i in timed["referral_items"] if not i.get("is_historical")]
    past = [i for i in timed["referral_items"] if i.get("is_historical")]
    assert len(live) == 1 and len(past) == 1, timed["referral_items"]
    # The concurrent one keeps its urgency and gains its dates.
    assert live[0]["urgency"] == "soon"
    assert "2025-11-09 to 2025-11-23" in live[0]["detail"], live[0]["detail"]
    # The 861-days-apart one is kept, but de-escalated rather than dropped.
    assert past[0]["urgency"] == "routine", past[0]
    assert "861" in past[0]["why_this_route"]
    assert timed["urgency"] == "soon", "a stale pairing must not set overall urgency"

    # --- Case 7d: concurrent double-dosing is its own, higher trigger ------
    double = triage_consultation(
        cross_check={"concurrent_exposure": [
            {"ingredient": "paracetamol", "status": "concurrent",
             "window_start": "2025-11-12", "window_end": "2025-11-22",
             "overlap_days": 11, "cumulative_daily_dose": 5000.0,
             "dosage_unit": "mg", "sources": [],
             "note": "Between 2025-11-12 and 2025-11-22, two separate prescriptions "
                     "supplied paracetamol."},
        ]},
        use_llm=False,
    )
    assert len(double["referral_items"]) == 1
    exposure_item = double["referral_items"][0]
    assert exposure_item["trigger"] == "concurrent_double_dose"
    assert exposure_item["urgency"] == "urgent", exposure_item
    assert exposure_item["route"] == "pharmacist"
    assert "5000.0 mg per day" in exposure_item["detail"], exposure_item["detail"]

    # --- Case 7e: a specialty is named only when something warrants it -----
    # A lab still inside its normal range, merely drifting toward the edge.
    # Worth raising at a routine appointment; not worth naming a specialist.
    drifting_only = triage_consultation(
        cross_check={},
        lab_trends={"trends": [
            {"test_name": "LDL Cholesterol", "confidence": 0.94,
             "explanation": "Still in range but climbing.",
             "crossed_into_abnormal_at": None, "approaching_threshold": True,
             "data_points": [{"flag": "normal"}, {"flag": "normal"}]},
        ]},
        use_llm=False,
    )
    assert drifting_only["consult_needed"] is True
    assert drifting_only["consult_type"] == "doctor"
    assert drifting_only["urgency"] == "routine"
    assert drifting_only["recommended_specialties"] == [], drifting_only["recommended_specialties"]
    assert drifting_only["specialty_note"], "an empty list needs explaining"
    assert drifting_only["referral_items"][0]["specialty"] is None
    assert "not outside the normal range" in drifting_only["referral_items"][0][
        "specialty_omitted_because"].replace("nothing here is outside", "not outside")

    # The same test once it has actually crossed IS an alert, and does get one.
    crossed = triage_consultation(
        cross_check={},
        lab_trends={"trends": [
            {"test_name": "LDL Cholesterol", "confidence": 0.94,
             "explanation": "Crossed high.",
             "crossed_into_abnormal_at": {"date": "2026-03-01", "flag": "high"},
             "approaching_threshold": False,
             "data_points": [{"flag": "normal"}, {"flag": "high"}]},
        ]},
        use_llm=False,
    )
    assert [s["specialty"] for s in crossed["recommended_specialties"]] == [
        "Heart specialist"], crossed["recommended_specialties"]

    # --- Every patient-facing string stays short and jargon-free -----------
    for result in (crossed, labs, allergy, pharmacist_only):
        assert result["headline"], result
        assert len(result["headline"]) <= 70, result["headline"]
        for spec in result["recommended_specialties"]:
            assert len(spec["reason"].split()) <= 18, spec["reason"]
            for jargon in ("Hepatology", "Nephrology", "Cardiology", "Endocrinology",
                           "Haematology", "Urology", "Rheumatology",
                           "General practitioner (family"):
                assert jargon not in spec["specialty"], spec["specialty"]

    assert crossed["headline"] == "See heart specialist in the next few days.", crossed["headline"]
    assert pharmacist_only["headline"] == "Speak to a pharmacist in the next few days."
    assert allergy["headline"] == "See your regular doctor today or tomorrow.", allergy["headline"]
    assert clean["headline"] == "Nothing to act on right now."

    # A historical pairing names no specialty, however severe it sounds —
    # courses that never overlapped leave nothing for a specialist to act on.
    historical_only = triage_consultation(
        cross_check={"potential_drug_interactions": [
            {"medications_involved": ["Cetirizine", "Chlorpheniramine"],
             "explanation": "Additive sedation.", "severity": "high", "confidence": 0.6,
             "timing": {"status": "not_concurrent", "window_start": None,
                        "window_end": None, "overlap_days": 0, "gap_days": 861,
                        "note": "…"}},
        ]},
        use_llm=False,
    )
    assert historical_only["recommended_specialties"] == [], historical_only
    assert historical_only["referral_items"][0]["is_historical"] is True

    # A genuine, current interaction still gets one.
    live_alert = triage_consultation(
        cross_check={"potential_drug_interactions": [
            {"medications_involved": ["Warfarin", "Amiodarone"],
             "explanation": "Bleeding risk.", "severity": "high", "confidence": 0.8,
             "timing": {"status": "concurrent", "window_start": "2026-01-01",
                        "window_end": "2026-01-14", "overlap_days": 14,
                        "gap_days": 0, "note": "…"}},
        ]},
        use_llm=False,
    )
    assert len(live_alert["recommended_specialties"]) == 1, live_alert
    assert live_alert["specialty_note"] is None

    # Nothing found at all -> no doctor, no specialty (unchanged behaviour).
    nothing = triage_consultation(cross_check={}, use_llm=False)
    assert nothing["recommended_specialties"] == []
    assert nothing["specialty_note"] is None

    # --- Case 7f: a guideline-backed combination is cited, not capped ------
    guideline = triage_consultation(
        cross_check={"guideline_flagged_combinations": [
            {"opioid": "Oxycodone", "depressant": "Diazepam", "status": "concurrent",
             "window_start": "2025-11-12", "window_end": "2025-11-22",
             "overlap_days": 11, "severity": "high",
             "plain": "Taking a strong painkiller together with a sedative can "
                      "dangerously slow your breathing.",
             "citation": {"source": "SAMHSA Overdose Prevention and Response Toolkit",
                          "page": 13, "publication_no": "PEP23-03-00-001"}},
        ]},
        use_llm=False,
    )
    assert len(guideline["referral_items"]) == 1
    combo_item = guideline["referral_items"][0]
    assert combo_item["trigger"] == "guideline_flagged_combination"
    assert combo_item["route"] == "doctor" and combo_item["urgency"] == "urgent"
    assert combo_item["confidence"] == 0.9, "a cited claim is not capped at 0.6"
    assert combo_item["reference"]["page"] == 13
    assert "2025-11-12 to 2025-11-22" in combo_item["detail"], combo_item["detail"]
    # Dates must also be structured, not only in the prose — a client filtering
    # by `timing` should not miss the most firmly dated finding in the report.
    assert combo_item["timing"]["status"] == "concurrent", combo_item["timing"]
    assert combo_item["timing"]["window_start"] == "2025-11-12"
    assert combo_item["timing"]["overlap_days"] == 11
    # It is a genuine alert, so naming a kind of doctor IS warranted.
    assert guideline["recommended_specialties"], guideline

    # --- Case 8: an unrecognised severity must not silently vanish ---------
    odd_severity = triage_consultation(
        cross_check={"potential_drug_interactions": [
            {"medications_involved": ["A", "B"], "explanation": "…",
             "severity": "catastrophic", "confidence": 0.7},
        ]},
        use_llm=False,
    )
    assert len(odd_severity["referral_items"]) == 1
    assert odd_severity["referral_items"][0]["severity"] == "moderate"

    for case_name, result in [
        ("pharmacist-only", pharmacist_only), ("allergy", allergy), ("labs", labs),
    ]:
        print(f"--- {case_name} ---")
        print(f"  consult_needed={result['consult_needed']} "
              f"type={result['consult_type']} urgency={result['urgency']} "
              f"confidence={result['confidence']}")
        for s in result["recommended_specialties"]:
            print(f"  specialty: {s['specialty']} ({s['basis']}, {s['confidence']})")
        print(f"  {result['summary']}")
        print()

    print("All checks passed.")
