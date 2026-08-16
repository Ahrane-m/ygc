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

LAB_SPECIALTY_RULES: List[Tuple[Tuple[str, ...], str, str]] = [
    (("alt", "ast", "sgpt", "sgot", "bilirubin", "alkaline phosphatase", "ggt",
      "alp"),
     "Hepatology (or Gastroenterology)",
     "these are liver-function tests"),
    (("creatinine", "egfr", "gfr", "urea", "bun"),
     "Nephrology",
     "these are kidney-function tests"),
    (("glucose", "hba1c", "a1c", "insulin"),
     "Endocrinology (diabetes care)",
     "these are blood-sugar tests"),
    (("tsh", "t3", "t4", "thyroid"),
     "Endocrinology",
     "these are thyroid-function tests"),
    (("cholesterol", "ldl", "hdl", "triglyceride", "lipid"),
     "Cardiology (or a lipid clinic)",
     "these are lipid/cardiovascular-risk tests"),
    (("hemoglobin", "haemoglobin", "hematocrit", "haematocrit", "platelet",
      "wbc", "rbc", "mcv", "white blood cell", "red blood cell"),
     "Hematology",
     "these are blood-count tests"),
    (("inr", "prothrombin", "aptt", "ptt"),
     "Hematology",
     "these are blood-clotting tests"),
    (("uric acid", "urate"),
     "Rheumatology",
     "raised uric acid is usually managed as a rheumatological problem"),
    (("psa",),
     "Urology",
     "PSA is a prostate test"),
    (("troponin", "bnp", "nt-probnp"),
     "Cardiology",
     "these are cardiac tests"),
]

GENERAL_PRACTITIONER = "General practitioner (family doctor)"

GP_FIRST_REASON = (
    "A general practitioner is the right first contact — they hold the whole record, "
    "can treat this directly if it is straightforward, and can refer on to a "
    "specialist if it is not. Going straight to a specialist is rarely necessary and "
    "usually slower."
)


def _match_lab_specialty(test_name: str) -> Optional[Tuple[str, str]]:
    """Returns (specialty, reason_fragment) for a test name the rule table
    covers, else None — in which case the LLM pass gets a chance at it."""
    name = (test_name or "").lower()
    for keywords, specialty, reason in LAB_SPECIALTY_RULES:
        for kw in keywords:
            # Word-boundary match so a short key like "alt" doesn't fire on
            # "Alkaline Phosphatase" or "Cobalt".
            if re.search(rf"\b{re.escape(kw)}\b", name):
                return specialty, reason
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
        items.append(_make_item(
            "drug_interaction",
            subject=" + ".join(involved) if involved else "unnamed medications",
            detail=finding.get("explanation") or "",
            confidence=_clamp_confidence(finding.get("confidence")),
            severity=severity,
        ))

    for finding in cross_check.get("duplicate_prescriptions") or []:
        items.append(_make_item(
            "duplicate_prescription",
            subject=finding.get("medication") or "unnamed medication",
            detail=finding.get("explanation") or "",
            confidence=_clamp_confidence(finding.get("confidence")),
        ))

    for finding in cross_check.get("conflicting_dosage_instructions") or []:
        items.append(_make_item(
            "dosage_conflict",
            subject=finding.get("medication") or "unnamed medication",
            detail=finding.get("explanation") or "",
            confidence=_clamp_confidence(finding.get("confidence")),
        ))

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

    return items


def _items_from_extraction_quality(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One item per document whose extraction was poor enough that the
    medication list built from it may not be trustworthy. This is a
    data-quality referral, not a clinical one, and is labelled as such.

    Two distinct problems are routed separately, because they have different
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
            items.append(_make_item(
                "translation_uncertain",
                subject=source,
                detail=risk["message"] or "",
                confidence=_clamp_confidence(
                    risk["effective_confidence"]
                    if risk["effective_confidence"] is not None
                    else translation_confidence,
                    default=0.7,
                ),
            ))

        overall = visit.get("overall_confidence")
        illegible = visit.get("illegible_or_low_confidence_fields") or []

        low_overall = isinstance(overall, (int, float)) and overall <= LOW_CONFIDENCE_THRESHOLD
        if not low_overall and not illegible:
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

Rules:
- Prefer a general practitioner / family doctor unless the finding clearly
  belongs to one specialty. Most findings do not need a specialist, and a GP
  is faster to reach, holds the whole record, and can refer onward. Naming a
  specialist unnecessarily sends the patient down a slower path.
- Name a recognised specialty in plain English ("Nephrology", "Cardiology",
  "General practitioner (family doctor)"). Never name an individual, a
  hospital, or a clinic.
- reason must say what about THIS finding points to that specialty, in one
  sentence a patient can follow. Do not explain what the finding means
  clinically and do not suggest what the treatment might be — you are routing,
  not diagnosing.
- Do not state or imply a diagnosis. "Liver-function tests are handled by
  hepatology" is routing; "this suggests liver disease" is a diagnosis and is
  not allowed.

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
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["id", "specialty", "reason", "confidence"],
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

        if item.get("lab_test"):
            matched = _match_lab_specialty(item["lab_test"])
            if matched:
                specialty, reason = matched
                item["specialty"] = {
                    "specialty": specialty,
                    "reason": (
                        f"{item['lab_test']} is usually handled by {specialty} because "
                        f"{reason}. A general practitioner is still a valid first "
                        "contact and can refer on."
                    ),
                    "confidence": 0.9,
                    "basis": "rule",
                }
                continue
        pending.append((idx, item))

    assigned = _suggest_specialties_llm(pending, model=model) if use_llm else {}

    for idx, item in pending:
        suggestion = assigned.get(idx)
        if suggestion:
            item["specialty"] = {**suggestion, "basis": "model"}
        else:
            item["specialty"] = {
                "specialty": GENERAL_PRACTITIONER,
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
) -> str:
    """Plain-language summary, filled from the fields computed above the same
    way lab_trends._explain() does — so it cannot say anything the routing
    does not already support."""
    if not items:
        return (
            "No automated trigger for a consultation was found: the safety cross-check "
            "reported no interactions, duplicates, dosage conflicts or allergy conflicts, "
            "and no lab test on file has drifted out of its reference range. This is NOT "
            "a clean bill of health — it only means these specific checks found nothing "
            "in the documents provided. Anyone with symptoms, questions, or a scheduled "
            "review should still see their doctor or pharmacist as normal."
        )

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
            f"{len(pharmacist_items)} finding(s) relate to how the medications on file "
            "fit together, which a pharmacist can resolve directly, without an "
            "appointment. Nothing found here requires a prescribing or diagnostic "
            "decision, so a doctor's appointment is not indicated by these checks alone."
        )

    low_confidence = [i for i in items if i["confidence"] <= LOW_CONFIDENCE_THRESHOLD]
    if low_confidence:
        lead += (
            f" Note that {len(low_confidence)} of these finding(s) came from documents "
            "that were hard to read — bring the original documents along so they can be "
            "checked directly."
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
        "summary": str,
        "emergency_advice": str,
        "note": str,
      }
    """
    items = _items_from_cross_check(cross_check or {})
    items += _items_from_lab_trends(lab_trends or {})
    items += _items_from_extraction_quality(timeline or {})

    _assign_specialties(items, model=model, use_llm=use_llm)

    # Most urgent first, then most confident, so the first line a reader sees
    # is the one that matters most.
    items.sort(key=lambda i: (-URGENCY_ORDER[i["urgency"]], -i["confidence"]))

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

    return {
        "consult_needed": bool(items),
        "consult_type": consult_type,
        "urgency": urgency,
        "urgency_meaning": URGENCY_MEANING[urgency] if urgency else None,
        "confidence": round(confidence, 2),
        "recommended_specialties": specialties,
        "pharmacist_actions": pharmacist_items,
        "doctor_actions": doctor_items,
        "referral_items": items,
        "summary": _summarize(items, consult_type, urgency, specialties),
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
    assert "Hepatology (or Gastroenterology)" in by_specialty, by_specialty
    assert "Nephrology" in by_specialty, by_specialty
    assert by_specialty["Hepatology (or Gastroenterology)"]["basis"] == "rule"
    # The stable in-range test must produce no referral at all.
    assert all(i["subject"] != "Vitamin B12" for i in labs["referral_items"])
    # Most urgent first: the crossing outranks the drift.
    assert labs["referral_items"][0]["lab_test"] == "ALT"

    # Word-boundary guard: "ALP"/"ALT" keys must not fire on unrelated names.
    assert _match_lab_specialty("Cobalt") is None
    assert _match_lab_specialty("Alkaline Phosphatase")[0].startswith("Hepatology")

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

    # --- Case 7: unreadable documents produce a data-quality referral ------
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
    assert len(poor_scan["referral_items"]) == 1
    assert poor_scan["referral_items"][0]["subject"] == "scan1.jpg"
    assert poor_scan["consult_type"] == "pharmacist"

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
    assert len(translated["referral_items"]) == 1, translated["referral_items"]
    item = translated["referral_items"][0]
    assert item["trigger"] == "translation_uncertain", item
    assert item["subject"] == "japanese_rx.pdf"
    assert item["route"] == "pharmacist" and item["urgency"] == "soon"
    # ocr 0.9 * translation 0.5 -> the combined figure, not either alone.
    assert item["confidence"] == 0.45, item["confidence"]

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
    triggers = {i["trigger"] for i in blurry_english["referral_items"]}
    assert triggers == {"low_extraction_confidence"}, triggers

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
