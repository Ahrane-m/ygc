"""
Same-Prescription Detection (document-level de-duplication)
=========================================
Recognises when two DIFFERENT files in a patient's record are actually the
same physical prescription — a scan and a phone photo of one piece of paper,
the same document re-sent from another device, a re-upload after a rename.

WHY THIS EXISTS (an observed, clinically-consequential false alarm)
------------------------------------------------------------------
A real record held one osteoarthritis prescription three times: once as
`08_09-11-2025_osteoarthritis.png` and twice as a WhatsApp photo of the same
page. Nothing was wrong with any of them individually — but every medication
on that prescription then appeared under several different
(date, source_file) pairs, and that is precisely the shape both
detect_exact_duplicate_medications() and the cross-check LLM read as "this
drug was prescribed more than once". The result:

    duplicate_prescriptions : Paracetamol, Omeprazole, Glucosamine, Diclofenac
    consult_triage          : pharmacist, soon, confidence 0.95

The patient was being told to see a pharmacist about a double-dosing risk
that did not exist. It was an artefact of how the documents were ingested,
not a fact about their medication history — and it is the more dangerous
direction of error, because a false alarm that turns out to be nothing is
what teaches someone to ignore the next one.

This module does NOT delete or drop anything. Every document stays in the
timeline with its own provenance; they are merely tagged with a shared
`prescription_group` so downstream duplicate detection counts *prescriptions*
rather than *files*. Losing a genuine document would be a worse failure than
the false alarm this fixes.

WHAT COUNTS AS THE SAME PRESCRIPTION
------------------------------------
Same patient, same prescriber, same document type, the same set of
medications at the same normalized doses and frequencies, and compatible
dates. All of those must agree — it is deliberately a high bar, because
merging two genuinely separate prescriptions would hide a real duplicate.
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dateutil import parser as dateutil_parser

# Salt / ester suffixes stripped before comparing ingredients. The same
# prescription re-extracted from a different photo legitimately comes back as
# "Diclofenac" one time and "Diclofenac sodium" the next (both were observed
# on the same real document), and an exact string match would call those two
# different prescriptions.
#
# Stripping these is safe HERE and would not be elsewhere: this only decides
# whether two documents are the same piece of paper, and that decision also
# requires the patient, prescriber, every other medication, and every dose to
# match. Two genuinely different prescriptions that differ only by salt form
# is not a real scenario. Never reuse this for interaction checking, where
# "fluticasone propionate" vs "fluticasone furoate" is a real distinction.
SALT_SUFFIXES = frozenset({
    "sodium", "potassium", "calcium", "magnesium", "hydrochloride", "hcl",
    "sulfate", "sulphate", "maleate", "tartrate", "besylate", "mesylate",
    "citrate", "acetate", "phosphate", "nitrate", "succinate", "fumarate",
    "propionate", "furoate", "valerate", "dipropionate", "monohydrate",
    "trihydrate", "anhydrous", "micronized", "micronised",
})

# Parenthetical form notes ("(topical gel)", "(oral)") that the extractor
# sometimes folds into the printed name.
_PAREN_RE = re.compile(r"\([^)]*\)")
_NON_WORD_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize_text(value: Any) -> str:
    """Lowercase, strip punctuation and parentheticals, collapse whitespace."""
    if not isinstance(value, str):
        return ""
    text = _PAREN_RE.sub(" ", value.lower())
    text = _NON_WORD_RE.sub(" ", text)
    return " ".join(text.split())


def _base_ingredient(name: str) -> str:
    """Ingredient name with any trailing salt/ester words removed, so
    'Diclofenac sodium' and 'Diclofenac' compare equal."""
    words = _normalize_text(name).split()
    while len(words) > 1 and words[-1] in SALT_SUFFIXES:
        words.pop()
    return " ".join(words)


# Strength/dose tokens the extractor folds into a printed name:
# "NALOXONE 400mcg", "Amoxicillin 500 mg", "Insulin 100IU/mL".
_DOSE_TOKEN_RE = re.compile(
    r"^\d+(?:\.\d+)?(?:mg|mcg|ug|g|ml|l|iu|u|meq|mmol|%)?$|^\d+$", re.IGNORECASE
)
_UNIT_ONLY = frozenset({"mg", "mcg", "ug", "g", "ml", "l", "iu", "u", "meq",
                        "mmol", "tablet", "tablets", "capsule", "capsules"})


def name_variants(name: Any) -> List[str]:
    """
    Every spelling under which a printed drug name might be stored in the
    reference graph, most specific first.

    The graph keys :Medicine on a bare lowercased ingredient ("naloxone"),
    but prescriptions are printed as "Naloxone Hydrochloride 400mcg". An exact
    lookup on the printed name misses, and — because a miss is indistinguishable
    from a drug genuinely not being in the reference data — the miss is silent:
    the finding quietly drops to unverified model recall with no error anywhere.
    So callers query every variant and take the first hit.

    Deliberately conservative. Salts, doses and units are removed because they
    qualify a drug rather than identify it; the name is never truncated to its
    first word, since "insulin glargine" and "insulin" are different products
    and matching one to the other would be worse than missing.
    """
    raw = (name or "") if isinstance(name, str) else ""
    if not raw.strip():
        return []

    variants: List[str] = []

    def add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in variants:
            variants.append(candidate)

    add(raw.strip().lower())
    normalized = _normalize_text(raw)
    add(normalized)

    # Drop dose figures and bare units, then re-strip salts: "naloxone
    # hydrochloride 400 mcg" -> "naloxone".
    words = [w for w in normalized.split()
             if not _DOSE_TOKEN_RE.match(w) and w not in _UNIT_ONLY]
    while len(words) > 1 and words[-1] in SALT_SUFFIXES:
        words.pop()
    add(" ".join(words))
    add(_base_ingredient(raw))
    return [v for v in variants if v]


def plausible_dates(raw: Any) -> frozenset:
    """
    Every date this string could reasonably mean, as `date` objects.

    Returns a SET rather than one date because prescription dates in this
    pipeline are genuinely ambiguous: the same document was extracted once as
    "09/11/2025" and once as "2025-11-09". Read day-first, the former is
    9 November; read month-first it is 11 September. Committing to either
    reading would split one prescription into two — so both are kept, and two
    dates are treated as compatible when their sets overlap.
    """
    if not isinstance(raw, str) or not raw.strip():
        return frozenset()
    text = raw.strip()
    found = set()
    for dayfirst in (True, False):
        try:
            found.add(dateutil_parser.parse(text, fuzzy=True, dayfirst=dayfirst).date())
        except (ValueError, OverflowError, TypeError):
            continue
    return frozenset(found)


def dates_compatible(a: Any, b: Any) -> bool:
    """True if two printed dates could refer to the same day. A missing or
    unparseable date is compatible with anything — the rest of the
    fingerprint (patient, prescriber, exact drug set and doses) is strong
    enough on its own, and a document whose date failed to extract should not
    be split off into a phantom second prescription because of it."""
    set_a, set_b = plausible_dates(a), plausible_dates(b)
    if not set_a or not set_b:
        return True
    return bool(set_a & set_b)


def _medication_key(med: Dict[str, Any]) -> Tuple:
    """One medication reduced to what identifies it on a prescription."""
    ingredients = tuple(sorted(
        _base_ingredient(i) for i in (med.get("ingredients") or []) if i and i.strip()
    ))
    if not ingredients:
        ingredients = (_base_ingredient(med.get("name") or "unknown"),)
    return (
        ingredients,
        med.get("dosage_value"),
        _normalize_text(med.get("dosage_unit")) or None,
        med.get("frequency_per_day"),
        bool(med.get("is_as_needed")),
    )


def prescription_fingerprint(doc: Dict[str, Any]) -> Tuple:
    """
    The identity of the prescription a document records, ignoring how it was
    scanned, named or dated.

    The date is deliberately NOT part of this tuple — it is compared
    separately via dates_compatible(), because the same date reaches this
    code in several irreconcilable formats.
    """
    medications = tuple(sorted(
        _medication_key(m) for m in (doc.get("medications") or [])
    ))
    return (
        _normalize_text(doc.get("patient_name")),
        _normalize_text(doc.get("provider_or_doctor")),
        doc.get("document_type") or "unknown",
        medications,
    )


def _is_groupable(doc: Dict[str, Any]) -> bool:
    """Only documents with actual medication content are grouped. Two lab
    reports with no medications would otherwise share the empty fingerprint
    and be merged into one 'prescription', which they are not."""
    return bool(doc.get("medications"))


def annotate_prescription_groups(docs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Tags each document with a `prescription_group` id, shared by documents
    that record the same physical prescription. Returns the same dicts
    (mutated in place) for convenience.

    A document that can't be grouped — no medications, or simply unique —
    gets its own group, so downstream code can always treat
    `prescription_group` as "which prescription is this" without special
    cases.
    """
    groups: List[Dict[str, Any]] = []  # [{fingerprint, dates, id}]

    for index, doc in enumerate(docs):
        if not _is_groupable(doc):
            doc["prescription_group"] = f"doc-{index}"
            continue

        fingerprint = prescription_fingerprint(doc)
        date = doc.get("date")

        for group in groups:
            if group["fingerprint"] != fingerprint:
                continue
            if not any(dates_compatible(date, d) for d in group["dates"]):
                continue
            doc["prescription_group"] = group["id"]
            group["dates"].append(date)
            break
        else:
            group_id = f"rx-{len(groups)}"
            groups.append({"fingerprint": fingerprint, "dates": [date], "id": group_id})
            doc["prescription_group"] = group_id

    return list(docs)


def find_duplicate_document_groups(
    docs: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Reports which documents record the same prescription, for review tooling.

    Returns one entry per group that has MORE THAN ONE document:
      {"prescription_group", "documents": [{source_file, date, document_url,
       cloudinary_public_id, content_sha256, uploaded_at}], "identical_files":
       bool, "medications": [str]}

    `identical_files` is True when every document in the group has the same
    content hash — i.e. literally the same file re-uploaded, as opposed to
    two different images of one page.
    """
    annotate_prescription_groups(docs)

    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for doc in docs:
        by_group.setdefault(doc["prescription_group"], []).append(doc)

    duplicates = []
    for group_id, members in by_group.items():
        if len(members) < 2:
            continue
        hashes = {m.get("content_sha256") for m in members}
        duplicates.append({
            "prescription_group": group_id,
            "identical_files": len(hashes) == 1 and None not in hashes,
            "medications": sorted({
                (m.get("name") or "unnamed")
                for doc in members for m in (doc.get("medications") or [])
            }),
            "documents": [
                {
                    "source_file": (m.get("_source") or {}).get("file"),
                    "date": m.get("date"),
                    "uploaded_at": m.get("uploaded_at"),
                    "document_url": m.get("document_url"),
                    "cloudinary_public_id": m.get("cloudinary_public_id"),
                    "content_sha256": m.get("content_sha256"),
                }
                for m in members
            ],
        })
    return duplicates


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # --- Date ambiguity, taken verbatim from the real record ---------------
    # One prescription was extracted three times, as "09 / 11 / 2025",
    # "09/11/2025" and "2025-11-09". All three must be seen as one date.
    assert dates_compatible("09/11/2025", "2025-11-09")
    assert dates_compatible("09 / 11 / 2025", "09/11/2025")
    assert dates_compatible("09 / 11 / 2025", "2025-11-09")
    # Genuinely different dates must not be merged.
    assert not dates_compatible("14/10/2023", "09/11/2025")
    assert not dates_compatible("2025-11-09", "2026-02-26")
    # A missing date never splits an otherwise identical prescription.
    assert dates_compatible(None, "09/11/2025")
    assert dates_compatible("", None)

    # --- Salt-form tolerance ------------------------------------------------
    assert _base_ingredient("Diclofenac sodium") == "diclofenac"
    assert _base_ingredient("Diclofenac") == "diclofenac"
    assert _base_ingredient("Glucosamine sulphate") == "glucosamine"
    assert _base_ingredient("Cetirizine hydrochloride") == "cetirizine"
    # A single-word name is never stripped away to nothing.
    assert _base_ingredient("Sodium") == "sodium"
    assert _base_ingredient("Paracetamol") == "paracetamol"

    # --- The real case: one prescription, three files ----------------------
    def rx(source, date, diclofenac_ingredient, glucosamine_name):
        return {
            "document_type": "prescription",
            "date": date,
            "patient_name": "RAMESH",
            "provider_or_doctor": "Dr. K. Jayasuriya",
            "medications": [
                {"name": "Paracetamol", "ingredients": ["Paracetamol"],
                 "dosage_value": 1000, "dosage_unit": "mg", "frequency_per_day": 3,
                 "is_as_needed": False},
                {"name": "Diclofenac sodium (topical gel)",
                 "ingredients": [diclofenac_ingredient],
                 "dosage_value": None, "dosage_unit": None, "frequency_per_day": 2,
                 "is_as_needed": False},
                {"name": "Omeprazole", "ingredients": ["Omeprazole"],
                 "dosage_value": 20, "dosage_unit": "mg", "frequency_per_day": 1,
                 "is_as_needed": False},
                {"name": glucosamine_name, "ingredients": ["Glucosamine sulfate"],
                 "dosage_value": 500, "dosage_unit": "mg", "frequency_per_day": 2,
                 "is_as_needed": False},
            ],
            "_source": {"file": source},
        }

    osteo_png = rx("08_09-11-2025_osteoarthritis.png", "09 / 11 / 2025",
                   "Diclofenac", "Glucosamine sulphate")
    whatsapp_1 = rx("WhatsApp Image 2026-08-16 at 12.11.14.jpeg", "09/11/2025",
                    "Diclofenac", "Glucosamine sulphate")
    whatsapp_2 = rx("WhatsApp Image 2026-08-16 at 12.11.14.jpeg", "2025-11-09",
                    "Diclofenac sodium", "Glucosamine sulfate")

    # A genuinely different prescription, 2 years earlier.
    pharyngitis = {
        "document_type": "prescription", "date": "14/10/2023",
        "patient_name": "RAMESH", "provider_or_doctor": "Dr. N. K. Wijesinghe",
        "medications": [
            {"name": "Amoxicillin", "ingredients": ["Amoxicillin"],
             "dosage_value": 500, "dosage_unit": "mg", "frequency_per_day": 3,
             "is_as_needed": False},
        ],
        "_source": {"file": "01_14-10-2023_acute_bacterial_pharyngitis.png"},
    }

    docs = [osteo_png, whatsapp_1, whatsapp_2, pharyngitis]
    annotate_prescription_groups(docs)

    assert osteo_png["prescription_group"] == whatsapp_1["prescription_group"], (
        osteo_png["prescription_group"], whatsapp_1["prescription_group"])
    assert osteo_png["prescription_group"] == whatsapp_2["prescription_group"], (
        "salt-form and date-format differences must not split one prescription")
    assert pharyngitis["prescription_group"] != osteo_png["prescription_group"], (
        "a genuinely different prescription must keep its own group")

    groups = find_duplicate_document_groups(docs)
    assert len(groups) == 1, groups
    assert len(groups[0]["documents"]) == 3
    assert groups[0]["identical_files"] is False, "different files, same prescription"

    # --- A real repeat prescription must NOT be merged away ----------------
    # Same drugs and doses, same doctor, but months later: that is a genuine
    # second prescription and must stay separate.
    repeat = rx("later_visit.png", "26/02/2026", "Diclofenac", "Glucosamine sulphate")
    docs2 = [osteo_png.copy(), repeat]
    for d in docs2:
        d.pop("prescription_group", None)
    annotate_prescription_groups(docs2)
    assert docs2[0]["prescription_group"] != docs2[1]["prescription_group"], (
        "a repeat prescription on a different date is not a duplicate file")

    # --- Different patients never merge ------------------------------------
    other_patient = rx("someone_else.png", "09/11/2025", "Diclofenac",
                       "Glucosamine sulphate")
    other_patient["patient_name"] = "SURESH"
    docs3 = [rx("a.png", "09/11/2025", "Diclofenac", "Glucosamine sulphate"),
             other_patient]
    annotate_prescription_groups(docs3)
    assert docs3[0]["prescription_group"] != docs3[1]["prescription_group"]

    # --- Documents with no medications each stand alone --------------------
    labs = [
        {"document_type": "lab_report", "date": "01/01/2026", "patient_name": "RAMESH",
         "medications": [], "_source": {"file": "lab1.pdf"}},
        {"document_type": "lab_report", "date": "01/06/2026", "patient_name": "RAMESH",
         "medications": [], "_source": {"file": "lab2.pdf"}},
    ]
    annotate_prescription_groups(labs)
    assert labs[0]["prescription_group"] != labs[1]["prescription_group"], (
        "two lab reports share an empty medication set but are not one prescription")
    assert not find_duplicate_document_groups(labs)

    print("One prescription across 3 files -> group:", osteo_png["prescription_group"])
    print("  files:", [d["source_file"] for d in groups[0]["documents"]])
    print("  medications:", ", ".join(groups[0]["medications"]))
    # --- name_variants: printed name -> the graph's bare ingredient key ----
    # A prescription says "Naloxone Hydrochloride 400mcg"; the graph stores
    # "naloxone". Without these variants the lookup misses silently.
    assert name_variants("Naloxone Hydrochloride") == ["naloxone hydrochloride", "naloxone"]
    assert "naloxone" in name_variants("NALOXONE 400mcg")
    assert "morphine" in name_variants("Morphine Sulfate")
    assert name_variants("Naloxone") == ["naloxone"]
    assert "amoxicillin" in name_variants("Amoxicillin 500 mg")
    # Never truncate to the first word: these are different products.
    assert "insulin" not in name_variants("Insulin glargine")
    assert name_variants("") == [] and name_variants(None) == []

    print("\nAll checks passed.")
