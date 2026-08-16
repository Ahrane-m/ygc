"""
Language Detection / Normalization Guard
=========================================
Fails an upload loudly when the pipeline cannot confirm it actually handled
the document's language, instead of letting it through as data that looks
fine and is quietly wrong.

WHY THIS IS A SAFETY CHECK, NOT A NICETY
----------------------------------------
This pipeline supports documents in any language by NORMALIZING at extraction
time: `ingredients` is always the English INN ("Amoxicilina", "アモキシシリン"
-> ["Amoxicillin"]), and dose/frequency are reduced to numbers. Everything
downstream is built on that promise:

  * retrieval._med_group_key() groups a drug across documents by its
    normalized ingredients, falling back to the PRINTED name when ingredients
    is empty.
  * detect_exact_duplicate_medications() keys on the ingredient set.
  * document_filter.py explicitly has no per-language branch, precisely
    because it can rely on those normalized fields.

So when normalization silently fails on one document — the model hands back
"アモキシシリン" as the ingredient, or leaves ingredients empty for a drug
printed in Tamil — the failure is invisible and lands exactly where it hurts
most: the same drug under two languages produces two different group keys, so
the duplicate is never spotted and the interaction check never sees both
halves. Nothing errors. The record just looks complete while missing the very
cross-document link the cross-check exists to find.

That is worth an error rather than a warning. A rejected file tells the user
something concrete they can act on; a silently half-normalized one tells them
their records are fine when they are not.

ONLY POSITIVE EVIDENCE COUNTS
-----------------------------
Every check below fires on evidence that something DID go wrong, never on the
mere absence of evidence — the same discipline the extraction prompt applies
to patient_gender ("never guess from the name alone") and lab_trends applies
to its glossary. An unlisted language, an unusual script, or a missing
`document_language` on a document whose fields all came back properly
normalized is not an error: the normalization is what matters, and it
demonstrably worked. This keeps a valid upload from being rejected because a
language merely looks unfamiliar.

No extra model call: like document_filter.py, every check here reads fields
process_document() already returned, so a rejected file costs nothing beyond
the extraction that was already paid for, and is rejected before the
Cloudinary upload, timeline rebuild and cross-check LLM call.
"""

import unicodedata
from typing import Any, Dict, List, Optional, Tuple


class UnsupportedLanguageError(ValueError):
    """Raised when a document's language could not be detected, or was
    detected but the extraction did not normalize into the English fields the
    rest of the pipeline depends on.

    Carries the filename, the detected language (when known) and the specific
    field-level problems, so an API layer can build an actionable message
    without re-deriving any of it."""

    def __init__(
        self,
        filename: str,
        reason: str,
        detected_language: Optional[str] = None,
        problems: Optional[List[str]] = None,
    ):
        self.filename = filename
        self.reason = reason
        self.detected_language = detected_language
        self.problems = problems or []
        super().__init__(f"'{filename}' could not be processed reliably: {reason}")


def _is_latin_script(text: str) -> bool:
    """True if every LETTER in `text` is Latin-script.

    Digits, punctuation, spaces and symbols are ignored — "Vitamin B12",
    "Co-amoxiclav 500mg" and "Levo-thyroxine (T4)" are all Latin. A string
    with no letters at all (e.g. "500") counts as Latin: there is nothing
    un-normalized about it.

    This is the load-bearing test for "did normalization happen?". INN
    generic names are Latin-script by definition, so a non-Latin character
    in an `ingredients` entry is proof the model returned the printed name
    instead of translating it — not a judgment call.
    """
    for char in text:
        if not char.isalpha():
            continue
        try:
            if not unicodedata.name(char).startswith("LATIN"):
                return False
        except ValueError:  # unnamed character — can't confirm it's Latin
            return False
    return True


def _has_letters(text: str) -> bool:
    return any(char.isalpha() for char in text)


def detect_language_problems(doc: Dict[str, Any]) -> List[str]:
    """
    Returns a list of specific, human-readable problems that mean this
    document's language was not handled reliably. Empty list == fine.

    Each entry names the exact field at fault so the message can point at
    something the user can actually look at on their own document.
    """
    problems: List[str] = []

    for index, med in enumerate(doc.get("medications") or []):
        name = (med.get("name") or "").strip()
        ingredients = [i for i in (med.get("ingredients") or []) if i and i.strip()]

        # (a) An ingredient that isn't Latin script is a definitive
        #     normalization failure: INN names are always Latin.
        untranslated = [i for i in ingredients if not _is_latin_script(i)]
        if untranslated:
            problems.append(
                f"medication {index + 1} ({name or 'unnamed'}) lists its active "
                f"ingredient as {', '.join(repr(i) for i in untranslated)}, which is "
                "still in the document's own script — it was not translated to the "
                "standard English drug name the rest of your records are matched on"
            )
            continue

        # (b) A drug printed in a non-Latin script with NO ingredient
        #     resolved. retrieval._med_group_key() then falls back to the
        #     printed name, so this drug can never match the same drug on an
        #     English document — the duplicate/interaction checks silently
        #     lose it. This is the quiet failure this module exists to catch.
        if not ingredients and name and _has_letters(name) and not _is_latin_script(name):
            problems.append(
                f"medication {index + 1} ('{name}') could not be matched to a standard "
                "English drug name, so it cannot be cross-checked against the "
                "medications already in your records"
            )

    return problems


def language_rejection_reason(doc: Dict[str, Any]) -> Optional[str]:
    """
    One-line explanation of why this document can't be processed reliably, or
    None if it can. Written to be shown to whoever uploaded the file, so it
    says what happened and what to do — not what the code checked.
    """
    problems = detect_language_problems(doc)
    if not problems:
        return None

    language = doc.get("document_language")
    additional = [l for l in (doc.get("additional_languages") or []) if l]

    if language and additional:
        where = (
            f"This document is in {language} and also contains "
            f"{', '.join(additional)}"
        )
    elif language:
        where = f"This document is in {language}"
    else:
        # Language genuinely undetermined AND normalization failed — the
        # combination is what makes this reportable, not the missing field.
        where = "The language of this document could not be determined"

    return (
        f"{where}, and some details could not be converted into the standard "
        f"English form your records are matched on ({len(problems)} field(s) "
        "affected). Uploading a clearer scan or photo usually fixes this. If the "
        "document is correct as-is, ask your pharmacist or doctor for a copy that "
        "also lists the generic (non-brand) drug names."
    )


def is_language_supported(doc: Dict[str, Any]) -> bool:
    """True if this extraction result can be trusted for cross-document
    matching. A document in ANY language passes as long as its fields came
    back normalized — the language itself is never the thing being rejected."""
    return not detect_language_problems(doc)


def assert_supported_language(doc: Dict[str, Any], filename: str) -> None:
    """
    Raises UnsupportedLanguageError if this document's language could not be
    handled reliably. Call once per uploaded file/page right after
    process_document(), alongside assert_medical_document().
    """
    reason = language_rejection_reason(doc)
    if reason is None:
        return
    raise UnsupportedLanguageError(
        filename,
        reason,
        detected_language=doc.get("document_language"),
        problems=detect_language_problems(doc),
    )


# ---------------------------------------------------------------------------
# Graduated red flag
#
# assert_supported_language() above is binary and fires only on proof. That
# deliberately leaves a gap: a document can be fully normalized on paper and
# still be the kind of document worth a second look — a photographed Japanese
# prescription whose drug names were transliterated, say. Nothing there is
# provably wrong, so rejecting it would be an overreach, but treating it as
# equal to a typed English printout would be a different mistake.
#
# This is the graduated half: it never blocks an upload, it raises a flag.
# It reads the two independent confidence axes the extractor now reports —
# ocr_confidence (could we read the page?) and translation_confidence (did we
# convert it faithfully?) — because they fail differently and are fixed
# differently. A bad read needs a better scan. A bad conversion needs a
# pharmacist to confirm the generic name. One blended number can't tell a
# user which of those to go do.
# ---------------------------------------------------------------------------

# At or below these, the corresponding axis is called out by name. Matches
# retrieval.LOW_CONFIDENCE_THRESHOLD so "low confidence" means the same thing
# to a user across the whole product.
LOW_OCR_CONFIDENCE = 0.6
LOW_TRANSLATION_CONFIDENCE = 0.6

# Between this and the low threshold, an axis is "middling" — not called out
# on its own, but enough to raise the flag when the document also had to be
# translated.
MIDDLING_CONFIDENCE = 0.8

# Applied to the two axes MULTIPLIED together. This catches the case neither
# individual threshold can: both axes sitting just above their own limits
# (0.65 read x 0.75 conversion) while combined trust is under a half. Without
# it, the compounding is computed and then ignored — the document with the
# worst real trust in the batch can show no flag at all.
LOW_EFFECTIVE_CONFIDENCE = 0.6

ENGLISH_NAMES = frozenset({"english", "en", "eng"})


def _is_english_only(doc: Dict[str, Any]) -> bool:
    """True if this document is English throughout, so nothing had to be
    translated. Unknown language is NOT treated as English."""
    language = (doc.get("document_language") or "").strip().lower()
    additional = [
        (l or "").strip().lower() for l in (doc.get("additional_languages") or [])
    ]
    if language not in ENGLISH_NAMES:
        return False
    return all(l in ENGLISH_NAMES for l in additional if l)


def _is_known_non_english(doc: Dict[str, Any]) -> bool:
    """True only when the extraction POSITIVELY reported a non-English
    language. Distinct from `not _is_english_only(doc)`, which is also true
    for a document that simply never reported a language at all.

    That distinction is the whole "only positive evidence" rule applied to
    the red flag: documents extracted before these fields existed carry no
    language at all, and flagging every one of them as "this was translated"
    would fire a warning on the entire back catalogue on the strength of a
    missing field. A heads-up nobody can act on is worse than none — it
    teaches people to dismiss the flag, including the times it is real.
    """
    language = (doc.get("document_language") or "").strip().lower()
    if language and language not in ENGLISH_NAMES:
        return True
    for extra in doc.get("additional_languages") or []:
        if (extra or "").strip().lower() not in ENGLISH_NAMES and (extra or "").strip():
            return True
    return False


def _as_confidence(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def assess_translation_risk(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Grades how much this document's English fields should be trusted, and
    returns a red flag when they warrant review.

    Returns:
      {
        "flag": "none" | "review" | "high",
        "is_english_only": bool,
        "document_language": str | None,
        "additional_languages": [str],
        "ocr_confidence": float | None,
        "translation_confidence": float | None,
        "effective_confidence": float | None,  # the two axes combined
        "reasons": [str],      # plain language, each naming which axis failed
        "message": str | None, # one line to show the user, None if flag=="none"
      }

    "high" means the document was translated AND at least one axis is
    genuinely low — the case where a drug name in the record may not be the
    drug on the page. "review" means it was translated and something is
    middling, or an axis is low on an English document (a bad scan is worth
    flagging even with no translation involved).
    """
    ocr = _as_confidence(doc.get("ocr_confidence"))
    translation = _as_confidence(doc.get("translation_confidence"))
    english_only = _is_english_only(doc)
    language = doc.get("document_language")
    additional = [l for l in (doc.get("additional_languages") or []) if l]

    # The two axes are independent measurements, but trust in the English
    # fields is bounded by BOTH: a perfect translation of a misread word is
    # still wrong. Multiplying keeps that honest, where taking the max would
    # let a confident translation paper over an unreadable page.
    if ocr is not None and translation is not None:
        effective = round(ocr * translation, 2)
    else:
        effective = ocr if translation is None else translation

    reasons: List[str] = []
    severity = "none"

    if ocr is not None and ocr <= LOW_OCR_CONFIDENCE:
        reasons.append(
            f"the document was hard to read ({ocr:.0%} confidence in making out the "
            "text) — a clearer scan or photo would help"
        )
        severity = "high" if not english_only else "review"

    if not english_only and translation is not None and translation <= LOW_TRANSLATION_CONFIDENCE:
        reasons.append(
            f"converting the drug names and doses into English was uncertain "
            f"({translation:.0%} confidence) — the generic drug names recorded here "
            "may not match what the document actually says"
        )
        severity = "high"

    if not reasons and effective is not None and effective <= LOW_EFFECTIVE_CONFIDENCE:
        # Neither axis is individually low, but they compound. Name whichever
        # is weaker so the advice points at the right fix — a clearer scan
        # versus confirming the drug names.
        if ocr is not None and (translation is None or ocr <= translation):
            weaker = (
                f"reading it was only {ocr:.0%} certain and, combined with everything "
                f"else, the details taken from it are about {effective:.0%} reliable "
                "overall — a clearer scan or photo would help"
            )
        else:
            weaker = (
                f"converting it into English was only {translation:.0%} certain and, "
                f"combined with everything else, the details taken from it are about "
                f"{effective:.0%} reliable overall"
            )
        reasons.append(weaker)
        severity = "high" if not english_only else "review"

    if _is_known_non_english(doc) and not reasons:
        # Nothing is low, but this document was still translated. Say so —
        # this is the red flag the feature exists for. Requires a POSITIVELY
        # reported non-English language, so a document that never reported
        # one (anything extracted before these fields existed) stays silent.
        printed_in = language or "a language that could not be identified"
        if additional:
            printed_in += f" (also containing {', '.join(additional)})"
        if (translation is not None and translation < MIDDLING_CONFIDENCE) or (
            ocr is not None and ocr < MIDDLING_CONFIDENCE
        ):
            reasons.append(
                f"this document is in {printed_in}, and the English drug names in your "
                "records were converted from it rather than printed on it"
            )
            severity = "review"
        else:
            reasons.append(
                f"this document is in {printed_in}; the English drug names in your "
                "records were converted from it rather than printed on it, and that "
                "conversion looked straightforward"
            )
            severity = "review"

    if not reasons:
        return {
            "flag": "none",
            "is_english_only": english_only,
            "document_language": language,
            "additional_languages": additional,
            "ocr_confidence": ocr,
            "translation_confidence": translation,
            "effective_confidence": effective,
            "reasons": [],
            "message": None,
        }

    if severity == "high":
        lead = (
            "Please check this document against your records before relying on it: "
        )
        tail = (
            " Ask your pharmacist to confirm the medicine names — they can check the "
            "original document against what was dispensed."
        )
    else:
        lead = "Worth a quick check: "
        tail = (
            " Nothing here is known to be wrong — this is a heads-up, not an error."
        )

    return {
        "flag": severity,
        "is_english_only": english_only,
        "document_language": language,
        "additional_languages": additional,
        "ocr_confidence": ocr,
        "translation_confidence": translation,
        "effective_confidence": effective,
        "reasons": reasons,
        "message": lead + "; ".join(reasons) + "." + tail,
    }


def assess_documents_translation_risk(
    docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Rolls per-document assessments up across a whole batch/timeline, so a
    caller can show one banner rather than walking every document.

    Returns {"flag", "flagged_documents": [{source_file, **assessment}],
    "counts": {"high": n, "review": n}, "message"}.
    """
    flagged: List[Dict[str, Any]] = []
    counts = {"high": 0, "review": 0}

    for doc in docs:
        assessment = assess_translation_risk(doc)
        if assessment["flag"] == "none":
            continue
        counts[assessment["flag"]] += 1
        flagged.append({
            "source_file": (doc.get("_source") or {}).get("file"),
            **assessment,
        })

    if counts["high"]:
        overall = "high"
        message = (
            f"{counts['high']} document(s) in your records need checking before you "
            "rely on the medicine names in them. Your pharmacist can confirm them "
            "against the original documents."
        )
    elif counts["review"]:
        overall = "review"
        message = (
            f"{counts['review']} document(s) in your records were not written in "
            "English, so the medicine names shown were converted from another "
            "language. Nothing is known to be wrong — worth mentioning to your "
            "pharmacist if anything looks unfamiliar."
        )
    else:
        overall = "none"
        message = None

    return {
        "flag": overall,
        "counts": counts,
        "flagged_documents": flagged,
        "message": message,
    }


def filter_unsupported_language_documents(
    docs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits extraction results into (kept, rejected), mirroring
    document_filter.filter_non_medical_documents() for callers that would
    rather partition a batch than raise on the first bad file."""
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for doc in docs:
        (kept if is_language_supported(doc) else rejected).append(doc)
    return kept, rejected


if __name__ == "__main__":
    import sys

    # Test data below is in several scripts; a cp1252 console cannot print it.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # --- Documents that MUST pass ------------------------------------------
    # The whole point is that a non-English document is fine when
    # normalization worked. Rejecting these would break multilingual support
    # rather than guard it.
    tamil_normalized = {
        "document_language": "Tamil",
        "additional_languages": [],
        "medications": [
            {"name": "மெட்ஃபோர்மின்", "ingredients": ["Metformin"]},
            {"name": "அம்லோடிபின்", "ingredients": ["Amlodipine"]},
        ],
    }
    mixed_language_normalized = {
        # Very common in practice: English drug names, Sinhala instructions.
        "document_language": "Sinhala",
        "additional_languages": ["English"],
        "medications": [{"name": "Amoxicillin", "ingredients": ["Amoxicillin"]}],
    }
    english_plain = {
        "document_language": "English",
        "additional_languages": [],
        "medications": [{"name": "Panadol", "ingredients": ["Paracetamol"]}],
    }
    lab_report_no_medications = {
        "document_language": "Japanese",
        "additional_languages": [],
        "medications": [],
        "lab_results": [{"test_name": "ALT", "value": "42"}],
    }
    language_unknown_but_normalized = {
        # Language undetermined, but every field came back normalized — the
        # missing label alone must NOT reject the upload.
        "document_language": None,
        "additional_languages": [],
        "medications": [{"name": "Amoxil", "ingredients": ["Amoxicillin"]}],
    }
    latin_with_punctuation = {
        "document_language": "Spanish",
        "additional_languages": [],
        "medications": [
            {"name": "Co-amoxiclav 500mg", "ingredients": ["Co-amoxiclav", "Vitamin B12"]},
        ],
    }

    for doc in (tamil_normalized, mixed_language_normalized, english_plain,
                lab_report_no_medications, language_unknown_but_normalized,
                latin_with_punctuation):
        assert is_language_supported(doc), (
            f"must not reject a normalized document: {doc.get('document_language')} "
            f"-> {detect_language_problems(doc)}"
        )

    # --- Documents that MUST be rejected -----------------------------------
    untranslated_ingredient = {
        "document_language": "Japanese",
        "additional_languages": [],
        # Normalization definitively failed: an INN is never in katakana.
        "medications": [{"name": "アモキシシリン", "ingredients": ["アモキシシリン"]}],
    }
    unresolved_non_latin_name = {
        "document_language": "Tamil",
        "additional_languages": ["English"],
        # No ingredient at all -> _med_group_key falls back to the printed
        # name, so this can never match "Metformin" on an English document.
        "medications": [{"name": "மெட்ஃபோர்மின்", "ingredients": []}],
    }
    undetected_language_and_unresolved = {
        "document_language": None,
        "additional_languages": [],
        "medications": [{"name": "アモキシシリン", "ingredients": []}],
    }

    for doc in (untranslated_ingredient, unresolved_non_latin_name,
                undetected_language_and_unresolved):
        assert not is_language_supported(doc), f"should have been rejected: {doc}"

    # --- The error itself must be usable -----------------------------------
    try:
        assert_supported_language(untranslated_ingredient, "japanese_rx.pdf")
        raise SystemExit("expected UnsupportedLanguageError")
    except UnsupportedLanguageError as e:
        assert e.filename == "japanese_rx.pdf"
        assert e.detected_language == "Japanese"
        assert len(e.problems) == 1
        assert "Japanese" in str(e), str(e)
        assert "clearer scan" in str(e), str(e)
        print("OK — untranslated ingredient:\n   ", e, "\n")

    try:
        assert_supported_language(unresolved_non_latin_name, "tamil_rx.jpg")
        raise SystemExit("expected UnsupportedLanguageError")
    except UnsupportedLanguageError as e:
        # A mixed-language document should say so, naming both languages.
        assert "Tamil" in str(e) and "English" in str(e), str(e)
        print("OK — mixed-language, unresolved drug:\n   ", e, "\n")

    try:
        assert_supported_language(undetected_language_and_unresolved, "blurry.jpg")
        raise SystemExit("expected UnsupportedLanguageError")
    except UnsupportedLanguageError as e:
        assert "could not be determined" in str(e), str(e)
        assert e.detected_language is None
        print("OK — language undetermined:\n   ", e, "\n")

    # --- Script detection edge cases ---------------------------------------
    assert _is_latin_script("Amoxicillin")
    assert _is_latin_script("Co-amoxiclav 500mg")
    assert _is_latin_script("Vitamin B12 (oral)")
    assert _is_latin_script("500"), "digits only — nothing to normalize"
    assert _is_latin_script(""), "empty string is not a failure"
    assert _is_latin_script("Paracétamol"), "accented Latin is still Latin"
    assert not _is_latin_script("アモキシシリン")
    assert not _is_latin_script("மெட்ஃபோர்மின்")
    assert not _is_latin_script("أموكسيسيلين")
    assert not _is_latin_script("Amoxicillin / アモキシシリン"), "mixed script fails"

    kept, rejected = filter_unsupported_language_documents([
        tamil_normalized, untranslated_ingredient, english_plain,
        unresolved_non_latin_name,
    ])
    assert len(kept) == 2 and len(rejected) == 2

    # --- Red flag: the two confidence axes ---------------------------------
    # A clean English document is the only case that raises nothing at all.
    clean_english = {
        "document_language": "English", "additional_languages": [],
        "ocr_confidence": 0.97, "translation_confidence": 1.0,
        "medications": [{"name": "Paracetamol", "ingredients": ["Paracetamol"]}],
    }
    assert assess_translation_risk(clean_english)["flag"] == "none"
    assert assess_translation_risk(clean_english)["message"] is None

    # Sharply printed but foreign: OCR is fine, translation is the risk. This
    # is the case the whole feature exists to surface.
    crisp_japanese = {
        "document_language": "Japanese", "additional_languages": [],
        "ocr_confidence": 0.96, "translation_confidence": 0.72,
        "medications": [{"name": "アモキシシリン", "ingredients": ["Amoxicillin"]}],
    }
    crisp = assess_translation_risk(crisp_japanese)
    assert crisp["flag"] == "review", crisp
    assert crisp["is_english_only"] is False
    assert "Japanese" in crisp["message"], crisp["message"]
    assert crisp["effective_confidence"] == 0.69, crisp["effective_confidence"]

    # Blurry but English: OCR is the risk, translation is not. Must NOT be
    # reported as a translation problem — different fix entirely.
    blurry_english = {
        "document_language": "English", "additional_languages": [],
        "ocr_confidence": 0.42, "translation_confidence": 1.0,
        "medications": [{"name": "Paracetamol", "ingredients": ["Paracetamol"]}],
    }
    blurry = assess_translation_risk(blurry_english)
    assert blurry["flag"] == "review", blurry
    assert "hard to read" in blurry["message"], blurry["message"]
    assert "converting" not in blurry["message"], (
        "an English document must never be reported as a translation problem"
    )

    # Both axes bad on a foreign document — the genuinely alarming combination.
    blurry_foreign = {
        "document_language": "Tamil", "additional_languages": ["English"],
        "ocr_confidence": 0.45, "translation_confidence": 0.4,
        "medications": [{"name": "மெட்ஃபோர்மின்", "ingredients": ["Metformin"]}],
    }
    worst = assess_translation_risk(blurry_foreign)
    assert worst["flag"] == "high", worst
    assert len(worst["reasons"]) == 2, worst["reasons"]
    assert "pharmacist" in worst["message"]
    assert worst["effective_confidence"] == 0.18, worst["effective_confidence"]

    # A confident translation must not paper over an unreadable page: the two
    # are combined by multiplying, not by taking the better of them.
    assert assess_translation_risk({
        "document_language": "Spanish", "additional_languages": [],
        "ocr_confidence": 0.3, "translation_confidence": 1.0,
    })["effective_confidence"] == 0.3

    # A well-translated foreign document still gets a heads-up (that IS the
    # red flag the feature asks for), but at the gentler level.
    good_foreign = assess_translation_risk({
        "document_language": "Spanish", "additional_languages": [],
        "ocr_confidence": 0.95, "translation_confidence": 0.95,
        "medications": [{"name": "Amoxicilina", "ingredients": ["Amoxicillin"]}],
    })
    assert good_foreign["flag"] == "review", good_foreign
    assert "not an error" in good_foreign["message"], good_foreign["message"]

    # Compounding: both axes individually above their thresholds, but their
    # product is not. Observed live — a poorly-scanned English document came
    # back 0.65 read / 0.75 conversion, i.e. under 50% combined, and showed no
    # flag at all until this case was handled.
    compounding = assess_translation_risk({
        "document_language": "English", "additional_languages": [],
        "ocr_confidence": 0.65, "translation_confidence": 0.75,
    })
    assert compounding["effective_confidence"] == 0.49, compounding
    assert compounding["flag"] == "review", compounding
    assert "clearer scan" in compounding["message"], compounding["message"]

    # The same compounding on a translated document is worse, and says so.
    compounding_foreign = assess_translation_risk({
        "document_language": "Tamil", "additional_languages": [],
        "ocr_confidence": 0.75, "translation_confidence": 0.65,
    })
    assert compounding_foreign["flag"] == "high", compounding_foreign
    assert "converting it into English" in compounding_foreign["message"]

    # Just above the line must stay quiet, so the flag keeps its meaning.
    assert assess_translation_risk({
        "document_language": "English", "additional_languages": [],
        "ocr_confidence": 0.9, "translation_confidence": 0.9,
    })["flag"] == "none"

    # Unknown language is not silently treated as English.
    assert _is_english_only({"document_language": None}) is False
    assert _is_english_only({"document_language": "English",
                             "additional_languages": ["Sinhala"]}) is False
    assert _is_english_only({"document_language": "English",
                             "additional_languages": []}) is True

    # Missing axes (e.g. a snapshot saved before these fields existed) must
    # not crash or invent a flag.
    legacy = assess_translation_risk({"document_language": "English",
                                      "additional_languages": []})
    assert legacy["flag"] == "none", legacy
    assert legacy["effective_confidence"] is None

    # A document from before ANY of these fields existed reports no language
    # at all. It must stay silent: "we don't know" is not evidence that it was
    # translated, and flagging the whole back catalogue would train users to
    # dismiss the flag entirely.
    pre_existing = assess_translation_risk({
        "medications": [{"name": "Paracetamol", "ingredients": ["Paracetamol"]}],
        "overall_confidence": 0.9,
    })
    assert pre_existing["flag"] == "none", pre_existing
    assert pre_existing["message"] is None
    assert _is_known_non_english({}) is False
    assert _is_known_non_english({"document_language": None}) is False
    assert _is_known_non_english({"document_language": "English"}) is False
    assert _is_known_non_english({"document_language": "Tamil"}) is True
    assert _is_known_non_english({"document_language": "English",
                                  "additional_languages": ["Sinhala"]}) is True

    # --- Batch rollup -------------------------------------------------------
    batch = assess_documents_translation_risk([
        {**clean_english, "_source": {"file": "a.pdf"}},
        {**crisp_japanese, "_source": {"file": "b.pdf"}},
        {**blurry_foreign, "_source": {"file": "c.jpg"}},
    ])
    assert batch["flag"] == "high", batch
    assert batch["counts"] == {"high": 1, "review": 1}, batch["counts"]
    assert len(batch["flagged_documents"]) == 2
    assert {d["source_file"] for d in batch["flagged_documents"]} == {"b.pdf", "c.jpg"}

    all_clean = assess_documents_translation_risk([clean_english])
    assert all_clean["flag"] == "none" and all_clean["message"] is None

    print("OK — crisp foreign document flagged on translation, not on reading:")
    print("   ", crisp["message"], "\n")
    print("OK — blurry English flagged on reading, not on translation:")
    print("   ", blurry["message"], "\n")
    print("OK — both axes low:")
    print("   ", worst["message"], "\n")

    print("All checks passed.")
