"""
WHO Essential Medicines List — full ingest
=========================================
Loads an entire WHO Model List of Essential Medicines PDF into Neo4j, not just
the antidotes section that poisoning_kg.py extracts.

    (:Medicine)-[:LISTED_IN {dosage_form, list_type, section, source_page}]->(:SourceDocument)
    (:Medicine)-[:IN_SECTION]->(:Section)
    (:Medicine)-[:HAS_ALTERNATIVE]->(:Medicine)
    (:Medicine)-[:IN_AWARE_GROUP]->(:AWaReGroup)
    (:Medicine)-[:FIRST_CHOICE_FOR|:SECOND_CHOICE_FOR]->(:Indication)
    (:Medicine)-[:HAS_RESTRICTION]->(:AgeRestriction)

WHY THE WHOLE LIST, NOT JUST ANTIDOTES
--------------------------------------
The antidote section gave the graph 15 medicines. The full list gives it a
few hundred, and — more usefully — gives them a THERAPEUTIC CLASSIFICATION
from an authoritative, citable source:

    2.2   Opioid analgesics          -> morphine, codeine, fentanyl, methadone
    24.3  Medicines for anxiety      -> diazepam, lorazepam
    6.2.1 Access group antibiotics   -> amoxicillin, ampicillin, cefalexin

That matters because reference_library.py currently classifies drugs from a
hand-curated list, honestly labelled as "not quoted from the source". Where
WHO's own sectioning covers a class, that curated guess can be replaced by a
citation — which is the difference between a finding graded `model_knowledge`
and one graded `reference_graph`.

Two other things the full list carries that the antidote section did not:

  * `Table 1.1: Medicines with age or weight restrictions` — real, quotable
    paediatric safety data ("doxycycline > 8 years", "metoclopramide not in
    neonates"). This pipeline extracts patient age at every upload and has
    had nothing authoritative to check it against.
  * Square-box therapeutic alternatives — WHO's own statement that one
    medicine may substitute for another, which is exactly the question behind
    "is this a duplicate?".

Same discipline as poisoning_kg.py: every field is copied verbatim from a
table cell. Nothing is inferred, and the source never states which drug
interacts with which, so nothing here claims that.

Usage:
    python eml_kg.py --ingest "C:/path/WHO-MHP-HPS-EML-2023.02-eng.pdf"
    python eml_kg.py                      # show what is loaded
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

from graph_db import run_read, run_write, session_scope

logger = logging.getLogger("eml_kg")

# A top-level section heading occupies a whole row: "2. MEDICINES FOR PAIN…".
SECTION_RE = re.compile(r"^(\d{1,2})\.\s+([A-Z][A-Za-z0-9 ,''\-–&/()]+)$")
# Subsections nest arbitrarily deep: "6.2.1 Access group antibiotics".
SUBSECTION_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})+)\s+(.+)$")
COMPLEMENTARY_RE = re.compile(r"^complementary\s+list$", re.IGNORECASE)
CHILDREN_LIST_RE = re.compile(r"Model List of Essential Medicines for Children", re.IGNORECASE)

# The square box (therapeutic-alternatives marker) renders as a private-use
# glyph in this PDF, not as U+25A1. Both are accepted.
SQUARE_BOX = "\uf06f\u25a1\u2610"
# Leading "o " is the same marker rendered with a different font.
LEADING_MARKER_RE = re.compile(rf"^[{SQUARE_BOX}]\s*|^o\s+(?=[a-z])")
ALTERNATIVES_HEADER_RE = re.compile(r"therapeutic\s+alternatives", re.IGNORECASE)
# "4th level ATC chemical subgroup (…)" is a CLASS reference, not a medicine.
ATC_CLASS_RE = re.compile(r"\bATC\b|chemical subgroup", re.IGNORECASE)

AWARE_GROUPS = {
    "access": "Access",
    "watch": "Watch",
    "reserve": "Reserve",
}

MAX_PAGES = 200


def _clean(text: Any) -> str:
    """Collapse the newlines pdfplumber keeps inside a wrapped table cell."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _medicine_name(raw: str) -> str:
    """
    The drug name from a name-cell, with the list's typographic markers
    stripped: square box, trailing footnote markers (* † a [c]), and anything
    after the "Therapeutic alternatives:" block.
    """
    first_line = (raw or "").split("\n")[0]
    name = LEADING_MARKER_RE.sub("", first_line).strip()
    # Trailing footnote markers, possibly several: "ceftriaxone* a", "doxycycline a"
    name = re.sub(r"\s*\[c\]\s*$", "", name)
    name = re.sub(r"[\*\u2020\u00b0]+\s*$", "", name)
    name = re.sub(r"\s+a$", "", name)
    name = re.sub(r"\s*\[c\]\s*", " ", name)
    return _clean(name)


def _alternatives(raw: str) -> List[str]:
    """
    The square-box therapeutic alternatives listed under a medicine.

    Only real medicine names are returned: entries like "4th level ATC
    chemical subgroup (C09AA ACE inhibitors, plain)" name a CLASS, not a
    substitutable product, and treating one as a drug would put a nonexistent
    medicine in the graph.
    """
    lines = (raw or "").split("\n")
    out: List[str] = []
    seen_header = False
    for line in lines:
        stripped = line.strip()
        if ALTERNATIVES_HEADER_RE.search(stripped):
            seen_header = True
            continue
        if not seen_header:
            continue
        if not stripped.startswith("-"):
            continue
        candidate = _clean(stripped.lstrip("-").strip())
        candidate = re.sub(r"\s*\(.*?\)\s*$", "", candidate).strip()
        # Alternatives carry the same footnote markers as the entries they
        # sit under ("- atenolol*"); left in, they would create a separate
        # :Medicine node from the atenolol listed elsewhere in the document.
        candidate = _medicine_name(candidate)
        if not candidate or ATC_CLASS_RE.search(candidate):
            continue
        # "lamivudine (for emtricitabine)" -> lamivudine (already stripped above)
        out.append(candidate)
    return list(dict.fromkeys(out))


def _choices(cell: str) -> List[str]:
    """Indications from a 'FIRST CHOICE' / 'SECOND CHOICE' antibiotic cell."""
    lines = (cell or "").split("\n")
    out: List[str] = []
    buffer = ""
    for line in lines:
        stripped = line.strip()
        if not stripped or re.match(r"^(FIRST|SECOND)\s+CHOICE$", stripped, re.I):
            continue
        if stripped.startswith(("\u2212", "-", "\u2013", "\u2014")):
            if buffer:
                out.append(_clean(buffer))
            buffer = stripped.lstrip("\u2212-\u2013\u2014").strip()
        else:
            # continuation of a wrapped indication
            buffer = f"{buffer} {stripped}".strip()
    if buffer:
        out.append(_clean(buffer))
    return [re.sub(r"\s*\[c\]\s*$", "", i).strip() for i in out if i]


def _detect_population(pdf: "pdfplumber.PDF") -> str:
    front = "\n".join((page.extract_text() or "") for page in pdf.pages[:4])
    return "children" if CHILDREN_LIST_RE.search(front) else "adult"


def extract_age_restrictions(pdf: "pdfplumber.PDF") -> List[Dict[str, str]]:
    """
    Table 1.1 — medicines with an age or weight restriction. A clean
    two-column table near the end of the document; located by its heading
    rather than by a fixed page number, since the two lists paginate
    differently.
    """
    restrictions: List[Dict[str, str]] = []
    for page in pdf.pages:
        text = page.extract_text() or ""
        if "Medicines with age or weight restrictions" not in text:
            continue
        for table in page.extract_tables() or []:
            for row in table:
                cells = [_clean(c) for c in row]
                non_empty = [c for c in cells if c]
                if len(non_empty) != 2:
                    continue
                name, restriction = non_empty
                if name.lower().startswith("table") or "restriction" in name.lower():
                    continue
                restrictions.append({
                    "name": _medicine_name(name),
                    "restriction": restriction,
                    "source_page": page.page_number,
                })
    return restrictions


def extract_full_list(pdf_path: str) -> Dict[str, Any]:
    """
    Parses an entire EML PDF into
    {"population", "list_title", "entries": [...], "age_restrictions": [...]}.

    Each entry: {name, dosage_form, section, section_title, list_type,
                 alternatives, aware_group, first_choice, second_choice,
                 source_page}
    """
    entries: List[Dict[str, Any]] = []
    section_number = section_title = None
    list_type = "core"
    aware_group = None

    logger.info("extract: opening '%s'", pdf_path)
    started = time.perf_counter()

    with pdfplumber.open(pdf_path) as pdf:
        population = _detect_population(pdf)
        list_title = (pdf.pages[2].extract_text() or "").split("\n")[0].strip()
        logger.info(
            "extract: '%s' — %d page(s), population=%s", pdf_path, len(pdf.pages), population,
        )

        for page in pdf.pages[:MAX_PAGES]:
            for table in page.extract_tables() or []:
                for row in table:
                    cells = [_clean(c) for c in row]
                    non_empty = [c for c in cells if c]
                    if not non_empty:
                        continue

                    # --- heading rows occupy the row alone ------------------
                    if len(non_empty) == 1:
                        marker = non_empty[0]
                        section_match = SECTION_RE.match(marker)
                        if section_match:
                            section_number, section_title = section_match.groups()
                            list_type, aware_group = "core", None
                            continue
                        sub_match = SUBSECTION_RE.match(marker)
                        if sub_match:
                            section_number, section_title = sub_match.groups()
                            list_type = "core"
                            aware_group = next(
                                (label for key, label in AWARE_GROUPS.items()
                                 if f"{key} group antibiotics" in section_title.lower()),
                                None,
                            )
                            continue
                        if COMPLEMENTARY_RE.match(marker):
                            list_type = "complementary"
                            continue
                        continue

                    # --- a FIRST/SECOND CHOICE row belongs to the medicine
                    #     above it: its name cell is empty ------------------
                    raw_name = row[0] if row else ""
                    if not _clean(raw_name) and entries:
                        joined = "\n".join(_c for _c in row if _c)
                        if re.search(r"(FIRST|SECOND)\s+CHOICE", joined, re.I):
                            for cell in row:
                                if not cell:
                                    continue
                                if re.search(r"FIRST\s+CHOICE", cell, re.I):
                                    entries[-1]["first_choice"] += _choices(cell)
                                elif re.search(r"SECOND\s+CHOICE", cell, re.I):
                                    entries[-1]["second_choice"] += _choices(cell)
                        continue

                    if section_number is None:
                        continue  # front matter, before the list proper

                    name = _medicine_name(raw_name)
                    if not name or len(name) > 120:
                        continue

                    entries.append({
                        "name": name,
                        "dosage_form": _clean(" ".join(non_empty[1:]))[:900],
                        "section": section_number,
                        "section_title": _clean(section_title or ""),
                        "list_type": list_type,
                        "alternatives": _alternatives(raw_name),
                        "aware_group": aware_group,
                        "first_choice": [],
                        "second_choice": [],
                        "source_page": page.page_number,
                    })

        age_restrictions = extract_age_restrictions(pdf)

    sections = sorted({e["section"] for e in entries})
    logger.info(
        "extract: '%s' done in %.0fms — %d entrie(s) across %d section(s), "
        "%d age restriction(s)",
        pdf_path, (time.perf_counter() - started) * 1000,
        len(entries), len(sections), len(age_restrictions),
    )
    return {
        "population": population,
        "list_title": list_title,
        "entries": entries,
        "age_restrictions": age_restrictions,
    }


def ingest_full_list(parsed: Dict[str, Any], source_document: str) -> Dict[str, int]:
    """Idempotent (MERGE-based) load of a whole EML into Neo4j."""
    entries = parsed["entries"]
    if not entries:
        logger.warning("ingest: '%s' produced 0 entries — nothing written", source_document)
        return {"entries": 0}

    logger.info(
        "ingest: writing %d medicine listing(s) from '%s' (population=%s)",
        len(entries), source_document, parsed["population"],
    )

    with session_scope("ingest_full_eml") as session:
        run_write(
            session, "ingest_full_eml", f"MERGE source '{source_document}'",
            """
            MERGE (s:SourceDocument {filename: $source_document})
              SET s.population = $population, s.list_title = $list_title
            """,
            source_document=source_document,
            population=parsed["population"],
            list_title=parsed["list_title"],
        )

        run_write(
            session, "ingest_full_eml", f"MERGE {len(entries)} listing(s) + sections",
            """
            MATCH (s:SourceDocument {filename: $source_document})
            UNWIND $rows AS row
            MERGE (m:Medicine {name: toLower(row.name)})
              ON CREATE SET m.display_name = row.name
            MERGE (m)-[l:LISTED_IN]->(s)
              SET l.dosage_form = row.dosage_form,
                  l.list_type = row.list_type,
                  l.section = row.section,
                  l.source_page = row.source_page
            MERGE (sec:Section {id: row.section + '|' + $population})
              SET sec.number = row.section,
                  sec.title = row.section_title,
                  sec.population = $population
            MERGE (m)-[:IN_SECTION]->(sec)
            """,
            source_document=source_document,
            population=parsed["population"],
            rows=entries,
        )

        alt_rows = [e for e in entries if e["alternatives"]]
        if alt_rows:
            run_write(
                session, "ingest_full_eml", f"link alternatives for {len(alt_rows)} medicine(s)",
                """
                UNWIND $rows AS row
                MATCH (m:Medicine {name: toLower(row.name)})
                UNWIND row.alternatives AS alt
                MERGE (a:Medicine {name: toLower(alt)})
                  ON CREATE SET a.display_name = alt
                MERGE (m)-[:HAS_ALTERNATIVE]->(a)
                """,
                rows=alt_rows,
            )

        aware_rows = [e for e in entries if e["aware_group"]]
        if aware_rows:
            run_write(
                session, "ingest_full_eml", f"link AWaRe group for {len(aware_rows)} antibiotic(s)",
                """
                UNWIND $rows AS row
                MATCH (m:Medicine {name: toLower(row.name)})
                MERGE (g:AWaReGroup {name: row.aware_group})
                MERGE (m)-[:IN_AWARE_GROUP]->(g)
                """,
                rows=aware_rows,
            )

        for field, relationship in (("first_choice", "FIRST_CHOICE_FOR"),
                                    ("second_choice", "SECOND_CHOICE_FOR")):
            rows = [e for e in entries if e[field]]
            if not rows:
                continue
            run_write(
                session, "ingest_full_eml",
                f"link {relationship} for {len(rows)} medicine(s)",
                f"""
                UNWIND $rows AS row
                MATCH (m:Medicine {{name: toLower(row.name)}})
                UNWIND row.{field} AS indication
                MERGE (i:Indication {{name: toLower(indication)}})
                  ON CREATE SET i.display_name = indication
                MERGE (m)-[:{relationship}]->(i)
                """,
                rows=rows,
            )

        restrictions = parsed["age_restrictions"]
        if restrictions:
            run_write(
                session, "ingest_full_eml",
                f"MERGE {len(restrictions)} age/weight restriction(s)",
                """
                UNWIND $rows AS row
                MERGE (m:Medicine {name: toLower(row.name)})
                  ON CREATE SET m.display_name = row.name
                MERGE (r:AgeRestriction {id: toLower(row.name) + '|' + row.restriction})
                  SET r.restriction = row.restriction,
                      r.source_page = row.source_page,
                      r.population = $population
                MERGE (m)-[:HAS_RESTRICTION]->(r)
                """,
                rows=restrictions,
                population=parsed["population"],
            )

    logger.info(
        "ingest: '%s' complete — %d listing(s), %d alternative link(s), "
        "%d restriction(s)",
        source_document, len(entries), len(alt_rows), len(parsed["age_restrictions"]),
    )
    return {
        "entries": len(entries),
        "alternatives": len(alt_rows),
        "age_restrictions": len(parsed["age_restrictions"]),
    }


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def medicines_in_section(section_title_fragment: str) -> List[Dict[str, Any]]:
    """
    Every medicine WHO files under a section whose title contains the given
    text — e.g. "Opioid analgesics". This is the authoritative replacement for
    a hand-curated drug-class list: it can be cited.
    """
    with session_scope("medicines_in_section") as session:
        return run_read(
            session, "medicines_in_section", f"section like '{section_title_fragment}'",
            """
            MATCH (m:Medicine)-[:IN_SECTION]->(s:Section)
            WHERE toLower(s.title) CONTAINS toLower($fragment)
            RETURN DISTINCT m.name AS name, m.display_name AS display_name,
                   s.number AS section, s.title AS section_title
            ORDER BY name
            """,
            fragment=section_title_fragment,
        )


def lookup_age_restrictions(drug_names: List[str]) -> List[Dict[str, Any]]:
    """Age/weight restrictions WHO records for any of these drugs."""
    names = [n for n in drug_names if n]
    if not names:
        return []
    with session_scope("lookup_age_restrictions") as session:
        return run_read(
            session, "lookup_age_restrictions", f"check {len(names)} drug name(s)",
            """
            UNWIND $names AS wanted
            MATCH (m:Medicine {name: toLower(wanted)})-[:HAS_RESTRICTION]->(r:AgeRestriction)
            RETURN DISTINCT wanted AS wanted, m.display_name AS display_name,
                   r.restriction AS restriction, r.source_page AS source_page,
                   r.population AS population
            ORDER BY wanted
            """,
            names=names,
        )


# Curated classes in reference_library.py, mapped to the WHO section whose
# title covers the same ground. WHO's list is deliberately MINIMAL — it names
# the essential medicines for a basic health system, not every member of a
# pharmacological class — so section 2.2 lists four opioids where the curated
# set has twenty. These sections therefore corroborate class membership; they
# cannot replace it. Swapping the curated list for this one would silently
# drop oxycodone and tramadol from opioid detection, which is a safety
# regression, not a provenance improvement.
CLASS_SECTIONS = {
    "opioid": "Opioid analgesics",
    "benzodiazepine": "Medicines for anxiety disorders",
    "opioid_reversal": "Specific",
}


def corroborate_class_membership() -> Dict[str, Dict[str, List[str]]]:
    """
    For each curated drug class, which members WHO's own sectioning confirms
    and which rest on the curated list alone.

    A confirmed member can be cited to a page of a published WHO list; an
    unconfirmed one cannot, and stays honestly labelled as curated. This is a
    per-drug provenance split, replacing a single blanket disclaimer over the
    whole class.
    """
    from reference_library import DRUG_CLASSES

    report: Dict[str, Dict[str, List[str]]] = {}
    for class_name, members in DRUG_CLASSES.items():
        fragment = CLASS_SECTIONS.get(class_name)
        who_names = set()
        if fragment:
            who_names = {row["name"] for row in medicines_in_section(fragment)}
            # A square-box alternative is WHO stating the drug substitutes for
            # a listed one, which places it in the same class.
            for row in medicines_in_section(fragment):
                with session_scope("corroborate") as session:
                    who_names.update(
                        alt["name"] for alt in run_read(
                            session, "corroborate", f"alternatives to {row['name']}",
                            """
                            MATCH (:Medicine {name: $name})-[:HAS_ALTERNATIVE]->(a:Medicine)
                            RETURN a.name AS name
                            """,
                            name=row["name"],
                        )
                    )
        report[class_name] = {
            "who_confirmed": sorted(m for m in members if m in who_names),
            "curated_only": sorted(m for m in members if m not in who_names),
        }
    return report


def loaded_summary() -> Dict[str, Any]:
    with session_scope("eml_summary") as session:
        counts = run_read(
            session, "eml_summary", "node counts",
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC",
        )
        sections = run_read(
            session, "eml_summary", "sections",
            """
            MATCH (s:Section)
            RETURN s.number AS number, s.title AS title, s.population AS population,
                   count{(s)<-[:IN_SECTION]-()} AS medicines
            ORDER BY toFloat(split(s.number,'.')[0]), s.number
            """,
        )
    return {"counts": counts, "sections": sections}


if __name__ == "__main__":
    import argparse
    import os
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)-12s %(message)s")
    logging.getLogger("pdfminer").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Ingest or inspect a full WHO EML in Neo4j.")
    parser.add_argument("--ingest", nargs="+", metavar="PDF",
                        help="one or more EML PDFs to load (idempotent)")
    parser.add_argument("--dry-run", action="store_true",
                        help="parse and report, without writing to Neo4j")
    args = parser.parse_args()

    if args.ingest:
        for path in args.ingest:
            parsed = extract_full_list(path)
            print(f"\n{os.path.basename(path)}  [{parsed['population']}]")
            print(f"  entries          : {len(parsed['entries'])}")
            print(f"  sections         : {len({e['section'] for e in parsed['entries']})}")
            print(f"  with alternatives: {sum(1 for e in parsed['entries'] if e['alternatives'])}")
            print(f"  age restrictions : {len(parsed['age_restrictions'])}")
            if not args.dry_run:
                ingest_full_list(parsed, source_document=os.path.basename(path))

    if not args.dry_run:
        summary = loaded_summary()
        print("\nGraph contents:")
        for row in summary["counts"]:
            print(f"  {str(row['label']):18} {row['n']}")
