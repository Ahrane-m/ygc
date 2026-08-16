"""
Antidote / poisoning-reference knowledge graph
=========================================
Extracts Section 4 ("Antidotes and other substances used in poisonings")
from a WHO Model List of Essential Medicines PDF and loads it into Neo4j
as a small, literal reference graph:

    (:Medicine)-[:LISTED_IN {dosage_form, list_type, category}]->(:SourceDocument)
    (:Medicine)-[:LISTED_UNDER]->(:AntidoteCategory)

WHO publishes two parallel lists -- the main EML (adults) and the EMLc
(children) -- and they disagree: the adult list carries 15 antidote
entries to the children's 10, and the dosage-form text differs for drugs
in both. So dosage_form / list_type / category are properties of a
*listing* (a medicine within one document), NOT of the medicine itself;
storing them on the node would let whichever PDF was ingested last
silently overwrite the other. Only the drug identity is shared.

Deliberately NOT an LLM extraction: every field is copied verbatim from
a table cell. The source document lists antidote/poisoning-management
drug names and dosage forms, but never states which poison each one
treats -- that pairing is general medical knowledge, not printed text,
so inventing it here (via an LLM or otherwise) would put unverified
medical claims in the graph. Keeping extraction deterministic means
there's nothing to hallucinate.

Usage:
    section = extract_antidote_section("who_eml.pdf")
    ingest_antidote_entries(section, source_document="who_eml.pdf")
    lookup_antidote_references(["naloxone", "ibuprofen"])   # bulk, one round trip
    lookup_antidote_reference("naloxone")                   # single-drug wrapper
"""

import re
from typing import Any, Dict, List, Optional

import pdfplumber

from graph_db import database_name, get_driver

SECTION_HEADING_RE = re.compile(
    r"^\d+\.\s*ANTIDOTES AND OTHER SUBSTANCES USED IN POISONINGS",
    re.IGNORECASE | re.MULTILINE,
)
NEXT_TOP_LEVEL_SECTION_RE = re.compile(r"^\d+\.\s+[A-Z]", re.MULTILINE)
SUBSECTION_RE = re.compile(r"^\d+\.\d+\s+(Non-specific|Specific)", re.IGNORECASE)
# The EMLc titles itself "...Essential Medicines for Children"; the adult
# EML never does. Read off the document itself rather than the filename,
# which a caller can name anything.
CHILDREN_LIST_RE = re.compile(
    r"Model List of Essential Medicines for Children", re.IGNORECASE
)
MAX_SECTION_PAGES = 5  # safety cap; the antidotes section is always 1-2 pages
TITLE_SCAN_PAGES = 4   # the title/citation block is always in the front matter


def _find_section_pages(pdf: "pdfplumber.PDF") -> List[int]:
    """Returns the 0-indexed page numbers spanning Section 4, by locating
    its heading and stopping at the next top-level numbered section."""
    start = None
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        if SECTION_HEADING_RE.search(text):
            start = i
            break
    if start is None:
        return []

    pages = [start]
    for i in range(start + 1, min(start + MAX_SECTION_PAGES, len(pdf.pages))):
        text = pdf.pages[i].extract_text() or ""
        # A page that opens with a *different* top-level heading (e.g.
        # "5. MEDICINES FOR...") means Section 4 has ended.
        if NEXT_TOP_LEVEL_SECTION_RE.search(text) and not SECTION_HEADING_RE.search(text):
            break
        pages.append(i)
    return pages


def _clean(text: str) -> str:
    """Collapse the newlines pdfplumber preserves inside a wrapped table
    cell, so a name is usable as a stable graph key."""
    return re.sub(r"\s+", " ", text).strip()


def _detect_population(pdf: "pdfplumber.PDF") -> str:
    """"children" for the EMLc, "adult" for the main EML -- decided by the
    document's own title text, not by the filename."""
    front_matter = "\n".join(
        (page.extract_text() or "") for page in pdf.pages[:TITLE_SCAN_PAGES]
    )
    return "children" if CHILDREN_LIST_RE.search(front_matter) else "adult"


def extract_antidote_section(pdf_path: str) -> Dict[str, Any]:
    """Extracts every (medicine name, dosage form) row from Section 4,
    tagged with which subsection (non_specific/specific) and list
    (core/complementary) it came from, plus which population (adult /
    children) the containing document covers. Nothing here is inferred --
    every entry field is copied verbatim from a table cell.

    Returns {"population": str, "entries": [...]}.
    """
    entries: List[Dict[str, Any]] = []
    subsection = None
    list_type = "core"
    # The WHO PDF renders several sections onto one page, and
    # extract_tables() returns all of them, so rows are only collected
    # between the Section 4 heading and the next top-level heading.
    in_section = False

    with pdfplumber.open(pdf_path) as pdf:
        population = _detect_population(pdf)
        for page_index in _find_section_pages(pdf):
            page = pdf.pages[page_index]
            for table in page.extract_tables() or []:
                for row in table:
                    cells = [_clean(c or "") for c in row]
                    non_empty = [c for c in cells if c]
                    if not non_empty:
                        continue

                    if len(non_empty) == 1:
                        marker = non_empty[0]
                        if SECTION_HEADING_RE.match(marker):
                            in_section = True
                            continue
                        if NEXT_TOP_LEVEL_SECTION_RE.match(marker):
                            in_section = False
                            continue
                        if not in_section:
                            continue
                        sub_match = SUBSECTION_RE.match(marker)
                        if sub_match:
                            subsection = sub_match.group(1).lower().replace("-", "_")
                            continue
                        if marker.lower() == "complementary list":
                            list_type = "complementary"
                            continue
                        continue

                    if not in_section:
                        continue

                    name, dosage_form = non_empty[0], " ".join(non_empty[1:])
                    entries.append({
                        "name": name,
                        "dosage_form": dosage_form,
                        "subsection": subsection,
                        "list_type": list_type,
                        "source_page": page.page_number,
                    })
    return {"population": population, "entries": entries}


def ingest_antidote_entries(section: Dict[str, Any], source_document: str) -> int:
    """Idempotent (MERGE-based) load into Neo4j -- re-ingesting the same
    PDF updates the existing listing in place rather than duplicating it.

    Per-listing facts live on the :LISTED_IN relationship, so ingesting the
    adult EML and the children's EMLc leaves two independent listings per
    shared drug instead of one overwriting the other.
    """
    entries = section["entries"]
    if not entries:
        return 0
    with get_driver().session(database=database_name()) as session:
        session.run(
            """
            MERGE (s:SourceDocument {filename: $source_document})
              SET s.population = $population
            WITH s
            UNWIND $rows AS row
            MERGE (m:Medicine {name: toLower(row.name)})
              SET m.display_name = row.name
            MERGE (m)-[l:LISTED_IN]->(s)
              SET l.dosage_form = row.dosage_form,
                  l.list_type = row.list_type,
                  l.category = row.subsection,
                  l.source_page = row.source_page
            MERGE (c:AntidoteCategory {name: row.subsection})
            MERGE (m)-[:LISTED_UNDER]->(c)
            """,
            rows=entries,
            source_document=source_document,
            population=section["population"],
        )
    return len(entries)


def lookup_antidote_references(drug_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Case-insensitive bulk lookup, keyed by the caller's original spelling.

    One round trip for the whole medication list rather than one per drug --
    a patient on 20 medications would otherwise mean 20 serial queries to a
    cloud database on every upload.

    Each hit carries every listing found, so a caller can see that e.g.
    naloxone is listed for both adults and children (and with which dosage
    forms), rather than getting whichever document was loaded last.
    """
    names = [n for n in drug_names if n]
    if not names:
        return {}
    with get_driver().session(database=database_name()) as session:
        records = session.run(
            """
            UNWIND $names AS wanted
            MATCH (m:Medicine {name: toLower(wanted)})-[l:LISTED_IN]->(s:SourceDocument)
            RETURN wanted AS wanted, m.display_name AS display_name,
                   l.category AS category, l.dosage_form AS dosage_form,
                   l.list_type AS list_type, s.filename AS source_document,
                   s.population AS population
            ORDER BY wanted, s.population
            """,
            names=names,
        ).data()

    found: Dict[str, Dict[str, Any]] = {}
    for r in records:
        entry = found.setdefault(r["wanted"], {
            "display_name": r["display_name"],
            "category": r["category"],
            "listings": [],
        })
        entry["listings"].append(
            {k: r[k] for k in ("population", "source_document", "list_type", "dosage_form")}
        )
    return found


def lookup_antidote_reference(drug_name: str) -> Optional[Dict[str, Any]]:
    """Single-drug convenience wrapper around lookup_antidote_references()."""
    return lookup_antidote_references([drug_name]).get(drug_name)
