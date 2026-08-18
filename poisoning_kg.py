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

import logging
import re
import time
from typing import Any, Dict, List, Optional

import pdfplumber

from graph_db import run_read, run_write, session_scope

logger = logging.getLogger("poisoning_kg")

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

    # Logged step by step because this runs BEFORE Neo4j is touched: when an
    # ingest loads zero entries, these lines are what separate "the PDF has no
    # antidote section" from "the section was found but its tables didn't
    # parse" — two failures that look identical from the graph's side.
    logger.info("extract: opening '%s'", pdf_path)
    started = time.perf_counter()

    with pdfplumber.open(pdf_path) as pdf:
        population = _detect_population(pdf)
        logger.info(
            "extract: '%s' has %d page(s), population=%s",
            pdf_path, len(pdf.pages), population,
        )

        section_pages = _find_section_pages(pdf)
        if not section_pages:
            logger.warning(
                "extract: no 'Antidotes and other substances used in poisonings' "
                "section heading found in '%s' — nothing to ingest", pdf_path,
            )
        else:
            logger.info(
                "extract: antidote section spans page(s) %s",
                ", ".join(str(p + 1) for p in section_pages),
            )

        for page_index in section_pages:
            page = pdf.pages[page_index]
            entries_before = len(entries)
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
                            # Positive parse milestones: these markers decide
                            # how every row after them is categorised, so
                            # seeing them fire is how you confirm entries were
                            # filed under the right subsection/list.
                            logger.debug(
                                "extract: entered subsection '%s' on page %d",
                                subsection, page.page_number,
                            )
                            continue
                        if marker.lower() == "complementary list":
                            list_type = "complementary"
                            logger.debug(
                                "extract: switched to complementary list on page %d",
                                page.page_number,
                            )
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
                    # Every successfully parsed row, at DEBUG — the per-page
                    # counts below are the INFO-level summary of these.
                    logger.debug(
                        "extract: + %s (%s) [%s/%s] p%d",
                        name, dosage_form or "no dosage form", subsection or "uncategorised",
                        list_type, page.page_number,
                    )

            logger.info(
                "extract: page %d yielded %d entrie(s)",
                page.page_number, len(entries) - entries_before,
            )

    categories = sorted({e["subsection"] for e in entries if e["subsection"]})
    logger.info(
        "extract: '%s' done in %.0fms — %d entrie(s), population=%s, categories=%s",
        pdf_path, (time.perf_counter() - started) * 1000, len(entries), population,
        categories or "none",
    )
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
        # Logged rather than returning quietly: a zero-entry ingest means the
        # PDF parse found nothing, and without this line the graph simply
        # looks unchanged for no stated reason.
        logger.warning(
            "ingest: '%s' has 0 entrie(s) — skipping Neo4j write entirely",
            source_document,
        )
        return 0

    logger.info(
        "ingest: preparing to write %d entrie(s) from '%s' (population=%s) to Neo4j",
        len(entries), source_document, section["population"],
    )

    with session_scope("ingest_antidote_entries") as session:
        summary = run_write(
            session,
            "ingest_antidote_entries",
            f"MERGE {len(entries)} listing(s) for '{source_document}'",
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

    # The load is MERGE-based and idempotent, so BOTH outcomes are successes
    # and both are stated plainly: a first load creates nodes, a re-ingest
    # creates nothing and updates in place. Logging only the second would
    # make the normal re-ingest look like a silent failure; logging only the
    # first would leave a successful fresh load unremarked.
    counters = getattr(summary, "counters", None)
    if counters is None:
        pass
    elif counters.nodes_created or counters.relationships_created:
        logger.info(
            "ingest: '%s' loaded successfully — created %d node(s) and %d "
            "relationship(s), set %d propert(ies)",
            source_document, counters.nodes_created, counters.relationships_created,
            counters.properties_set,
        )
    else:
        logger.info(
            "ingest: '%s' loaded successfully — created no new nodes or relationships "
            "because this document was already in the graph; %d propert(ies) updated "
            "in place",
            source_document, counters.properties_set,
        )
    logger.info("ingest: '%s' complete — %d entrie(s) submitted", source_document, len(entries))
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
        # No query is sent at all. Logged because api.py treats an empty
        # result and an unreachable graph identically — this line marks which
        # of the two happened.
        logger.info("lookup: no drug names supplied — no Neo4j query sent")
        return {}

    logger.info("lookup: checking %d drug name(s) against the antidote graph", len(names))

    # A prescription prints "Naloxone Hydrochloride 400mcg"; the graph keys on
    # "naloxone". Matching only the printed form misses, and the miss looks
    # exactly like the drug genuinely not being in the reference data — so the
    # citation is lost silently and the finding drops to unverified model
    # recall. Every variant is offered and the original spelling is carried
    # through as the result key.
    from document_dedup import name_variants

    pairs = [
        {"wanted": name, "candidate": variant, "exact": variant == name.strip().lower()}
        for name in names for variant in name_variants(name)
    ]

    with session_scope("lookup_antidote_references") as session:
        records = run_read(
            session,
            "lookup_antidote_references",
            f"match {len(names)} medicine name(s)",
            """
            UNWIND $pairs AS pair
            MATCH (m:Medicine {name: pair.candidate})-[l:LISTED_IN]->(s:SourceDocument)
            RETURN pair.wanted AS wanted, pair.exact AS exact_match,
                   m.display_name AS display_name,
                   l.category AS category, l.dosage_form AS dosage_form,
                   l.list_type AS list_type, s.filename AS source_document,
                   s.population AS population
            ORDER BY wanted, s.population
            """,
            pairs=pairs,
        )

    found: Dict[str, Dict[str, Any]] = {}
    for r in records:
        entry = found.setdefault(r["wanted"], {
            "display_name": r["display_name"],
            "category": r["category"],
            "matched_exactly": r["exact_match"],
            "listings": [],
        })
        entry["listings"].append(
            {k: r[k] for k in ("population", "source_document", "list_type", "dosage_form")}
        )

    if found:
        # Exact and normalized hits are logged apart. A normalized hit means
        # the printed name did NOT match and only did so after salts and doses
        # were stripped — worth seeing, because it is the case that used to
        # fail silently.
        normalized = sorted(k for k, v in found.items() if not v["matched_exactly"])
        logger.info(
            "lookup: %d of %d drug name(s) are listed as antidotes: %s",
            len(found), len(names), ", ".join(sorted(found)),
        )
        if normalized:
            logger.info(
                "lookup: %d matched only after normalizing the printed name: %s",
                len(normalized), ", ".join(normalized),
            )
        # Each hit spelled out, at DEBUG. A drug listed in both the adult EML
        # and the children's EMLc returns two listings, and this is where you
        # confirm both came back rather than whichever was ingested last.
        for wanted, ref in sorted(found.items()):
            for listing in ref["listings"]:
                logger.debug(
                    "lookup:   %s -> %s [%s] %s (%s)",
                    wanted, ref["display_name"], listing["population"],
                    listing["dosage_form"] or "no dosage form",
                    listing["source_document"],
                )
    else:
        logger.info(
            "lookup: none of the %d drug name(s) are listed in the antidote graph "
            "(the graph was reached — this is a genuine no-match, not a failure)",
            len(names),
        )
    return found


def lookup_antidote_reference(drug_name: str) -> Optional[Dict[str, Any]]:
    """Single-drug convenience wrapper around lookup_antidote_references()."""
    return lookup_antidote_references([drug_name]).get(drug_name)
