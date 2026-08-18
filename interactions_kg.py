"""
FDA drug-interaction reference — enzyme roles in the knowledge graph
====================================================================
Loads the FDA's "Examples of Drugs that Interact with CYP Enzymes and
Transporter Systems" table into Neo4j:

    (:Medicine)-[:INHIBITS   {strength}]->(:Enzyme)
    (:Medicine)-[:INDUCES    {strength}]->(:Enzyme)
    (:Medicine)-[:METABOLIZED_BY {sensitivity}]->(:Enzyme)
    (:Enzyme {name, kind})                       kind = cyp | transporter
    (:PotencyDefinition)-[:DEFINED_IN]->(:SourceDocument)

WHAT THIS SOURCE DOES AND DOES NOT SAY
--------------------------------------
It does NOT say "drug A interacts with drug B". Not once. It says:

    alfentanil   3A sensitive substrate
    adagrasib    3A strong inhibitor

The interaction between those two exists only when you JOIN them on the
shared enzyme. That join is standard pharmacology and it is exactly what a
graph is good at — but the resulting pair is DERIVED, not quoted, and the
distinction is the whole point of this module.

So derived pairs are never written as edges. There is no
`(:Medicine)-[:INTERACTS_WITH]->(:Medicine)` in this schema, deliberately:
storing one would make an inference indistinguishable from a citation the
moment anyone queried it. Pairs are computed on demand by
`potential_interactions()`, which returns them tagged
`evidence="derived_pharmacokinetic"` and `requires_clinical_review=True`,
carrying the two role statements that WERE quoted so a reader can check the
reasoning.

This matters because a shared-enzyme pair is a POTENTIAL pharmacokinetic
interaction, not a clinically significant one. CYP3A alone has 40 sensitive
substrates and 20 strong inhibitors in this table — 800 theoretical pairs,
most of them clinically unremarkable. Asserting those as cited findings would
bury the real alerts in noise and would launder model-grade inference into
reference-grade evidence, which is the failure the evidence grading in this
repository exists to prevent.

Source is a US Government work (public domain). It is a web page rather than a
PDF, so the retrieved snapshot is archived under reference_sources/ and the
retrieval date is recorded on the source node — a URL alone is not a citation
if the page changes.

Usage:
    python interactions_kg.py --ingest          # parse archived snapshot -> Neo4j
    python interactions_kg.py --refresh         # re-download the snapshot first
    python interactions_kg.py                   # show what is loaded
"""

import datetime as _dt
import html
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from graph_db import run_read, run_write, session_scope

logger = logging.getLogger("interactions_kg")

SOURCE_URL = (
    "https://www.fda.gov/drugs/drug-interactions-labeling/"
    "healthcare-professionals-fdas-examples-drugs-interact-cyp-enzymes-and-transporter-systems"
)
SOURCE_ID = "fda-cyp-transporter-examples"
SOURCE_TITLE = (
    "FDA's Examples of Drugs that Interact with CYP Enzymes and Transporter Systems"
)
SNAPSHOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "reference_sources", "fda_cyp_interactions.html"
)

# Role words, longest first so "moderately sensitive substrate" is matched
# before "substrate".
_STRENGTHS = (
    "moderately sensitive", "sensitive", "strong", "moderate", "weak",
)
_ROLES = {"inhibitor": "INHIBITS", "inducer": "INDUCES", "substrate": "METABOLIZED_BY"}
# Clinical weight order. A blank strength means the source records none
# (transporters), which ranks above weak but below moderate.
_STRENGTH_RANK = {"strong": 0, "moderate": 1, "": 2, "weak": 3}

# Enzyme names are matched against a fixed vocabulary rather than by
# splitting on punctuation, because the page fuses footnote markers straight
# onto the name: "1A220" is CYP1A2 plus footnote 20, and "OATP1B113" is
# OATP1B1 plus footnote 13. Stripping trailing digits generically would be
# worse than useless here — 1A2, 2C8, OAT1, OATP1B1 and MATE1 all legitimately
# end in a digit, so "1A2" would be mangled into "1A". Longest-first matching
# against known names resolves both cases without guessing, and anything
# outside the vocabulary is reported as unparsed rather than invented.
_CYP_ENZYMES = ("1A2", "2B6", "2C19", "2C8", "2C9", "2D6", "2E1", "3A")
_TRANSPORTERS = (
    "OATP1B1", "OATP1B3", "OATP1B", "P-gp", "BCRP", "OAT1", "OAT3",
    "OCT2", "MATE2-K", "MATE1",
)
# Longest first so OATP1B1 wins over OATP1B, and 2C19 over 2C9's prefix.
_ENZYME_VOCAB: Tuple[str, ...] = tuple(
    sorted(_CYP_ENZYMES + _TRANSPORTERS, key=len, reverse=True)
)
_ENZYME_RE = re.compile(
    "(" + "|".join(re.escape(name) for name in _ENZYME_VOCAB) + ")", re.IGNORECASE
)
_CANONICAL = {name.lower(): name for name in _ENZYME_VOCAB}
# A footnote hangs off the END of the cell, after the role word:
# "3A moderate inhibitor5", "CYP3A moderate inducer b".
_CELL_FOOTNOTE_RE = re.compile(
    r"\b(inhibitor|inducer|substrate)s?\s*\d*\s*[a-z]?$", re.IGNORECASE
)

_TAG_RE = re.compile(r"<[^>]+>")
_TABLE_RE = re.compile(r"<table.*?</table>", re.S)
_ROW_RE = re.compile(r"<tr.*?>(.*?)</tr>", re.S)
_CELL_RE = re.compile(r"<t[dh].*?>(.*?)</t[dh]>", re.S)
# Footnote digits are appended to some names: "adefovir1", "itraconazole2".
_FOOTNOTE_RE = re.compile(r"(?<=[a-z])\d+$")


def _text(fragment: str) -> str:
    """Cell text with markup, entities and non-breaking spaces resolved."""
    plain = html.unescape(_TAG_RE.sub("", fragment)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", plain).strip()


def _drug_name(raw: str) -> str:
    name = _FOOTNOTE_RE.sub("", _text(raw).lower()).strip()
    # "grapefruit juice", "St. John's wort" are legitimate entries; only
    # parenthetical qualifiers are dropped.
    return re.sub(r"\s*\(.*?\)\s*$", "", name).strip()


def parse_cell(cell: str) -> List[Dict[str, str]]:
    """
    One table cell -> the enzyme roles it states.

        "2D6; 3A weak inhibitor"  -> [ {2D6, weak, INHIBITS},
                                       {3A,  weak, INHIBITS} ]
        "P-gp inhibitor"          -> [ {P-gp, '', INHIBITS} ]

    Returns [] for an empty cell. A cell whose trailing word is not a known
    role is returned as [] rather than guessed at — an unparsed cell is a
    visible gap, a mis-parsed one is silent bad data.
    """
    value = _text(cell)
    if not value:
        return []

    # Drop any trailing footnote marker before looking for the role word.
    cleaned = _CELL_FOOTNOTE_RE.sub(lambda m: m.group(1).lower(), value)

    role_key = next((r for r in _ROLES if cleaned.lower().endswith(r)), None)
    if not role_key:
        return []
    remainder = cleaned[: -len(role_key)].strip()

    # A strength stated once at the end applies to every enzyme in the cell
    # ("2D6; 3A weak inhibitor"), but each segment may also carry its own
    # ("1A2 moderate; 2C9 weak inducer"), which overrides it.
    default_strength = ""
    for candidate in _STRENGTHS:
        if remainder.lower().endswith(candidate):
            default_strength = candidate
            remainder = remainder[: -len(candidate)].strip()
            break

    entries: List[Dict[str, str]] = []
    for segment in re.split(r"[;,]|\band\b", remainder):
        segment = segment.strip()
        if not segment:
            continue
        strength = next(
            (c for c in _STRENGTHS if c in segment.lower()), default_strength
        )
        for match in _ENZYME_RE.finditer(segment):
            enzyme = _CANONICAL[match.group(1).lower()]
            entries.append({
                "enzyme": enzyme,
                "kind": "cyp" if enzyme in _CYP_ENZYMES else "transporter",
                # The source records no potency for transporters.
                "strength": "" if enzyme in _TRANSPORTERS else strength,
                "relationship": _ROLES[role_key],
                "statement": value,
            })

    # De-duplicate: "BCRP and OATP1B transporters" can name the same enzyme
    # twice once the trailing noun is ignored.
    seen = set()
    unique = []
    for entry in entries:
        key = (entry["enzyme"], entry["relationship"])
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


def parse_snapshot(path: str = SNAPSHOT) -> Dict[str, Any]:
    """
    Parses the archived FDA page into
    {"roles": [...], "definitions": [...], "unparsed": [...]}.

    `unparsed` carries any non-empty cell the parser did not understand, so a
    change to the page's wording shows up as a reported number rather than as
    quietly missing data.
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        markup = handle.read()

    tables = _TABLE_RE.findall(markup)
    if not tables:
        raise ValueError(f"no tables found in {path} — page structure changed?")

    roles: List[Dict[str, str]] = []
    unparsed: List[str] = []
    rows = _ROW_RE.findall(tables[0])
    for row in rows[1:]:
        cells = _CELL_RE.findall(row)
        if len(cells) < 2:
            continue
        drug = _drug_name(cells[0])
        if not drug:
            continue
        for cell in cells[1:]:
            parsed = parse_cell(cell)
            if parsed:
                for entry in parsed:
                    roles.append({**entry, "drug": drug})
            elif _text(cell):
                unparsed.append(f"{drug}: {_text(cell)}")

    # Legend tables 1..n define what strong/moderate/weak mean. Without these
    # the strengths on the edges are labels with no stated meaning.
    definitions: List[Dict[str, str]] = []
    for table in tables[1:]:
        table_rows = _ROW_RE.findall(table)
        if not table_rows:
            continue
        heading = _text(_CELL_RE.findall(table_rows[0])[-1]) if _CELL_RE.findall(table_rows[0]) else ""
        for row in table_rows[1:]:
            cells = [_text(c) for c in _CELL_RE.findall(row)]
            if len(cells) >= 2 and cells[0] and cells[1]:
                definitions.append({
                    "category": cells[0], "definition": cells[1], "applies_to": heading,
                })

    logger.info(
        "parse: %d role statement(s), %d potency definition(s), %d unparsed cell(s)",
        len(roles), len(definitions), len(unparsed),
    )
    return {"roles": roles, "definitions": definitions, "unparsed": unparsed}


def ingest_interactions(parsed: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Idempotent load of the FDA enzyme-role table."""
    parsed = parsed or parse_snapshot()
    roles = parsed["roles"]
    if not roles:
        logger.warning("ingest: 0 role statements parsed — nothing written")
        return {"roles": 0}

    retrieved = _dt.datetime.fromtimestamp(os.path.getmtime(SNAPSHOT)).date().isoformat()
    logger.info("ingest: writing %d enzyme-role statement(s)", len(roles))

    with session_scope("ingest_interactions") as session:
        run_write(
            session, "ingest_interactions", f"MERGE source '{SOURCE_ID}'",
            """
            MERGE (s:SourceDocument {filename: $id})
              SET s.list_title = $title, s.url = $url, s.publisher = 'U.S. FDA',
                  s.retrieved = $retrieved, s.population = 'adult',
                  s.licence = 'US Government work, public domain'
            """,
            id=SOURCE_ID, title=SOURCE_TITLE, url=SOURCE_URL, retrieved=retrieved,
        )

        # One statement per relationship type: the type cannot be parameterised
        # in Cypher, and building it by string interpolation from table data
        # would be an injection route.
        for relationship in ("INHIBITS", "INDUCES", "METABOLIZED_BY"):
            subset = [r for r in roles if r["relationship"] == relationship]
            if not subset:
                continue
            run_write(
                session, "ingest_interactions",
                f"MERGE {len(subset)} :{relationship} edge(s)",
                f"""
                MATCH (s:SourceDocument {{filename: $id}})
                UNWIND $rows AS row
                MERGE (m:Medicine {{name: row.drug}})
                  ON CREATE SET m.display_name = row.drug
                MERGE (e:Enzyme {{name: row.enzyme}})
                  SET e.kind = row.kind
                MERGE (m)-[x:{relationship}]->(e)
                  SET x.strength = row.strength,
                      x.statement = row.statement,
                      x.source = $id
                MERGE (m)-[:LISTED_IN]->(s)
                """,
                id=SOURCE_ID, rows=subset,
            )

        definitions = parsed["definitions"]
        if definitions:
            run_write(
                session, "ingest_interactions",
                f"MERGE {len(definitions)} potency definition(s)",
                """
                MATCH (s:SourceDocument {filename: $id})
                UNWIND $rows AS row
                MERGE (d:PotencyDefinition {id: row.applies_to + '|' + row.category})
                  SET d.category = row.category, d.definition = row.definition,
                      d.applies_to = row.applies_to
                MERGE (d)-[:DEFINED_IN]->(s)
                """,
                id=SOURCE_ID, rows=definitions,
            )

    drugs = {r["drug"] for r in roles}
    enzymes = {r["enzyme"] for r in roles}
    logger.info(
        "ingest: complete — %d statement(s) covering %d drug(s) and %d enzyme(s)",
        len(roles), len(drugs), len(enzymes),
    )
    return {"roles": len(roles), "drugs": len(drugs), "enzymes": len(enzymes),
            "definitions": len(parsed["definitions"])}


# ---------------------------------------------------------------------------
# Derivation — computed, never stored
# ---------------------------------------------------------------------------

def potential_interactions(
    drug_names: Sequence[str],
    include_weak: bool = False,
) -> List[Dict[str, Any]]:
    """
    Pairs among the supplied drugs that share an enzyme as substrate and
    inhibitor/inducer.

    These are DERIVED. The FDA table states each drug's role separately and
    never pairs them, so every result is returned with
    `evidence="derived_pharmacokinetic"`, `requires_clinical_review=True`, and
    both underlying quoted statements attached.

    Only pairs drawn from the caller's own drug list are considered, so the
    result is bounded by the patient's medications rather than by the
    hundreds of theoretical pairs in the table. `include_weak` is off by
    default: weak inhibitors rarely produce a clinically meaningful change and
    including them is mostly noise.
    """
    # Printed names carry salts and doses the graph does not store; see
    # document_dedup.name_variants. Without this the pair simply never forms.
    from document_dedup import name_variants

    names = sorted({v for n in drug_names for v in name_variants(n)})
    if len(names) < 2:
        return []

    strengths = ["strong", "moderate"] + (["weak"] if include_weak else [])

    with session_scope("potential_interactions") as session:
        rows = run_read(
            session, "potential_interactions",
            f"derive pairs among {len(names)} drug(s)",
            """
            MATCH (sub:Medicine)-[s:METABOLIZED_BY]->(e:Enzyme)<-[a:INHIBITS|INDUCES]-(act:Medicine)
            WHERE sub.name IN $names AND act.name IN $names
              AND sub.name <> act.name
              AND (a.strength IN $strengths OR a.strength = '')
            RETURN act.display_name AS affecting_drug,
                   sub.display_name AS affected_drug,
                   e.name AS enzyme, e.kind AS enzyme_kind,
                   type(a) AS mechanism, a.strength AS strength,
                   a.statement AS affecting_statement,
                   s.statement AS affected_statement
            ORDER BY CASE a.strength WHEN 'strong' THEN 0 WHEN 'moderate' THEN 1
                                     ELSE 2 END, enzyme
            """,
            names=names, strengths=strengths,
        )

    # One drug pair can share several pathways — clarithromycin and
    # simvastatin share 3A, OATP1B1 and OATP1B3 — and the query returns a row
    # for each. Emitting three findings for one interaction is the same
    # false-alarm inflation this pipeline strips out elsewhere, so pairs are
    # collapsed to one result and every shared pathway is listed beneath it.
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (row["affecting_drug"], row["affected_drug"])
        entry = grouped.setdefault(key, {
            "affecting_drug": row["affecting_drug"],
            "affected_drug": row["affected_drug"],
            "pathways": [],
            "evidence": "derived_pharmacokinetic",
            "requires_clinical_review": True,
            "source": SOURCE_ID,
            "source_url": SOURCE_URL,
        })
        entry["pathways"].append({
            "enzyme": row["enzyme"], "enzyme_kind": row["enzyme_kind"],
            "mechanism": row["mechanism"], "strength": row["strength"],
            "affecting_statement": row["affecting_statement"],
            "affected_statement": row["affected_statement"],
        })

    results = []
    for entry in grouped.values():
        pathways = sorted(entry["pathways"], key=lambda p: _STRENGTH_RANK.get(p["strength"], 2))
        lead = pathways[0]
        direction = "raise" if lead["mechanism"] == "INHIBITS" else "lower"
        shared = ", ".join(dict.fromkeys(p["enzyme"] for p in pathways))
        entry.update({
            "pathways": pathways,
            "strength": lead["strength"],
            "mechanism": lead["mechanism"],
            "shared_pathways": shared,
            "derivation": (
                f"{entry['affecting_drug']} is recorded as \"{lead['affecting_statement']}\" "
                f"and {entry['affected_drug']} as \"{lead['affected_statement']}\". "
                f"They share {shared}, so the first may {direction} levels of the "
                f"second. The source states each role separately; it does not state "
                f"that these two drugs interact."
            ),
        })
        results.append(entry)
    results.sort(key=lambda r: (_STRENGTH_RANK.get(r["strength"], 2), r["affecting_drug"]))

    if results:
        logger.info(
            "derive: %d potential pharmacokinetic pair(s) among %d drug(s) "
            "(derived by shared-enzyme join, not quoted)",
            len(results), len(names),
        )
    return results


def enzyme_roles(drug_names: Sequence[str]) -> List[Dict[str, Any]]:
    """The quoted FDA role statements for these drugs — no derivation."""
    from document_dedup import name_variants

    names = sorted({v for n in drug_names for v in name_variants(n)})
    if not names:
        return []
    with session_scope("enzyme_roles") as session:
        return run_read(
            session, "enzyme_roles", f"roles for {len(names)} drug(s)",
            """
            MATCH (m:Medicine)-[x:INHIBITS|INDUCES|METABOLIZED_BY]->(e:Enzyme)
            WHERE m.name IN $names
            RETURN m.display_name AS drug, type(x) AS role, e.name AS enzyme,
                   e.kind AS kind, x.strength AS strength, x.statement AS statement
            ORDER BY drug, enzyme
            """,
            names=names,
        )


def loaded_summary() -> Dict[str, Any]:
    with session_scope("interactions_summary") as session:
        enzymes = run_read(
            session, "interactions_summary", "enzymes",
            """
            MATCH (e:Enzyme)
            RETURN e.name AS name, e.kind AS kind,
                   count{(e)<-[:METABOLIZED_BY]-()} AS substrates,
                   count{(e)<-[:INHIBITS]-()} AS inhibitors,
                   count{(e)<-[:INDUCES]-()} AS inducers
            ORDER BY substrates + inhibitors + inducers DESC
            """,
        )
    return {"enzymes": enzymes}


def _self_test() -> None:
    assert parse_cell("") == []
    assert parse_cell("3A sensitive substrate") == [{
        "enzyme": "3A", "kind": "cyp", "strength": "sensitive",
        "relationship": "METABOLIZED_BY", "statement": "3A sensitive substrate"}]
    two = parse_cell("2D6; 3A weak inhibitor")
    assert [e["enzyme"] for e in two] == ["2D6", "3A"]
    assert all(e["strength"] == "weak" and e["relationship"] == "INHIBITS" for e in two)
    # Transporters carry no strength and are typed apart from cytochromes.
    pgp = parse_cell("P-gp inhibitor")
    assert pgp[0]["kind"] == "transporter" and pgp[0]["strength"] == ""
    assert parse_cell("3A moderately sensitive substrate")[0]["strength"] == "moderately sensitive"
    # An unrecognised trailing word is skipped, never guessed.
    assert parse_cell("3A something else") == []
    assert _drug_name("adefovir1") == "adefovir"
    assert _drug_name("Adagrasib") == "adagrasib"

    # Footnote markers fused onto the enzyme name. Generic digit-stripping
    # would turn 1A2 into 1A and OATP1B1 into OATP1B; the vocabulary must
    # recover the real name and keep the digits that belong to it.
    assert parse_cell("1A220 inhibitor")[0]["enzyme"] == "1A2"
    assert parse_cell("OATP1B113 substrate")[0]["enzyme"] == "OATP1B1"
    assert parse_cell("2C19 strong inhibitor")[0]["enzyme"] == "2C19"   # not 2C9
    # Footnotes after the role word, digit and letter forms.
    assert parse_cell("3A moderate inhibitor5")[0]["strength"] == "moderate"
    assert parse_cell("CYP3A moderate inducer b")[0]["enzyme"] == "3A"
    # Per-segment strength overrides the cell-level default.
    mixed = parse_cell("1A2 moderate; 2C9 weak inducer")
    assert {e["enzyme"]: e["strength"] for e in mixed} == {"1A2": "moderate", "2C9": "weak"}
    # "and" separates enzymes just as ";" does.
    assert {e["enzyme"] for e in parse_cell("2C8 and 3A strong inhibitor")} == {"2C8", "3A"}
    print("All checks passed.")


if __name__ == "__main__":
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Load or inspect FDA enzyme-role data.")
    parser.add_argument("--ingest", action="store_true", help="load into Neo4j (idempotent)")
    parser.add_argument("--refresh", action="store_true", help="re-download the snapshot first")
    parser.add_argument("--self-test", action="store_true", help="parser checks, no network or DB")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        raise SystemExit(0)

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)-16s %(message)s")

    if args.refresh:
        import urllib.request

        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            os.makedirs(os.path.dirname(SNAPSHOT), exist_ok=True)
            with open(SNAPSHOT, "wb") as handle:
                handle.write(response.read())
        logger.info("refresh: snapshot updated at %s", SNAPSHOT)

    if args.ingest:
        parsed = parse_snapshot()
        if parsed["unparsed"]:
            print(f"\n{len(parsed['unparsed'])} unparsed cell(s) — review before trusting:")
            for entry in parsed["unparsed"][:10]:
                print(f"   {entry}")
        counts = ingest_interactions(parsed)
        print(f"\nLoaded {counts['roles']} role statement(s): "
              f"{counts['drugs']} drug(s), {counts['enzymes']} enzyme(s), "
              f"{counts['definitions']} potency definition(s).")

    print("\nEnzymes in the graph (substrates / inhibitors / inducers):")
    for row in loaded_summary()["enzymes"][:20]:
        print(f"  {row['name']:12} {row['kind']:12} "
              f"{row['substrates']:>3} / {row['inhibitors']:>3} / {row['inducers']:>3}")
