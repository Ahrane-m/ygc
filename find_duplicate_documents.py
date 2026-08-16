"""
Duplicate Document Review (read-only)
=========================================
Lists documents already stored for a user that record the SAME physical
prescription — the same file uploaded twice, or a scan and a phone photo of
one page. Nothing is modified: this only reports what is there.

Existing records predate the upload-time duplicate check, so a record built
before it can still hold several copies of one prescription. Every copy makes
each medication on it look prescribed again, which the cross-check reads as a
duplicate and the consult triage turns into a pharmacist referral — so it is
worth knowing what is in there.

Usage:
    python find_duplicate_documents.py --user <user_id>
    python find_duplicate_documents.py --user <user_id> --json

Reads MongoDB via db.py (MONGODB_URI). Cloudinary originals are never
touched, and nothing is deleted — if you decide to remove a document after
reviewing this, do it deliberately with its cloudinary_public_id and _id in
hand.
"""

import argparse
import json
import sys

from dotenv import load_dotenv

load_dotenv()

import db  # noqa: E402  (must follow load_dotenv — reads MONGODB_URI on use)
from document_dedup import find_duplicate_document_groups  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List duplicate/re-uploaded documents for one user (read-only)."
    )
    parser.add_argument("--user", required=True, help="user_id to inspect")
    parser.add_argument("--json", action="store_true", help="emit raw JSON instead of a report")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    docs = db.load_documents(args.user)
    if not docs:
        print(f"No documents found for user {args.user}.")
        return 0

    groups = find_duplicate_document_groups(docs)

    if args.json:
        print(json.dumps({
            "user_id": args.user,
            "documents_total": len(docs),
            "duplicate_groups": groups,
        }, indent=2, default=str))
        return 0

    print(f"User {args.user}: {len(docs)} document(s) on file.")
    if not groups:
        print("No duplicate documents found — every document is a distinct prescription.")
        return 0

    redundant = sum(len(g["documents"]) - 1 for g in groups)
    print(
        f"\n{len(groups)} prescription(s) are stored more than once "
        f"({redundant} redundant document(s) in total).\n"
    )

    for i, group in enumerate(groups, start=1):
        kind = (
            "the same file uploaded more than once"
            if group["identical_files"]
            else "different files of the same prescription (e.g. a scan and a photo)"
        )
        print(f"GROUP {i} — {kind}")
        print(f"  medications: {', '.join(group['medications']) or 'none recorded'}")
        for j, doc in enumerate(group["documents"]):
            marker = "  KEEP  " if j == 0 else "  extra "
            print(f"  {marker} {doc['source_file'] or 'unknown file'}")
            print(f"           date printed on document : {doc['date'] or 'not recorded'}")
            print(f"           uploaded at              : {doc['uploaded_at'] or 'unknown'}")
            print(f"           cloudinary_public_id     : {doc['cloudinary_public_id'] or 'none'}")
            if doc["content_sha256"]:
                print(f"           content sha256           : {doc['content_sha256'][:16]}…")
        print()

    print(
        "Nothing has been changed. 'KEEP' simply marks the first document in each\n"
        "group — review them before removing anything, since the copies can differ\n"
        "in extraction quality and you may prefer to keep a different one.\n"
        "\n"
        "New uploads are already protected: an identical file is now recognised and\n"
        "skipped, and documents recording one prescription share a prescription_group\n"
        "so they no longer produce duplicate-prescription findings."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
