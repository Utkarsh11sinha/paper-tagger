"""Download paper titles, abstracts and categories from the public arXiv API.

Usage:
    python fetch_arxiv.py                      # ~2000 recent papers per category
    python fetch_arxiv.py --per-category 500   # smaller / faster

The script appends to data/raw/arxiv_papers.jsonl and skips papers it already
has, so it can be stopped and restarted. arXiv asks clients to wait ~3 seconds
between requests; please keep the default delay.
"""
import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from config import DEFAULT_CATEGORIES, RAW_PATH

API_URL = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
# arXiv categories look like "cs.CV", "quant-ph", "physics.comp-ph".
# ACM ("I.2.6") and MSC ("68T05") codes also show up as <category> and are dropped.
CATEGORY_RE = re.compile(r"^[a-z][a-z\-]*(\.[A-Za-z][A-Za-z\-]*)?$")


def fetch_page(session, category, start, size, retries=5):
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": size,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    for attempt in range(retries):
        try:
            resp = session.get(API_URL, params=params, timeout=90)
            if resp.status_code == 200:
                entries = ET.fromstring(resp.content).findall("a:entry", NS)
                # The API occasionally returns an empty page transiently.
                if entries or attempt >= 2:
                    return entries
            else:
                print(f"  HTTP {resp.status_code}, retrying...")
        except (requests.RequestException, ET.ParseError) as exc:
            print(f"  {type(exc).__name__}: {exc}, retrying...")
        time.sleep(5 * (attempt + 1))
    return []


def parse_entry(entry):
    raw_id = entry.findtext("a:id", default="", namespaces=NS)
    arxiv_id = re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[-1])
    title = " ".join(entry.findtext("a:title", default="", namespaces=NS).split())
    abstract = " ".join(entry.findtext("a:summary", default="", namespaces=NS).split())
    cats = []
    for c in entry.findall("a:category", NS):
        term = c.get("term", "")
        if CATEGORY_RE.match(term) and term not in cats:
            cats.append(term)
    primary = entry.find("arxiv:primary_category", NS)
    return {
        "id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "categories": cats,
        "primary_category": primary.get("term") if primary is not None else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--per-category", type=int, default=2000)
    ap.add_argument("--page-size", type=int, default=500)
    ap.add_argument("--delay", type=float, default=3.1, help="seconds between requests")
    ap.add_argument("--out", default=str(RAW_PATH))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            seen = {json.loads(line)["id"] for line in f if line.strip()}
        print(f"Resuming: {len(seen)} papers already in {out_path}")

    session = requests.Session()
    session.headers["User-Agent"] = "arxiv-paper-tagger/1.0 (educational project)"

    with out_path.open("a", encoding="utf-8") as f:
        for cat in args.categories:
            added = 0
            for start in range(0, args.per_category, args.page_size):
                size = min(args.page_size, args.per_category - start)
                entries = fetch_page(session, cat, start, size)
                time.sleep(args.delay)
                if not entries:
                    break
                for entry in entries:
                    rec = parse_entry(entry)
                    if not rec["abstract"] or not rec["categories"] or rec["id"] in seen:
                        continue
                    seen.add(rec["id"])
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    added += 1
                f.flush()
            print(f"{cat:<16} +{added:>5} new papers  (total {len(seen)})")

    print(f"Done. {len(seen)} papers saved to {out_path}")


if __name__ == "__main__":
    main()
