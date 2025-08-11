"""
ECB Guide → JSON extractor

- Input: a PDF (e.g., ssm.supervisory_guide202507.en.pdf)
- Output: JSON list of objects:
    {
      "title": "...",                 # big section heading
      "subtitle": "... or 'N/A'",     # numbered subsection heading (e.g., "1 Guidelines ...")
      "paragraph_number": "1",        # from "1. ..." at the start of a paragraph
      "page": 5,                      # printed page number (taken from the header)
      "text": "..."                   # paragraph text (tables/footnotes excluded)
    }

Notes
- Only paragraphs that begin with "<number>." are captured; tables are thereby excluded.
- Footers/headers are stripped; tiny-font footnotes are skipped.
- Subtitles are set from 12.5–18pt numbered headings (e.g., "1 Documentation ...").
- Printed page numbers are parsed from the running header.

Install:
    pip install pymupdf

Run:
    python extract_egim_json.py \
        --pdf /path/to/ssm.supervisory_guide202507.en.pdf \
        --out /path/to/egim_202507_paragraphs.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any

import fitz  # PyMuPDF


# ---------- Configurable heuristics ----------
# Paragraph must start with "N. "
PARA_START_RE = re.compile(r"^\s*(\d+)\.\s+")

# Numbered subtitle like "1 Guidelines at consolidated ..."
NUM_SUBTITLE_RE = re.compile(r"^\s*(\d+)\s+[A-Za-z].*")

# Running header that carries the printed page number
HEADER_NUM_RE = re.compile(r"ECB guide to internal models\s+–\s+.*?\s+(\d+)\s*$")

# Strip any leftover header/footer text fragments from blocks
HEADER_STRIP_RE = re.compile(r"ECB guide to internal models\s+–\s+.*?\s+\d+\s*")

# Font-size thresholds (empirically inspected for this PDF)
TITLE_MIN_PT = 18.0       # e.g., "Foreword", "Overarching principles for internal models"
SUBTITLE_MIN_PT = 12.5    # e.g., "1 Guidelines at consolidated ...", "2 Documentation ..."
FOOTNOTE_MAX_PT = 8.5     # footnotes are often ~7–8pt in this document

# Optional quick table heading filter (best-effort)
TABLE_HEADING_PREFIXES = ("Table ", "Relevant regulatory references")


# ---------- Helpers ----------

def normalize_text(s: str) -> str:
    """Collapse whitespace, remove soft hyphens and header fragments."""
    s = s.replace("\u00ad", "")      # soft hyphen
    s = s.replace("-\n", "")         # hyphenated line break
    s = s.replace("\n", " ")
    s = HEADER_STRIP_RE.sub(" ", s)  # remove header/footer remnants if any slipped in
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_printed_page_number(page: fitz.Page) -> int:
    """Extract the printed page number from the running header; fallback to PDF index."""
    d = page.get_text("dict", sort=True)
    for block in d.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        raw = "".join(
            span.get("text", "")
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        )
        m = HEADER_NUM_RE.search(raw)
        if m:
            return int(m.group(1))
    # Fallback if not found (shouldn’t happen for these pages)
    return page.number + 1


def collect_text_blocks(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Return page text blocks with:
        { "idx": int, "text": str, "max_size": float }
    """
    d = page.get_text("dict", sort=True)
    blocks: List[Dict[str, Any]] = []
    for bi, block in enumerate(d.get("blocks", [])):
        if block.get("type", 0) != 0:
            continue
        raw = "".join(
            span.get("text", "")
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        )
        txt = normalize_text(raw)
        if not txt:
            continue
        sizes = [
            span.get("size", 0.0)
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ]
        max_size = max(sizes) if sizes else 0.0
        blocks.append({"idx": bi, "text": txt, "max_size": max_size})
    return blocks


def looks_like_table_heading(text: str) -> bool:
    return any(text.startswith(pfx) for pfx in TABLE_HEADING_PREFIXES)


# ---------- Core extraction ----------

def extract_json(
    pdf_path: Path,
    out_path: Path,
    start_printed_page: int = 5,
    end_printed_page: int = 357,
) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)

    results: List[Dict[str, Any]] = []
    title: str | None = None
    subtitle: str | None = None
    cur_para: Dict[str, Any] | None = None
    buf: List[str] = []

    def flush():
        nonlocal cur_para, buf
        if cur_para is not None:
            text = normalize_text(" ".join(buf))
            if text:
                cur_para["text"] = text
                results.append(cur_para)
        cur_para = None
        buf = []

    # Map from printed page → PDF index to iterate only desired range
    printed_to_index: Dict[int, int] = {}
    for i in range(doc.page_count):
        pp = get_printed_page_number(doc.load_page(i))
        printed_to_index[pp] = i

    for printed_p in range(start_printed_page, end_printed_page + 1):
        if printed_p not in printed_to_index:
            # Some PDFs repeat headers in annexes differently; skip if not present
            continue
        pidx = printed_to_index[printed_p]
        page = doc.load_page(pidx)
        blocks = collect_text_blocks(page)

        # Pass 1: detect big titles (20pt) for the page
        for b in blocks:
            if b["max_size"] >= TITLE_MIN_PT and len(b["text"]) < 120 and b["text"].lower() != "contents":
                title = b["text"]
                subtitle = None

        # Pass 2: detect numbered subtitles (12.5–18pt)
        for b in blocks:
            if SUBTITLE_MIN_PT <= b["max_size"] < TITLE_MIN_PT and len(b["text"]) < 220:
                if NUM_SUBTITLE_RE.match(b["text"]):
                    flush()
                    subtitle = b["text"]

        # Pass 3: numbered paragraphs and body continuation
        for b in blocks:
            txt = b["text"]
            # Skip tiny-footnotes
            if b["max_size"] < FOOTNOTE_MAX_PT:
                continue
            # Skip obvious table headings
            if looks_like_table_heading(txt):
                continue
            # Skip subtitles themselves in this pass
            if SUBTITLE_MIN_PT <= b["max_size"] < TITLE_MIN_PT and NUM_SUBTITLE_RE.match(txt):
                continue

            m = PARA_START_RE.match(txt)
            if m:
                flush()
                para_no = m.group(1)
                rest = txt[m.end():].strip()
                cur_para = {
                    "title": title or "N/A",
                    "subtitle": subtitle or "N/A",
                    "paragraph_number": para_no,
                    "page": printed_p,
                    "text": "",
                }
                if rest:
                    buf.append(rest)
            else:
                # Append continuation text ONLY if we’re within a paragraph
                if cur_para is not None:
                    buf.append(txt)

    flush()
    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Extract numbered paragraphs from ECB Guide PDF into JSON.")
    ap.add_argument("--pdf", required=True, type=Path, help="Path to the PDF file.")
    ap.add_argument("--out", required=True, type=Path, help="Path to write the JSON.")
    ap.add_argument("--start", type=int, default=4, help="Printed page to start from (default: 5).")
    ap.add_argument("--end", type=int, default=272, help="Printed page to end at (default: 357).")
    args = ap.parse_args()

    _ = extract_json(args.pdf, args.out, start_printed_page=args.start, end_printed_page=args.end)
    print(f"Wrote JSON to: {args.out}")

if __name__ == "__main__":
    main()
