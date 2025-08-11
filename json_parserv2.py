"""
ECB Guide → JSON extractor (with section + subsection)

Output objects:
{
  "title": "Overarching principles for internal models",
  "section": "1 Overarching principles for internal models",   # or "A Credit risk"
  "subsection": "1.2 Guidelines at consolidated and subsidiary levels",  # or "A.1 …"
  "paragraph_number": "1",
  "page": 6,    # printed page (from header)
  "text": "…"
}

Install:
    pip install pymupdf

Run:
    python json_parser.py --pdf <input.pdf> --out <output.json> --start 5 --end 357
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF


# ---------- Heuristics / Patterns ----------
TITLE_MIN_PT = 18.0          # big chapter title size (e.g., 20pt)
HEADING_MIN_PT = 12.5        # section/subsection headings (e.g., 14pt)
FOOTNOTE_MAX_PT = 8.5        # tiny text → footnotes

# Numbered paragraph like "1. ..."
PARA_START_RE = re.compile(r"^\s*(\d+)\.\s+")

# Headings
SEC_NUM_RE         = re.compile(r"^\s*(\d+)\s+(.+)$")            # "1 Overarching principles …" (rarely printed as such)
SEC_ALPHA_RE       = re.compile(r"^\s*([A-Z])\s+(.+)$")          # "A General topics for credit risk"
SUBSEC_NUMDOT_RE   = re.compile(r"^\s*(\d+)\.(\d+)\s+(.+)$")     # "1.2 Guidelines …"
SUBSEC_ALPHADOT_RE = re.compile(r"^\s*([A-Z])\.(\d+)\s+(.+)$")   # "A.1 Scope …"

# Header/footer handling
HEADER_NUM_RE   = re.compile(r"ECB guide to internal models\s+–\s+.*?\s+(\d+)\s*$")
HEADER_STRIP_RE = re.compile(r"ECB guide to internal models\s+–\s+.*?\s+\d+\s*")

TABLE_HEADING_PREFIXES = ("Table ", "Relevant regulatory references")


# ---------- Helpers ----------
def normalize_text(s: str) -> str:
    s = s.replace("\u00ad", "")      # soft hyphen
    s = s.replace("-\n", "")         # hyphenated line break
    s = s.replace("\n", " ")
    s = HEADER_STRIP_RE.sub(" ", s)  # remove header/footer remnants
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_printed_page_number(page: fitz.Page) -> int:
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
    return page.number + 1


def collect_blocks(page: fitz.Page) -> List[Dict[str, Any]]:
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

    # Current context
    current_title: str | None = None            # big chapter title (20pt)
    current_section: str = "N/A"               # "1 Overarching principles …" or "A Credit risk"
    current_subsection: str = "N/A"            # "1.2 Guidelines …" or "A.1 …"
    current_chapter_id: str | None = None      # "1" or "A" (derived from headings)

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

    # Map printed page → pdf index to iterate only desired range
    printed_to_index = {}
    for i in range(doc.page_count):
        printed_to_index[get_printed_page_number(doc.load_page(i))] = i

    for printed_p in range(start_printed_page, end_printed_page + 1):
        if printed_p not in printed_to_index:
            continue
        pidx = printed_to_index[printed_p]
        page = doc.load_page(pidx)
        blocks = collect_blocks(page)

        # Pass A: detect big chapter titles (>= TITLE_MIN_PT)
        for b in blocks:
            if b["max_size"] >= TITLE_MIN_PT and len(b["text"]) < 200 and b["text"].lower() != "contents":
                current_title = b["text"]
                # Reset section/subsection until we see numbered/lettered headings
                current_section = "N/A"
                current_subsection = "N/A"
                current_chapter_id = None

        # Pass B: detect section/subsection headings in reading order
        for b in blocks:
            if b["max_size"] < HEADING_MIN_PT:
                continue

            txt = b["text"]

            # Subsections first (1.2 ..., A.1 ...)
            m = SUBSEC_NUMDOT_RE.match(txt)
            if m:
                current_chapter_id = m.group(1)  # "1"
                # Section label is "<chapter> <title>" once we know the chapter id
                if current_title:
                    current_section = f"{current_chapter_id} {current_title}"
                else:
                    current_section = current_chapter_id or "N/A"
                current_subsection = txt
                continue

            m = SUBSEC_ALPHADOT_RE.match(txt)
            if m:
                current_chapter_id = m.group(1)  # "A"
                current_section = f"{current_chapter_id} {current_title}" if current_title else current_chapter_id
                current_subsection = txt
                continue

            # Then plain sections (1 ..., A ...)
            m = SEC_NUM_RE.match(txt)
            if m:
                current_chapter_id = m.group(1)  # "1"
                current_section = f"{current_chapter_id} {current_title}" if current_title else current_chapter_id
                # If the document uses only "1 <heading>" (no dot level), treat it as subsection label too
                current_subsection = txt
                continue

            m = SEC_ALPHA_RE.match(txt)
            if m:
                current_chapter_id = m.group(1)  # "A"
                current_section = f"{current_chapter_id} {current_title}" if current_title else current_chapter_id
                current_subsection = txt
                continue

        # Pass C: numbered paragraphs
        for b in blocks:
            # Skip small footnotes
            if b["max_size"] < FOOTNOTE_MAX_PT:
                continue

            txt = b["text"]
            # Skip headings and table headings
            if (
                (b["max_size"] >= HEADING_MIN_PT and (
                    SUBSEC_NUMDOT_RE.match(txt) or SUBSEC_ALPHADOT_RE.match(txt) or
                    SEC_NUM_RE.match(txt) or SEC_ALPHA_RE.match(txt)
                ))
                or looks_like_table_heading(txt)
            ):
                continue

            m = PARA_START_RE.match(txt)
            if m:
                flush()
                para_no = m.group(1)
                rest = txt[m.end():].strip()
                cur_para = {
                    "title": current_title or "N/A",
                    "section": current_section or "N/A",
                    "subsection": current_subsection or "N/A",
                    "paragraph_number": para_no,
                    "page": printed_p,
                    "text": "",
                }
                if rest:
                    buf.append(rest)
            else:
                if cur_para is not None:
                    buf.append(txt)

    flush()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Extract numbered paragraphs into JSON with section/subsection.")
    ap.add_argument("--pdf", required=True, type=Path, help="Path to the PDF file.")
    ap.add_argument("--out", required=True, type=Path, help="Path to write the JSON.")
    ap.add_argument("--start", type=int, default=5, help="Printed page to start (default: 5).")
    ap.add_argument("--end", type=int, default=357, help="Printed page to end (default: 357).")
    args = ap.parse_args()

    _ = extract_json(args.pdf, args.out, start_printed_page=args.start, end_printed_page=args.end)
    print(f"Wrote JSON to: {args.out}")

if __name__ == "__main__":
    main()



# python json_parserv2.py `
#   --pdf "C:\Users\fercv\OneDrive\Desktop\MAS_Pilot\json_utils\ssm.supervisory_guides202402_internalmodels.en.pdf" `
#   --out "C:\Users\fercv\OneDrive\Desktop\MAS_Pilot\json_utils\egim_202402_paragraphs.json" `
#   --start 5 `
#   --end 999

# python json_parserv2.py `
#   --pdf "C:\Users\fercv\OneDrive\Desktop\MAS_Pilot\json_utils\ssm.supervisory_guide202507.en.pdf" `
#   --out "C:\Users\fercv\OneDrive\Desktop\MAS_Pilot\json_utils\egim_202507_paragraphsv2.json" `
#   --start 5 `
#   --end 357