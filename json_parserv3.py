"""
ECB Guide → JSON extractor (reading-order; section + subsection)

Output objects:
{
  "title": "Overarching principles for internal models",
  "section": "3 Overarching principles for internal models",   # change a line below if you want no leading '3'
  "subsection": "3 Data governance",
  "paragraph_number": "9",
  "page": 10,
  "text": "…"
}

Install:
    pip install pymupdf

Run:
    python json_parser.py --pdf <input.pdf> --out <output.json> --start 5 --end 357
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List
import fitz  # PyMuPDF

# --- Heuristics / Patterns ---
TITLE_MIN_PT   = 18.0   # big chapter titles (e.g., 20pt)
HEADING_MIN_PT = 12.5   # section/subsection headings (e.g., 14pt)
FOOTNOTE_MAX_PT = 8.5   # tiny text → footnotes

# Numbered paragraph like "1. ..."
PARA_START_RE = re.compile(r"^\s*(\d+)\.\s+")

# Headings
SUBSEC_NUMDOT_RE   = re.compile(r"^\s*(\d+)\.(\d+)\s+(.+)$")   # "1.2 Guidelines …"
SUBSEC_ALPHADOT_RE = re.compile(r"^\s*([A-Z])\.(\d+)\s+(.+)$") # "A.1 Scope …"
SEC_NUM_RE         = re.compile(r"^\s*(\d+)\s+(.+)$")          # "1 Overarching principles …"
SEC_ALPHA_RE       = re.compile(r"^\s*([A-Z])\s+(.+)$")        # "A Credit risk"

# Header/footer handling
HEADER_NUM_RE   = re.compile(r"ECB guide to internal models\s+–\s+.*?\s+(\d+)\s*$")
HEADER_STRIP_RE = re.compile(r"ECB guide to internal models\s+–\s+.*?\s+\d+\s*")

TABLE_HEADING_PREFIXES = ("Table ", "Relevant regulatory references")

def normalize_text(s: str) -> str:
    s = s.replace("\u00ad", "").replace("-\n", "").replace("\n", " ")
    s = HEADER_STRIP_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_printed_page_number(page: fitz.Page) -> int:
    d = page.get_text("dict", sort=True)
    for block in d.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        raw = "".join(span.get("text", "") for line in block.get("lines", []) for span in line.get("spans", []))
        m = HEADER_NUM_RE.search(raw)
        if m: return int(m.group(1))
    return page.number + 1

def collect_blocks(page: fitz.Page) -> List[Dict[str, Any]]:
    d = page.get_text("dict", sort=True)
    blocks: List[Dict[str, Any]] = []
    for bi, block in enumerate(d.get("blocks", [])):
        if block.get("type", 0) != 0:
            continue
        raw = "".join(span.get("text", "") for line in block.get("lines", []) for span in line.get("spans", []))
        txt = normalize_text(raw)
        if not txt: continue
        sizes = [span.get("size", 0.0) for line in block.get("lines", []) for span in line.get("spans", [])]
        max_size = max(sizes) if sizes else 0.0
        blocks.append({"idx": bi, "text": txt, "max_size": max_size})
    return blocks

def looks_like_table_heading(text: str) -> bool:
    return any(text.startswith(pfx) for pfx in TABLE_HEADING_PREFIXES)

def extract_json(pdf_path: Path, out_path: Path, start_printed_page: int = 5, end_printed_page: int = 357):
    doc = fitz.open(pdf_path)

    # printed page -> pdf index
    printed_to_index = {}
    for i in range(doc.page_count):
        printed_to_index[get_printed_page_number(doc.load_page(i))] = i

    results: List[Dict[str, Any]] = []

    # Current context (updated in reading order)
    title: str | None = None
    section: str = "N/A"
    subsection: str = "N/A"
    chapter_id: str | None = None

    cur_para: Dict[str, Any] | None = None
    buf: List[str] = []

    def flush():
        nonlocal cur_para, buf
        if cur_para is not None:
            t = normalize_text(" ".join(buf))
            if t:
                cur_para["text"] = t
                results.append(cur_para)
        cur_para = None
        buf = []

    for printed in range(start_printed_page, end_printed_page + 1):
        if printed not in printed_to_index:
            continue
        page = doc.load_page(printed_to_index[printed])
        blocks = collect_blocks(page)

        # *** Single pass in reading order ***
        for b in blocks:
            txt, sz = b["text"], b["max_size"]

            # Big chapter titles
            if sz >= TITLE_MIN_PT and len(txt) < 200 and txt.lower() != "contents":
                title = txt
                section = "N/A"
                subsection = "N/A"
                chapter_id = None
                continue

            # Section / Subsection headings
            if sz >= HEADING_MIN_PT:
                m = SUBSEC_NUMDOT_RE.match(txt)
                if m:
                    chapter_id = m.group(1)
                    # If you want section WITHOUT the leading number, replace next line with: section = title or "N/A"
                    # section = f"{chapter_id} {title}" if title else chapter_id or "N/A"
                    section = title or "N/A"
                    subsection = txt
                    continue

                m = SUBSEC_ALPHADOT_RE.match(txt)
                if m:
                    chapter_id = m.group(1)
                    # section = f"{chapter_id} {title}" if title else chapter_id or "N/A"
                    section = title or "N/A"
                    subsection = txt
                    continue

                m = SEC_NUM_RE.match(txt)
                if m and len(txt.split()) > 2:
                    chapter_id = m.group(1)
                    # section = f"{chapter_id} {title}" if title else chapter_id or "N/A"
                    section = title or "N/A"
                    subsection = txt
                    continue

                m = SEC_ALPHA_RE.match(txt)
                if m and len(txt.split()) > 2:
                    chapter_id = m.group(1)
                    # section = f"{chapter_id} {title}" if title else chapter_id or "N/A"
                    section = title or "N/A"
                    subsection = txt
                    continue

            # Skip tiny footnotes and table headings
            if sz < FOOTNOTE_MAX_PT or looks_like_table_heading(txt):
                continue

            # Numbered paragraphs
            m = PARA_START_RE.match(txt)
            if m:
                flush()
                cur_para = {
                    "title": title or "N/A",
                    "section": section or "N/A",
                    "subsection": subsection or "N/A",
                    "paragraph_number": m.group(1),
                    "page": printed,
                    "text": ""
                }
                rest = txt[m.end():].strip()
                if rest: buf.append(rest)
            else:
                if cur_para is not None:
                    buf.append(txt)

    flush()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="Extract numbered paragraphs into JSON with section/subsection (reading order).")
    ap.add_argument("--pdf", required=True, type=Path, help="Path to the PDF file.")
    ap.add_argument("--out", required=True, type=Path, help="Path to write the JSON.")
    ap.add_argument("--start", type=int, default=5, help="Printed page to start (default: 5).")
    ap.add_argument("--end", type=int, default=357, help="Printed page to end (default: 357).")
    args = ap.parse_args()

    _ = extract_json(args.pdf, args.out, start_printed_page=args.start, end_printed_page=args.end)
    print(f"Wrote JSON to: {args.out}")

if __name__ == "__main__":
    main()

# python json_parserv3.py `
#   --pdf "C:\Users\fercv\OneDrive\Desktop\MAS_Pilot\json_utils\ssm.supervisory_guides202402_internalmodels.en.pdf" `
#   --out "C:\Users\fercv\OneDrive\Desktop\MAS_Pilot\json_utils\egim_202402_paragraphsv2.json" `
#   --start 4 `
#   --end 999

# python json_parserv3.py `
#   --pdf "C:\Users\fercv\OneDrive\Desktop\MAS_Pilot\json_utils\ssm.supervisory_guide202507.en.pdf" `
#   --out "C:\Users\fercv\OneDrive\Desktop\MAS_Pilot\json_utils\egim_202507_paragraphsv2.json" `
#   --start 5 `
#   --end 357