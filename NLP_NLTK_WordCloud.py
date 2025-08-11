# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud


# ----------------------------- Config Models ----------------------------- #

@dataclass(frozen=True)
class PreprocessConfig:
    """WHY: Centralize knobs; WHAT: language, stops, lemma POS, min len."""
    language: str = "english"
    extra_stopwords: Iterable[str] = ()
    lemmatizer_pos: str = "n"          # {'n','v','a','r'}
    min_token_len: int = 2


@dataclass(frozen=True)
class WordCloudConfig:
    """WHY: Reproducible rendering; WHAT: WC params."""
    width: int = 1500
    height: int = 1500
    background_color: str = "white"
    max_words: int = 150
    collocations: bool = False
    random_state: int = 42
    colormap: str = "PuBu_r"


# ----------------------------- NLTK Utilities ---------------------------- #

def ensure_nltk() -> None:
    """WHY: Needed resources; WHAT: check/download once."""
    required = [
        ("punkt", "tokenizers"),
        ("stopwords", "corpora"),
        ("wordnet", "corpora"),
        ("omw-1.4", "corpora"),
    ]
    for pkg, kind in required:
        try:
            nltk.data.find(f"{kind}/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


# ------------------------------ IO Utilities ----------------------------- #

def _read_text_from_url(url: str) -> str:
    """WHY: Support GitHub; WHAT: fetch UTF-8 text from a HTTP(S) URL."""
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def load_json_texts(src: Union[Path, str]) -> str:
    """
    WHY: Allow local paths *or* GitHub RAW URLs without changing callers.
    WHAT: Loads a JSON array of objects and concatenates 'text' fields.
    """
    if isinstance(src, str) and src.startswith(("http://", "https://")):
        data = json.loads(_read_text_from_url(src))
    else:
        p = Path(src)
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of paragraph objects.")
    return " ".join(str(obj.get("text", "")) for obj in data)


# ---------------------------- Text Preprocessing ------------------------- #

_DASH_TRANS = {
    ord("\u2010"): ord("-"),
    ord("\u2011"): ord("-"),
    ord("\u2012"): ord("-"),
    ord("\u2013"): ord("-"),
    ord("\u2014"): ord("-"),
    ord("\u2212"): ord("-"),
}

def normalize_text(text: str) -> str:
    """WHY: Stable tokens; WHAT: lower, strip accents, normalize dashes."""
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    s = unicodedata.normalize("NFKD", text).lower()
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.translate(_DASH_TRANS)

def tokenize_expand(text: str, language: str) -> List[str]:
    """WHY: Catch 'ml' in 'ml-based'/'ml/ai'; WHAT: tokenize + split -/."""
    tokens = word_tokenize(text, language=language)
    tokens = [t for t in tokens if any(c.isalpha() for c in t)]
    parts: List[str] = []
    split_re = re.compile(r"[-/]")
    for tok in tokens:
        parts.extend(split_re.split(tok))
    return [p for p in parts if p]

def remove_stop_words(
    tokens: Sequence[str], language: str, extra_stopwords: Iterable[str]
) -> List[str]:
    """WHY: Drop closed-class/domain filler; WHAT: NLTK stops ∪ extra."""
    base = set(stopwords.words(language))
    extra = {w.lower() for w in extra_stopwords}
    stop_set = base | extra
    return [t for t in tokens if t.lower() not in stop_set]

def lemmatize_tokens(tokens: Sequence[str], pos: str) -> List[str]:
    """WHY: Canonical counts; WHAT: WordNet lemmatization by POS."""
    if pos not in {"n", "v", "a", "r"}:
        raise ValueError("pos must be one of {'n','v','a','r'}")
    lem = WordNetLemmatizer()
    return [lem.lemmatize(t.lower(), pos=pos) for t in tokens]

def ml_counts(raw_norm_text: str, tokens: Sequence[str]) -> Tuple[int, int, int]:
    """WHY: Verify ML survival; WHAT: (raw-standalone, raw-hyph, tokens=='ml')."""
    stand = len(re.findall(r"\bml\b", raw_norm_text, flags=re.IGNORECASE))
    hyph = len(re.findall(r"\bml\-", raw_norm_text, flags=re.IGNORECASE))
    in_tokens = sum(1 for t in tokens if t == "ml")
    return stand, hyph, in_tokens

def build_freqs(tokens: Sequence[str], min_len: int) -> Dict[str, int]:
    """WHY: WordCloud input; WHAT: filter by length and count."""
    if min_len < 1:
        raise ValueError("min_len must be >= 1")
    filtered = [t for t in tokens if len(t) >= min_len]
    return dict(Counter(filtered))


# ------------------------------- N-grams --------------------------------- #

def ngrams(tokens: Sequence[str], n: int) -> Counter:
    """WHY: Collocations; WHAT: Counter of n-gram tuples."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

def plot_top_ngrams(counts: Counter, top_k: int, title: str,
                    out_path: Path) -> Path:
    """WHY: Quick compare; WHAT: bar chart of top n-grams."""
    top = counts.most_common(top_k)
    labels = [" ".join(k) for k, _ in top]
    values = [v for _, v in top]
    plt.figure(figsize=(10, 7))
    plt.barh(range(len(values)), values)
    plt.yticks(range(len(labels)), labels)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


# ------------------------------ Visualization ---------------------------- #

def render_wordcloud(freqs: Dict[str, int], wc_cfg: WordCloudConfig,
                     out_path: Path) -> Path:
    """WHY: Persist visual; WHAT: render/sleep-free save (Agg backend)."""
    wc = WordCloud(
        width=wc_cfg.width,
        height=wc_cfg.height,
        background_color=wc_cfg.background_color,
        max_words=wc_cfg.max_words,
        collocations=wc_cfg.collocations,
        normalize_plurals=True,
        prefer_horizontal=1.0,
        random_state=wc_cfg.random_state,
        colormap=wc_cfg.colormap,
    ).generate_from_frequencies(freqs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wc.to_file(str(out_path))
    return out_path


def compose_side_by_side(image_paths: Sequence[Path], titles: Sequence[str],
                         sup_title: str, footnote: str,
                         out_path: Path) -> Path:
    """
    WHY: Report-friendly compare; WHAT: stitch images + footnote.
    Fixes layout: constrained_layout + aspect='auto'.
    """
    if len(image_paths) != len(titles):
        raise ValueError("image_paths and titles must have same length.")

    fig, axes = plt.subplots(
        1, len(image_paths), figsize=(16, 9), dpi=300,
        # constrained_layout=True
    )
    if len(image_paths) == 1:
        axes = [axes]

    for ax, img_path, title in zip(axes, image_paths, titles):
        ax.imshow(plt.imread(img_path), aspect="auto")
        ax.set_title(title, fontsize=14, pad=10)
        ax.axis("off")

    fig.suptitle(sup_title, fontsize=22, y=0.995)
    
    fig.tight_layout(rect=[0.01, 0.08, 0.99, 0.96])
    
    if footnote:
        fig.text(0.01, 0.02, footnote, ha="left", va="bottom", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# --------------------------------- Main ---------------------------------- #

def process_file(src: Union[Path, str], prep: PreprocessConfig,
                 wc_cfg: WordCloudConfig, out_dir: Path,
                 make_ngram_plots: bool = True) -> Tuple[Path, Dict[str, int]]:
    """WHY: Reuse & test; WHAT: read->prep->freqs->WC (+ optional n-grams)."""
    t0 = time.perf_counter()
    raw = load_json_texts(src)
    norm = normalize_text(raw)

    tokens = tokenize_expand(norm, language=prep.language)
    tokens = remove_stop_words(tokens, prep.language, prep.extra_stopwords)
    tokens = lemmatize_tokens(tokens, pos=prep.lemmatizer_pos)

    stand, hyph, ml_tok = ml_counts(norm, tokens)
    print(f"[{src}] ML raw_standalone={stand}, raw_hyphenated={hyph}, "
          f"tokens=='ml'={ml_tok}")

    freqs = build_freqs(tokens, min_len=prep.min_token_len)
    stem = Path(str(src)).stem
    wc_out = out_dir / f"{stem}_wordcloudv2.png"
    render_wordcloud(freqs, wc_cfg, wc_out)

    if make_ngram_plots:
        bi = ngrams(tokens, 2)
        tri = ngrams(tokens, 3)
        plot_top_ngrams(bi, 20, f"Top 20 Bigrams — {stem}",
                        out_dir / f"{stem}_bigrams_top20v2.png")
        plot_top_ngrams(tri, 20, f"Top 20 Trigrams — {stem}",
                        out_dir / f"{stem}_trigrams_top20v2.png")

    elapsed = time.perf_counter() - t0
    stats = {
        "tokens": len(tokens),
        "vocab": len(freqs),
        "ml_tokens": ml_tok,
        "seconds": round(elapsed, 2),
    }
    print(f"[{stem}] saved -> {wc_out.name} | {stats}")
    return wc_out, stats


def main() -> None:
    """WHY: No globals; WHAT: run both files, compose comparison."""
    ensure_nltk()

    cwd = Path(os.getcwd())
    out_dir = cwd / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- GitHub RAW (edit branch/filenames as needed) ----
    json_sources: List[Union[str, Path]] = [
        ("https://raw.githubusercontent.com/Fercv87/gpt5_testing/main/"
         "egim_202507_paragraphsv2.json"),
        ("https://raw.githubusercontent.com/Fercv87/gpt5_testing/main/"
         "egim_202402_paragraphsv2.json"),
    ]

    prep = PreprocessConfig(
        language="english",
        extra_stopwords={"accordance"},
        lemmatizer_pos="n",
        min_token_len=2,
    )
    wc_cfg = WordCloudConfig(
        width=1500, height=1500, background_color="white", max_words=150,
        collocations=False, random_state=42, 
        # colormap="PuBu_r",
        colormap="gist_earth",
    )

    images: List[Path] = []
    titles: List[str] = []

    for src in json_sources:
        img, stats = process_file(
            src=src, prep=prep, wc_cfg=wc_cfg, out_dir=out_dir,
            make_ngram_plots=True,
        )
        images.append(img)
        label = "Jul. 2025 ECB Guide" if "202507" in str(src) else "Feb. 2024 ECB Guide"
        titles.append(label)  # ← no token/vocab/ml stats in the title

    foot = (
        "Pre-processing: lowercasing, accent stripping, dash normalization; "
        f"NLTK tokenization; stopword removal ({prep.language}) + extras "
        f"{set(prep.extra_stopwords)}; WordNet pos='{prep.lemmatizer_pos}'; "
        f"min_len≥{prep.min_token_len}; "
    )

    combined = out_dir / "egim_wordclouds_combinedv2.png"
    compose_side_by_side(
        image_paths=images,
        titles=titles,
        sup_title=("Wordcloud comparison of ECB guides to internal models "
                   "(07/25 vs 02/24)"),
        footnote=foot,
        out_path=combined,
    )
    print(f"Saved combined figure -> {combined}")


if __name__ == "__main__":
    main()



for ax, (img_path, title) in zip(axes, rendered):
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.set_title(title, fontsize=14, pad=10)
    ax.axis("off")

fig.suptitle("Wordcloud comparison of ECB guides to internal models (07/25 vs 02/24)", fontsize=22, y=0.995)
