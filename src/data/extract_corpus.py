"""
Extract structured features and editorial-category labels from the Arabic news corpus.

Input:  Excel file with columns [filename, url, date, file_path, general_context]
Output: Parquet file with extracted features + editorial_category label
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

# ---- Editorial category mapping ----
# Maps each domain to one of five editorial categories.
# Domains are matched after stripping 'www.' and applying subdomain merges.

EDITORIAL_CATEGORY = {
    # State-aligned
    "gate.ahram.org.eg": "state_aligned",
    "al-vefagh.net": "state_aligned",
    "aps.dz": "state_aligned",
    "arabic.rt.com": "state_aligned",
    # Mainstream pan-Arab
    "aljazeera.net": "mainstream_pan_arab",
    "alarabiya.net": "mainstream_pan_arab",
    "al-akhbar.com": "mainstream_pan_arab",
    "bbc.com": "mainstream_pan_arab",
    "france24.com": "mainstream_pan_arab",
    "trtarabi.com": "mainstream_pan_arab",
    "majalla.com": "mainstream_pan_arab",
    # Independent / critical
    "daraj.media": "independent_critical",
    "raseef22.net": "independent_critical",
    "megaphone.news": "independent_critical",
    # Social issues
    "sharikawalaken.media": "social_issues",
    "kohljournal.press": "social_issues",
    "jeem.me": "social_issues",
    "takatoat.org": "social_issues",
    "tadwein.org": "social_issues",
    "asfariinstitute.org": "social_issues",
    "feminism-mena.fes.de": "social_issues",
    "7ayez.com": "social_issues",
    "alpheratzmag.com": "social_issues",
    "alsifr.org": "social_issues",
    "arab-reform.net": "social_issues",
    "shechecks.net": "social_issues",
    # National / local
    "alwasat.ly": "national_local",
    "hespress.com": "national_local",
    "alyaoum24.com": "national_local",
    "ar.yabiladi.com": "national_local",
}

# Subdomain merges: map full domain -> canonical domain
SUBDOMAIN_MERGES = {
    "doc.aljazeera.net": "aljazeera.net",
}

# ---- Patterns ----
ARABIC_CHAR_RANGE = re.compile(r"[\u0600-\u06FF]")
WHITESPACE_RUN = re.compile(r"\s+")


def normalize_domain(raw_domain: str) -> str:
    """Lowercase, strip www., and apply known subdomain merges."""
    if not raw_domain:
        return ""
    d = raw_domain.lower().removeprefix("www.")
    return SUBDOMAIN_MERGES.get(d, d)


def extract_url_features(url: str) -> dict:
    """Pull domain and decoded path out of a URL."""
    if not isinstance(url, str) or not url:
        return {"domain": None, "url_path": None}

    parsed = urlparse(url)
    domain = normalize_domain(parsed.netloc)
    path = unquote(parsed.path).strip("/")

    return {
        "domain": domain,
        "url_path": path,
    }


def assign_category(domain: str | None) -> str:
    """Map domain to its editorial category, or 'unknown'."""
    if not domain:
        return "unknown"
    return EDITORIAL_CATEGORY.get(domain, "unknown")


def clean_body(text: str) -> str:
    """Normalize whitespace and trim."""
    if not isinstance(text, str):
        return ""
    return WHITESPACE_RUN.sub(" ", text).strip()


def extract_title(body: str, max_title_len: int = 200) -> str | None:
    """First non-empty line if short enough, else first sentence-ish chunk."""
    if not isinstance(body, str) or not body.strip():
        return None
    first_line = body.split("\n", 1)[0].strip()
    if first_line and len(first_line) <= max_title_len:
        return first_line
    match = re.search(r"^(.{20,200}?)[\.\?\؟!]", body)
    return match.group(1).strip() if match else body[:max_title_len].strip()


def arabic_char_ratio(text: str) -> float:
    """Fraction of characters in the Arabic Unicode block."""
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    return len(ARABIC_CHAR_RANGE.findall(text)) / len(text)


def article_id(url: str) -> str:
    """Stable 16-char hash of URL for use as primary key."""
    if not isinstance(url, str):
        url = ""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all extractors. Returns a new DataFrame."""
    out = pd.DataFrame()

    # Identifiers
    out["article_id"] = df["url"].apply(article_id)
    out["url"] = df["url"]

    # URL-derived features
    url_features = df["url"].apply(extract_url_features).apply(pd.Series)
    out = pd.concat([out, url_features], axis=1)

    # Editorial category
    out["editorial_category"] = out["domain"].apply(assign_category)

    # Date features
    out["date"] = pd.to_datetime(df["date"], errors="coerce")
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["weekday"] = out["date"].dt.day_name()

    # Body
    out["body"] = df["general_context"].apply(clean_body)
    out["title"] = out["body"].apply(extract_title)
    out["body_length"] = out["body"].str.len()
    out["body_word_count"] = out["body"].apply(
        lambda s: len(s.split()) if isinstance(s, str) else 0
    )

    # Quality signal
    out["arabic_char_ratio"] = out["body"].apply(arabic_char_ratio)

    return out


def report(out: pd.DataFrame) -> None:
    """Print a concise quality report."""
    print(f"\n{'=' * 60}")
    print(f"Extracted {len(out):,} rows.")
    print(f"{'=' * 60}")

    print("\nEditorial category distribution:")
    cat_counts = out["editorial_category"].value_counts()
    for cat, n in cat_counts.items():
        pct = 100 * n / len(out)
        print(f"  {cat:<25} {n:>7,}  ({pct:5.1f}%)")

    print("\nUnknown-category articles (not in our taxonomy):")
    unknown = out[out["editorial_category"] == "unknown"]
    if len(unknown) > 0:
        print(unknown["domain"].value_counts().head(10).to_string())
    else:
        print("  (none)")

    print(f"\nDate range: {out['date'].min()} -> {out['date'].max()}")
    print(f"Missing dates: {out['date'].isna().sum()}")

    print("\nBody length stats (characters):")
    print(out["body_length"].describe().round(0).to_string())

    print("\nArabic character ratio stats:")
    print(out["arabic_char_ratio"].describe().round(3).to_string())

    suspect = out[out["arabic_char_ratio"] < 0.5]
    print(f"\nRows with <50% Arabic characters: {len(suspect):,}")

    print(f"\nDuplicate URLs:        {out['url'].duplicated().sum():,}")
    print(f"Duplicate article_ids: {out['article_id'].duplicated().sum():,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input .xlsx file")
    parser.add_argument(
        "--output",
        default="data/processed/corpus.parquet",
        help="Path for output .parquet file",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {in_path} ...")
    df = pd.read_excel(in_path)
    print(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

    print("\nExtracting features ...")
    out = extract_features(df)

    report(out)

    print(f"\nSaving to {out_path} ...")
    out.to_parquet(out_path, index=False)
    print(f"Done. File size: {out_path.stat().st_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
