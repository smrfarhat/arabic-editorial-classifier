"""Whole-corpus statistics across all four countries."""

import pandas as pd
from pathlib import Path

GDELT_DIR = Path("data/raw/gdelt")
COUNTRIES = ["SU", "YM", "LY", "LE"]

all_dfs = []
for code in COUNTRIES:
    files = [f for f in (GDELT_DIR / code).glob("*.parquet") if f.stat().st_size > 0]
    for f in files:
        df = pd.read_parquet(f)
        df["country_code"] = code
        all_dfs.append(df)

corpus = pd.concat(all_dfs, ignore_index=True)
print(f"Total articles: {len(corpus):,}")
print(f"Unique URLs:    {corpus['url'].nunique():,}")
print(f"Cross-country duplicates: {len(corpus) - corpus['url'].nunique():,}")

print("\nArticles per country:")
print(corpus["country_code"].value_counts().to_string())

print("\nTop 20 domains overall:")
print(corpus["domain"].value_counts().head(20).to_string())

print("\nLanguage tags returned by GDELT:")
print(corpus["language"].value_counts().to_string())

# Date range
corpus["date"] = pd.to_datetime(corpus["seendate"].str[:8], format="%Y%m%d")
print(f"\nDate range: {corpus['date'].min().date()} to {corpus['date'].max().date()}")
print(f"Median date: {corpus['date'].median().date()}")

# Articles per month
print("\nArticles per month (last 12):")
monthly = corpus.groupby(corpus["date"].dt.to_period("M")).size()
print(monthly.tail(12).to_string())
