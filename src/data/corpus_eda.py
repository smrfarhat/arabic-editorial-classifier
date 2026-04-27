"""
Exploratory analysis of the labeled Arabic editorial corpus.
Run after extract_corpus.py to understand class balance, body lengths per
category, date coverage, and quality signals before model training.
"""

from pathlib import Path

import pandas as pd

CORPUS_PATH = Path("data/processed/corpus.parquet")


def main():
    df = pd.read_parquet(CORPUS_PATH)
    print(f"Loaded {len(df):,} articles\n")

    # --- Category breakdown ---
    print("=" * 60)
    print("Editorial category counts")
    print("=" * 60)
    cat_counts = df["editorial_category"].value_counts()
    print(cat_counts.to_string())
    print(
        f"\nClass imbalance ratio (max/min): {cat_counts.max() / cat_counts.min():.1f}x"
    )

    # --- Domains per category ---
    print("\n" + "=" * 60)
    print("Domains per category")
    print("=" * 60)
    for cat in cat_counts.index:
        sub = df[df["editorial_category"] == cat]
        domains = sub["domain"].value_counts()
        print(f"\n{cat} ({len(sub):,} articles, {len(domains)} domains):")
        for domain, n in domains.items():
            print(f"  {domain:<30} {n:>6,}")

    # --- Body length per category ---
    print("\n" + "=" * 60)
    print("Body length (characters) per category")
    print("=" * 60)
    print(
        df.groupby("editorial_category")["body_length"].describe().round(0).to_string()
    )

    # --- Date distribution ---
    print("\n" + "=" * 60)
    print("Articles per month")
    print("=" * 60)
    monthly = df.groupby(df["date"].dt.to_period("M")).size()
    print(monthly.to_string())

    # --- Quality flags ---
    print("\n" + "=" * 60)
    print("Quality flags")
    print("=" * 60)
    too_short = (df["body_length"] < 100).sum()
    too_long = (df["body_length"] > 50000).sum()
    low_arabic = (df["arabic_char_ratio"] < 0.5).sum()
    no_date = df["date"].isna().sum()

    print(f"  Body < 100 chars:           {too_short:,}")
    print(f"  Body > 50,000 chars:        {too_long:,}")
    print(f"  Arabic char ratio < 0.5:    {low_arabic:,}")
    print(f"  Missing date:               {no_date:,}")

    # --- Recommended training-ready subset ---
    print("\n" + "=" * 60)
    print("After quality filtering")
    print("=" * 60)
    clean = df[
        (df["body_length"] >= 100)
        & (df["body_length"] <= 50000)
        & (df["arabic_char_ratio"] >= 0.5)
        & (df["editorial_category"] != "unknown")
        & (df["date"].notna())
    ]
    print(f"  Clean training set: {len(clean):,} articles")
    print("\n  Per category:")
    print(clean["editorial_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
