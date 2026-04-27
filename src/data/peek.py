"""Browse articles from the GDELT corpus — flexible filtering and display."""

import argparse
import pandas as pd
from pathlib import Path

GDELT_DIR = Path("data/raw/gdelt")


def load_country(country_code):
    """Load all articles for one country into a single DataFrame."""
    country_dir = GDELT_DIR / country_code
    files = [f for f in country_dir.glob("*.parquet") if f.stat().st_size > 0]
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="SU", choices=["SU", "YM", "LY", "LE"])
    parser.add_argument("--n", type=int, default=10, help="how many articles to show")
    parser.add_argument("--domain", help="filter to articles from this domain")
    parser.add_argument("--contains", help="filter to titles containing this text")
    parser.add_argument(
        "--random", action="store_true", help="random sample instead of first N"
    )
    args = parser.parse_args()

    df = load_country(args.country)
    print(f"Loaded {len(df):,} articles for {args.country}")

    if args.domain:
        df = df[df["domain"].str.contains(args.domain, case=False, na=False)]
        print(f"  After domain filter '{args.domain}': {len(df):,}")

    if args.contains:
        df = df[df["title"].str.contains(args.contains, case=False, na=False)]
        print(f"  After title filter '{args.contains}': {len(df):,}")

    if len(df) == 0:
        print("No articles match.")
        return

    sample = df.sample(args.n, random_state=42) if args.random else df.head(args.n)

    for i, (_, row) in enumerate(sample.iterrows()):
        print(f"\n[{i + 1}] {row['seendate']}  |  {row['domain']}")
        print(f"    {row['title']}")
        print(f"    {row['url']}")


if __name__ == "__main__":
    main()
