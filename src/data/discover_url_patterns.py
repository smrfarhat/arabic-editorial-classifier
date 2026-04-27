"""
Sample URLs from each domain in the corpus and print their decoded paths.
Used to discover URL patterns so we can write outlet-specific section parsers.
"""

import argparse
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/corpus.parquet",
        help="Path to extracted corpus parquet",
    )
    parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=5,
        help="How many URLs to print per domain",
    )
    parser.add_argument(
        "--output",
        default="data/processed/url_patterns.txt",
        help="Where to save the report (also printed to console)",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    domains = df["domain"].value_counts()

    lines = []
    lines.append(f"Total rows: {len(df):,}")
    lines.append(f"Total unique domains: {len(domains)}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("Domain breakdown (most common first)")
    lines.append("=" * 80)
    lines.append(domains.to_string())
    lines.append("")

    lines.append("=" * 80)
    lines.append(f"Sample URLs per domain ({args.samples_per_domain} each)")
    lines.append("=" * 80)

    for domain in domains.index:
        domain_df = df[df["domain"] == domain]
        sample = domain_df.sample(
            min(args.samples_per_domain, len(domain_df)),
            random_state=42,
        )

        lines.append("")
        lines.append(f"--- {domain} ({len(domain_df):,} articles) ---")
        for _, row in sample.iterrows():
            url = row["url"]
            parsed = urlparse(url)
            path = unquote(parsed.path)
            lines.append(f"  {path}")

    output = "\n".join(lines)
    print(output)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    print(f"\n\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
