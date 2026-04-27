"""
Debug version: try multiple GDELT query approaches to find one that works.
"""

from gdeltdoc import GdeltDoc, Filters


def try_query(name, **kwargs):
    """Run a query and report success or failure."""
    print(f"\n--- Attempt: {name} ---")
    try:
        filters = Filters(**kwargs)
        gd = GdeltDoc()
        articles = gd.article_search(filters)
        print(f"  SUCCESS: {len(articles)} articles")
        if len(articles) > 0:
            print(f"  Columns: {list(articles.columns)}")
            print(f"  First title: {articles.iloc[0].get('title', 'no title')}")
            print(f"  First domain: {articles.iloc[0].get('domain', 'no domain')}")
        return articles
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def main():
    # Attempt 1: English keyword + Arabic language filter
    try_query(
        "English keyword + Arabic language",
        keyword="conflict",
        start_date="2024-06-01",
        end_date="2024-06-07",
        country="SU",
        language="Arabic",
    )

    # Attempt 2: Quoted Arabic phrase
    try_query(
        "Quoted Arabic phrase",
        keyword='"صراع"',
        start_date="2024-06-01",
        end_date="2024-06-07",
        country="SU",
        language="Arabic",
    )

    # Attempt 3: Longer Arabic phrase (multiple words)
    try_query(
        "Arabic two-word phrase",
        keyword="الحرب السودان",
        start_date="2024-06-01",
        end_date="2024-06-07",
        country="SU",
        language="Arabic",
    )

    # Attempt 4: No keyword at all, just country + language filter
    try_query(
        "No keyword, just filters",
        start_date="2024-06-01",
        end_date="2024-06-07",
        country="SU",
        language="Arabic",
    )

    # Attempt 5: English keyword, no language filter (sanity check that GDELT works at all)
    try_query(
        "Sanity check: English keyword, no language filter",
        keyword="Sudan war",
        start_date="2024-06-01",
        end_date="2024-06-07",
    )


if __name__ == "__main__":
    main()
