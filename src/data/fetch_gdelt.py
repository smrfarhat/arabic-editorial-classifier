"""
Fetch Arabic-language news articles from GDELT for our four target countries.

Behavior:
- One week, one country at a time (chunks the full date range)
- Retries each request up to 5 times with 5s delay
- Hard-terminates on any week that exhausts all retries
- Resumable: re-running picks up exactly where it left off
- Logs to both stdout and logs/fetch_gdelt.log in real time
"""

import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from gdeltdoc import Filters, GdeltDoc

# Configuration
COUNTRIES = {
    "SU": "Sudan",
    "YM": "Yemen",
    "LY": "Libya",
    "LE": "Lebanon",
}
START_DATE = date(2023, 1, 1)
END_DATE = date(2025, 12, 31)
WEEK_STEP = timedelta(days=7)

OUTPUT_DIR = Path("data/raw/gdelt")
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "fetch_gdelt.log"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

SLEEP_BETWEEN_QUERIES = 5  # seconds between successful queries
RETRY_DELAY = 5  # seconds between failed attempts
MAX_RETRIES = 5  # total attempts before terminating


def setup_logging():
    """Configure logging to write to both console and log file in real time."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("gdelt_fetcher")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if rerun in same process

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — append mode so resumed runs accumulate the full history
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Force unbuffered output so log file updates in real time
    for handler in logger.handlers:
        handler.flush = lambda h=handler: (
            h.stream.flush() if hasattr(h, "stream") else None
        )

    return logger


def date_range_weekly(start, end):
    """Yield (week_start, week_end) tuples covering the full range."""
    current = start
    while current < end:
        week_end = min(current + WEEK_STEP - timedelta(days=1), end)
        yield current, week_end
        current += WEEK_STEP


def save_progress(country_code, week_start, status):
    """Save current progress so resume knows exactly where to pick up."""
    progress = {
        "last_country": country_code,
        "last_week": week_start.isoformat(),
        "last_status": status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def fetch_with_retry(gd, country_code, week_start, week_end, logger):
    """
    Fetch one week of articles. Retry up to MAX_RETRIES times on failure.
    Returns articles DataFrame on success, None if all retries failed.
    """
    filters = Filters(
        start_date=week_start.isoformat(),
        end_date=week_end.isoformat(),
        country=country_code,
        language="Arabic",
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            articles = gd.article_search(filters)
            if attempt > 1:
                logger.info(f"    Recovered on attempt {attempt}")
            return articles
        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.warning(
                    f"    Attempt {attempt}/{MAX_RETRIES} failed: {e}. "
                    f"Retrying in {RETRY_DELAY}s..."
                )
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"    All {MAX_RETRIES} attempts failed. Last error: {e}")

    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("Starting GDELT fetch run")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info(f"Countries: {list(COUNTRIES.values())}")
    logger.info("=" * 70)

    gd = GdeltDoc()
    total_articles_this_run = 0
    weeks_processed_this_run = 0
    weeks_skipped_resume = 0

    for country_code, country_name in COUNTRIES.items():
        logger.info("")
        logger.info(f"=== {country_name} ({country_code}) ===")
        country_dir = OUTPUT_DIR / country_code
        country_dir.mkdir(exist_ok=True)

        for week_start, week_end in date_range_weekly(START_DATE, END_DATE):
            output_file = country_dir / f"{week_start.isoformat()}.parquet"

            # Resume: skip weeks already fetched (file exists, even if empty)
            if output_file.exists():
                weeks_skipped_resume += 1
                continue

            articles = fetch_with_retry(gd, country_code, week_start, week_end, logger)

            if articles is None:
                # Hard termination — do NOT create marker, log, save progress, exit
                save_progress(country_code, week_start, "FAILED")
                logger.error("=" * 70)
                logger.error(
                    f"TERMINATING: {country_code} week {week_start} failed after "
                    f"{MAX_RETRIES} retries."
                )
                logger.error(
                    "Re-run this script to resume. It will retry this exact week first."
                )
                logger.error("=" * 70)
                sys.exit(1)

            if len(articles) > 0:
                articles.to_parquet(output_file, index=False)
                total_articles_this_run += len(articles)
                logger.info(f"  {week_start} -> {week_end}: {len(articles)} articles")
            else:
                # Legitimate empty week — save marker so we don't refetch
                output_file.touch()
                logger.info(f"  {week_start} -> {week_end}: 0 articles")

            weeks_processed_this_run += 1
            save_progress(country_code, week_start, "OK")
            time.sleep(SLEEP_BETWEEN_QUERIES)

    # Made it through everything
    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPLETE.")
    logger.info(f"  Weeks processed this run: {weeks_processed_this_run}")
    logger.info(f"  Weeks skipped (already fetched): {weeks_skipped_resume}")
    logger.info(f"  Articles fetched this run: {total_articles_this_run}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
