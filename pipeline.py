"""
Main script for running the news fetching pipeline.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure console **and** file logging.

    Configuration via environment variables (all optional):
    * ``LOG_LEVEL`` – ``DEBUG`` | ``INFO`` | ``WARNING`` | ``ERROR`` | ``CRITICAL`` (default: ``INFO``)
    * ``LOG_DIR``   – directory for log files (default: ``./logs``)
    * ``LOG_FILE``  – filename   (default: ``pipeline.log`` inside *LOG_DIR*)
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # ‼ we create the directory first so FileHandler never fails on GitHub Actions
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path(os.getenv("LOG_FILE", log_dir / "pipeline.log"))

    # A *pretty* console handler if Rich is available, else plain stdout
    try:
        from rich.logging import RichHandler  # type: ignore

        console_handler: logging.Handler = RichHandler(
            show_time=False,
            omit_repeated_times=False,
            rich_tracebacks=True,
        )
    except Exception:  # pragma: no cover – Rich isn't a hard dependency
        console_handler = logging.StreamHandler(sys.stdout)

    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
        handlers=[console_handler, file_handler],
    )


# Initialise logging *before* importing any local modules that also call
# logging.getLogger() so everybody inherits the same configuration.
setup_logging()
logger = logging.getLogger(__name__)
logger.debug("Logging initialised")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Local imports (after logging + dotenv so they can rely on both)
# ---------------------------------------------------------------------------
from news_fetcher import fetch_from_all_sources  # noqa: E402
from source_manager import get_default_sources  # noqa: E402
from utils import clean_url, create_slug, extract_source_domain, format_href  # noqa: E402
from llm_selector import LLMSelector  # noqa: E402
from db_operations import DatabaseManager  # noqa: E402
from database_functions import SupabaseClient  # noqa: E402

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class NewsFetchingPipeline:
    """Coordinates the news fetching pipeline."""

    def __init__(
        self,
        sources: Optional[List[Dict[str, Any]]] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        self.sources = sources
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "openai")
        self.llm_model = llm_model
        self.llm_selector = LLMSelector(provider=self.llm_provider, model=self.llm_model)

        # Database
        try:
            self.supabase_client = SupabaseClient()
            self.db_manager = DatabaseManager(supabase_client=self.supabase_client)
            logger.info("Supabase client and DB manager ready")
        except Exception:
            logger.exception("Failed to initialise Supabase components – falling back to console mode")
            self.supabase_client = None
            self.db_manager = DatabaseManager()

        logger.info("Pipeline uses LLM: %s", self.llm_selector.get_llm_info())

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------

    @staticmethod
    def _validate_env() -> bool:
        """Return *True* when all mandatory environment variables are present."""
        required = ("SUPABASE_URL", "SUPABASE_KEY")
        missing = [var for var in required if not os.getenv(var)]
        if missing:
            for var in missing:
                logger.error("Missing required environment variable: %s", var)
            return False
        return True

    # ---------------------------------------------------------------------
    # Orchestration
    # ---------------------------------------------------------------------

    async def run(self) -> int:
        if not self._validate_env():
            return 1

        try:
            # -----------------------------------------------------------------
            # 1) Load sources
            # -----------------------------------------------------------------
            if self.sources is None:
                self.sources = await get_default_sources()

            if not self.sources:
                logger.error("No sources configured – aborting")
                return 1

            # -----------------------------------------------------------------
            # 2) Fetch
            # -----------------------------------------------------------------
            api_token = self.llm_selector.get_api_token()
            news_items = await fetch_from_all_sources(
                sources=self.sources,
                provider=self.llm_provider,
                api_token=api_token,
            )

            if not news_items:
                logger.warning("No news items fetched – pipeline ends early")
                return 1

            logger.info("Fetched %d news items", len(news_items))

            # -----------------------------------------------------------------
            # 3) Store
            # -----------------------------------------------------------------
            formatted = [
                {
                    "uniqueName": item.get("id", ""),
                    "source": item.get("source", ""),
                    "headline": item.get("headline", ""),
                    "href": item.get("href", ""),
                    "url": item.get("url", ""),
                    "publishedAt": item.get("published_at", ""),
                    "isProcessed": False,
                }
                for item in news_items
            ]

            for article in formatted:
                logger.debug("NewsItem → %s", json.dumps(article, ensure_ascii=False))

            stored = self.db_manager.store_articles(formatted)
            logger.info("Stored %d articles in SourceArticles", stored)
            return 0

        except Exception:
            logger.exception("Pipeline crashed with an unhandled exception")
            return 1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def _main() -> int:  # pragma: no cover – keeps the real entry point tiny
    pipeline = NewsFetchingPipeline()
    return await pipeline.run()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(_main()))
