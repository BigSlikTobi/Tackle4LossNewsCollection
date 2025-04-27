import asyncio
import json
import os
import re
import logging
from typing import List, Dict, Any, Type
from pydantic import BaseModel, Field
# crawl4ai imports
from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
# Local imports
from utils import clean_url
from llm_selector import LLMSelector


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class NewsItem(BaseModel):
    """Pydantic schema for a single scraped article."""

    uniqueName: str = Field(..., alias="id", description="Unique ID (lowercase with hyphens)")
    source: str = Field(..., description="Website domain or brand name")
    headline: str = Field(..., description="Extracted headline text")
    href: str = Field(..., description="Relative URL (href) from the anchor tag")
    url: str = Field(..., description="Full URL of the news item")
    publishedAt: str = Field(..., alias="published_at", description="Publication date in YYYY-MM-DD format")
    isProcessed: bool = Field(default=False, description="Flag indicating if the item has been processed")

# ---------------------------------------------------------------------------
# Static mappings
# ---------------------------------------------------------------------------

PROVIDER_MAP = {
    LLMSelector.OPENAI: "openai",
    LLMSelector.GEMINI: "google",
    LLMSelector.ANTHROPIC: "anthropic",
    LLMSelector.DEEPSEEK: "deepseek",
}

MODEL_MAP = {
    LLMSelector.OPENAI: "gpt-4o-mini",
    LLMSelector.GEMINI: "gemini-pro",
    LLMSelector.ANTHROPIC: "claude-3-sonnet",
    LLMSelector.DEEPSEEK: "deepseek-reasoner",
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_blacklist() -> List[str]:
    """Read blacklist.json and return the list of URLs."""
    try:
        with open("blacklist.json", "r", encoding="utf-8") as handle:
            data = json.load(handle)
            return [item.get("url") for item in data if item.get("url")]
    except FileNotFoundError:
        logger.warning("blacklist.json not found – continuing without a blacklist")
        return []
    except json.JSONDecodeError as exc:
        logger.error("Invalid blacklist.json: %s", exc)
        return []

def is_url_blacklisted(url: str, blacklist: List[str]) -> bool:
    """Return *True* if *url* starts with any of the black‑listed prefixes."""
    return any(url.startswith(bad) for bad in blacklist)

# ---------------------------------------------------------------------------
# Core scraping logic
# ---------------------------------------------------------------------------

async def fetch_news(
    url: str,
    base_url: str,
    provider: str,
    api_token: str,
    schema: Type[BaseModel] = NewsItem,
    max_items: int = 10,
    time_period: str = "last 24 hours",
) -> List[Dict[str, Any]]:
    """Scrape one site and return a list of JSON‑serialisable news items."""

    blacklist = load_blacklist()

    # ---------------------------------------------------------------------
    # Prompt for LLM‑based extraction
    # ---------------------------------------------------------------------

    instruction = f"""
    Extract sports news articles from the given webpage and return JSON array matching this schema:
    {schema.schema_json()}

    Rules:
      1. Collect anchor tags that link to articles – extract their visible text as *headline*.
      2. *headline* must be complete, meaningful text – ignore nav links or single words.
      3. Use the anchor's *href* for *href*; build *url* by prefixing {base_url} when *href* is relative.
      4. Set *source* to the site domain (e.g. "nfl.com").
      5. Build *id* (alias uniqueName) as a slug from *headline* (lowercase, hyphens, ascii only).
      6. Put the article's publication date (or today) in YYYY‑MM‑DD in *published_at*.
      7. Only include articles from the {time_period}.
      8. Return **at most {max_items}** items.
      9. Focus on NFL‑related content; skip ads, navigation, videos, etc.
    """

    # ---------------------------------------------------------------------
    # Environment‑specific parameters
    # ---------------------------------------------------------------------

    is_ci = os.getenv("GITHUB_ACTIONS", "").lower() == "true"
    timeout_ms = 60_000 if is_ci else 30_000
    max_retries = 3 if is_ci else 1

    # **Unified** selector that waits until *any* likely headline element exists.
    wait_for_selector = (
        "a[href*='/news'], a[href*='/story'], article, .article, "
        "a[data-qa*='headline'], h1, h2"
    )

    # ---------------------------------------------------------------------
    # Build Crawl4AI extraction strategy
    # ---------------------------------------------------------------------

    crawl4ai_provider = PROVIDER_MAP.get(provider, "openai")
    model = MODEL_MAP.get(provider, "gpt-4o-mini")

    strategy_kwargs: Dict[str, Any] = {
        "llm_provider": crawl4ai_provider,
        "llm_api_token": api_token,
        "cache_mode": CacheMode.DISABLED,
        "schema": schema.schema_json(),
        "extraction_type": "schemas",
        "instruction": instruction,
        "temperature": 0.2,
        "model": model,
    }

    if provider == LLMSelector.DEEPSEEK:
        # LiteLLM mapping for DeepSeek
        os.environ.setdefault("DEEPSEEK_API_KEY", api_token)
        os.environ["LITELLM_MODEL_ALIAS"] = f"{model}=deepseek/{model}"
        strategy_kwargs["litellm_params"] = {
            "model": f"deepseek/{model}",
            "api_base": "https://api.deepseek.com/v1",
        }

    strategy = LLMExtractionStrategy(**strategy_kwargs)

    logger.info("Using LLM provider=%s model=%s", provider, model)

    # ---------------------------------------------------------------------
    # Crawl with retries
    # ---------------------------------------------------------------------

    result = None
    for attempt in range(1, max_retries + 2):  # first run + retries
        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=1,
                    extraction_strategy=strategy,
                    cache_mode=CacheMode.DISABLED,
                    wait_for_selector=wait_for_selector,
                    timeout=timeout_ms,
                )
            if result and result.extracted_content:
                break
            logger.info("No content after attempt %d/%d for %s", attempt, max_retries + 1, url)
        except Exception as exc:
            logger.error("Error attempt %d/%d for %s: %s", attempt, max_retries + 1, url, exc)
        await asyncio.sleep(2)

    if not (result and result.extracted_content):
        logger.error("Giving up – nothing extracted from %s", url)
        return []

    # ---------------------------------------------------------------------
    # Decode bytes → str
    # ---------------------------------------------------------------------

    extracted_raw: str
    try:
        extracted_raw = (
            result.extracted_content
            if isinstance(result.extracted_content, str)
            else result.extracted_content.decode("utf-8", "replace")
        )
    except Exception as exc:
        logger.error("Decoding error: %s", exc)
        extracted_raw = str(result.extracted_content)

    if not extracted_raw:
        logger.error("Empty extraction from %s after decoding", url)
        return []

    # ---------------------------------------------------------------------
    # Parse JSON block inside the raw string
    # ---------------------------------------------------------------------

    try:
        start = extracted_raw.find("[")
        end = extracted_raw.rfind("]") + 1
        json_blob = extracted_raw[start:end] if start >= 0 and end > start else extracted_raw
        data = json.loads(json_blob)
    except json.JSONDecodeError as exc:
        logger.error("JSON decode error from %s: %s", url, exc)
        logger.debug("Raw content (first 400 chars): %s…", extracted_raw[:400])
        return []

    if not data:
        logger.warning("Extractor returned an empty list for %s", url)
        logger.debug("Full raw content (first 400 chars): %s…", extracted_raw[:400])
        return []

    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) else []

    # ---------------------------------------------------------------------
    # Post‑process & clean
    # ---------------------------------------------------------------------

    cleaned: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        headline = item.get("headline", "").strip()
        href = item.get("href", "")
        if not headline or not href:
            continue

        # Build ID slug
        slug = re.sub(r"[^\w-]", "", headline.lower().replace(" ", "-"))[:100]
        full_url = href if href.startswith("http") else f"{base_url.rstrip('/')}/{href.lstrip('/')}"
        clean_full_url = clean_url(full_url)

        if is_url_blacklisted(clean_full_url, blacklist):
            logger.debug("Skipping blacklisted URL %s", clean_full_url)
            continue

        cleaned.append(
            {
                "id": slug,
                "source": item.get("source", ""),  # overwritten by caller
                "headline": headline,
                "href": href,
                "url": clean_full_url,
                "published_at": item.get("published_at", ""),
                "isProcessed": False,
            }
        )

    if not cleaned:
        logger.debug("No valid articles after cleaning for %s. Raw extract (600 chars): %s…", url, extracted_raw[:600])

    return cleaned

# ---------------------------------------------------------------------------
# Helper that queries several sources sequentially (kept unchanged)
# ---------------------------------------------------------------------------

async def fetch_from_all_sources(
    sources: List[Dict[str, Any]],
    provider: str,
    api_token: str,
) -> List[Dict[str, Any]]:
    """Fetch articles from all *enabled* sources and merge the results."""

    all_items: List[Dict[str, Any]] = []
    for site in sources:
        if not site.get("execute", True):
            continue
        try:
            logger.info("Fetching news from %s", site["name"])
            items = await fetch_news(
                url=site["url"],
                base_url=site["base_url"],
                provider=provider,
                api_token=api_token,
            )
            if items:
                for art in items:
                    art["source"] = site["name"]  # ensure correct source name
                all_items.extend(items)
                logger.info("Scraped %d items from %s", len(items), site["name"])
            else:
                logger.warning("No news items scraped from %s", site["name"])
        except Exception as exc:
            logger.error("Error scraping %s: %s", site["name"], exc)
    return all_items
