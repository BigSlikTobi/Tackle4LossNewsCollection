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
from urllib.parse import urlparse


# Set up logging
# Note: Assuming logging is configured elsewhere (e.g., in pipeline.py)
# If running this file standalone, configure logging here.
logger = logging.getLogger(__name__)

# --- Debugging: Increase Verbosity ---
# Consider enabling these for detailed GHA debugging
# logging.getLogger("crawl4ai").setLevel(logging.DEBUG)
# logging.getLogger("litellm").setLevel(logging.DEBUG)
# -------------------------------------

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
    LLMSelector.ANTHROPIC: "claude-3-sonnet", # Note: Original file had claude-3-sonnet, assuming this is intended. Update if needed.
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

    # --- Debugging: Log API Token Usage ---
    if api_token:
         logger.info(f"Using API Token for crawl4ai (Provider: {provider}): {api_token[:5]}...{api_token[-4:]}")
    else:
         logger.warning(f"API Token for {provider} is missing or empty!")
    # ---------------------------------------

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
    # --- Debugging: Increase GHA timeout further if needed ---
    # timeout_ms = 90_000 if is_ci else 30_000
    # ---------------------------------------------------------

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
        # Ensure Deepseek key is set if used - check llm_selector.py logic too
        os.environ.setdefault("DEEPSEEK_API_KEY", api_token or "") # Pass token if available
        os.environ["LITELLM_MODEL_ALIAS"] = f"{model}=deepseek/{model}"
        strategy_kwargs["litellm_params"] = {
            "model": f"deepseek/{model}",
            "api_base": "https://api.deepseek.com/v1",
        }

    strategy = LLMExtractionStrategy(**strategy_kwargs)

    logger.info("Using LLM provider=%s model=%s for URL: %s", provider, model, url)

    # ---------------------------------------------------------------------
    # Crawl with retries
    # ---------------------------------------------------------------------

    result = None
    for attempt in range(1, max_retries + 2):  # first run + retries
        logger.info("Starting crawl attempt %d/%d for %s", attempt, max_retries + 1, url)
        try:
            # --- Debugging: Enable verbose crawler output ---
            async with AsyncWebCrawler(verbose=True) as crawler: # Set verbose=True
            # ---------------------------------------------
            # Original line: async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=1, # Low threshold, relies on LLM filter
                    extraction_strategy=strategy,
                    cache_mode=CacheMode.DISABLED,
                    wait_for_selector=wait_for_selector,
                    timeout=timeout_ms,
                )
            if result:
                logger.info("Crawl attempt %d for %s finished. Result has content: %s",
                            attempt, url, bool(result.extracted_content))
                if result.extracted_content:
                    break # Success!
            else:
                 logger.warning("Crawl attempt %d for %s returned no result object.", attempt, url)

            logger.warning("No content after attempt %d/%d for %s", attempt, max_retries + 1, url)

        # --- Debugging: More specific exception logging ---
        except asyncio.TimeoutError:
             logger.error("Timeout Error during crawler.arun attempt %d for %s after %d ms",
                          attempt, url, timeout_ms, exc_info=True)
        except Exception as exc:
            logger.error("Explicit error during crawler.arun attempt %d for %s: %s",
                         attempt, url, exc, exc_info=True) # Log traceback
        # -------------------------------------------------

        if attempt <= max_retries:
             logger.info("Waiting 2 seconds before retry for %s", url)
             await asyncio.sleep(2) # Wait before retrying

    if not (result and result.extracted_content):
        logger.error("Giving up – nothing extracted from %s after %d attempts", url, max_retries + 1)
        return []

    # ---------------------------------------------------------------------
    # Decode bytes → str
    # ---------------------------------------------------------------------

    extracted_raw: str
    try:
        if isinstance(result.extracted_content, str):
             extracted_raw = result.extracted_content
        elif isinstance(result.extracted_content, bytes):
             extracted_raw = result.extracted_content.decode("utf-8", "replace")
        else:
             logger.warning("Unexpected type for extracted_content: %s. Converting to string.", type(result.extracted_content))
             extracted_raw = str(result.extracted_content)

    except Exception as exc:
        logger.error("Decoding error for content from %s: %s", url, exc)
        # Try decoding differently or just represent as string
        try:
            extracted_raw = str(result.extracted_content)
            logger.warning("Falling back to str() representation after decoding error.")
        except Exception as str_exc:
             logger.error("Could not even convert extracted_content to string: %s", str_exc)
             extracted_raw = "" # Give up if even str() fails

    # --- Debugging: Log Raw Extracted Content ---
    logger.debug(f"Raw extracted content from {url} (first 500 chars): {extracted_raw[:500]}")
    # -------------------------------------------

    if not extracted_raw:
        logger.error("Empty extraction from %s after decoding/processing", url)
        return []

    # ---------------------------------------------------------------------
    # Parse JSON block inside the raw string
    # ---------------------------------------------------------------------

    data: List[Dict[str, Any]] = []
    try:
        # Be more robust finding JSON - sometimes LLM adds ```json ... ```
        json_match = re.search(r'```json\s*(\[.*\])\s*```', extracted_raw, re.DOTALL | re.MULTILINE)
        if json_match:
            json_blob = json_match.group(1)
            logger.debug("Found JSON block within markdown code fence for %s", url)
        else:
            # Fallback to finding first '[' and last ']'
            start = extracted_raw.find("[")
            end = extracted_raw.rfind("]") + 1
            if start >= 0 and end > start:
                 json_blob = extracted_raw[start:end]
                 logger.debug("Found JSON block using bracket search for %s", url)
            else:
                 json_blob = extracted_raw # Assume raw might be JSON if no brackets found
                 logger.warning("Could not find JSON array structure in raw output for %s, trying to parse raw.", url)

        parsed_json = json.loads(json_blob)

        # Ensure parsed data is a list of dicts
        if isinstance(parsed_json, list):
             data = [item for item in parsed_json if isinstance(item, dict)]
             if len(data) != len(parsed_json):
                  logger.warning("Filtered out non-dict items from parsed JSON list for %s", url)
        elif isinstance(parsed_json, dict):
             logger.warning("LLM returned a single dict, wrapping in a list for %s", url)
             data = [parsed_json]
        else:
             logger.error("Parsed JSON is not a list or dict for %s. Type: %s", url, type(parsed_json))
             data = []

    except json.JSONDecodeError as exc:
        logger.error("JSON decode error from %s: %s", url, exc)
        logger.debug("Raw content that failed JSON parsing (first 600 chars): %s…", extracted_raw[:600])
        return []
    except Exception as e:
        logger.error("Unexpected error during JSON parsing for %s: %s", url, e, exc_info=True)
        return []


    if not data:
        logger.warning("Extractor returned an empty list or non-list/dict for %s", url)
        logger.debug("Full raw content (first 600 chars): %s…", extracted_raw[:600])
        return []

    # ---------------------------------------------------------------------
    # Post‑process & clean
    # ---------------------------------------------------------------------

    cleaned: List[Dict[str, Any]] = []
    processed_slugs = set() # Track slugs to avoid duplicates from the same scrape

    for item in data:
        # Basic validation
        if not isinstance(item, dict):
             logger.warning("Skipping non-dictionary item in data list for %s: %s", url, item)
             continue

        headline = item.get("headline", "").strip()
        href = item.get("href", "").strip()

        # Require headline and href
        if not headline:
             logger.debug("Skipping item with missing headline for %s: %s", url, item)
             continue
        if not href:
             logger.debug("Skipping item with missing href for %s: %s", url, item)
             continue

        # Build ID slug - more robust slug generation
        slug_base = re.sub(r'[^\w\s-]', '', headline.lower()) # Keep spaces temporarily
        slug = re.sub(r'[-\s]+', '-', slug_base).strip('-')[:100] # Replace spaces/multiple hyphens
        if not slug: # Handle cases where headline was only special chars
             slug = f"article-{hash(headline)}"[:100]
             logger.warning("Generated hash-based slug for headline: %s", headline)

        # Ensure slug uniqueness within this batch
        original_slug = slug
        counter = 1
        while slug in processed_slugs:
             slug = f"{original_slug}-{counter}"[:100]
             counter += 1
        processed_slugs.add(slug)

        # Construct and clean full URL
        try:
             if href.startswith("http://") or href.startswith("https://"):
                  full_url = href
             elif href.startswith("//"): # Handle protocol-relative URLs
                  parsed_base = urlparse(base_url)
                  full_url = f"{parsed_base.scheme}:{href}"
             else:
                  full_url = f"{base_url.rstrip('/')}/{href.lstrip('/')}"

             clean_full_url = clean_url(full_url) # Use utility function
        except Exception as url_err:
             logger.error("Error constructing/cleaning URL for href '%s' (base: %s): %s", href, base_url, url_err)
             continue # Skip if URL processing fails


        # Check blacklist
        if is_url_blacklisted(clean_full_url, blacklist):
            logger.debug("Skipping blacklisted URL %s (original href: %s)", clean_full_url, href)
            continue

        # Get publication date (ensure it's present, default could be added if needed)
        published_at = item.get("published_at", "")
        if not published_at:
            logger.debug("Missing 'published_at' for item: %s", headline)
            # Optionally set a default, e.g., today's date
            # from datetime import date
            # published_at = date.today().isoformat()

        # Get source (might be overwritten later, but good to have fallback)
        source_name = item.get("source", "")
        if not source_name:
            logger.debug("Missing 'source' in LLM output for item: %s. Will be overwritten.", headline)


        cleaned.append(
            {
                # Use the validated/unique slug
                "id": slug,
                # Source might be overwritten by caller, use LLM output as fallback
                "source": source_name,
                "headline": headline,
                # Store original href and cleaned full URL
                "href": href,
                "url": clean_full_url,
                "published_at": published_at,
                "isProcessed": False, # Default value from Pydantic model
            }
        )

    if not cleaned:
        logger.warning("No valid articles after cleaning for %s. Raw extract (600 chars): %s…", url, extracted_raw[:600])

    logger.info("Successfully extracted and cleaned %d items from %s", len(cleaned), url)
    return cleaned

# ---------------------------------------------------------------------------
# Helper that queries several sources sequentially
# ---------------------------------------------------------------------------

async def fetch_from_all_sources(
    sources: List[Dict[str, Any]],
    provider: str,
    api_token: str,
) -> List[Dict[str, Any]]:
    """Fetch articles from all *enabled* sources and merge the results."""

    all_items: List[Dict[str, Any]] = []
    enabled_sources = [site for site in sources if site.get("execute", True)]

    if not enabled_sources:
        logger.warning("no sources are marked for execution (execute=false)")
        return []

    logger.info("Starting fetch for %d enabled sources using provider: %s", len(enabled_sources), provider)

    # Use asyncio.gather for potential concurrency (optional)
    # tasks = []
    # for site in enabled_sources:
    #      logger.info("Scheduling fetch for %s (%s)", site["name"], site["url"])
    #      tasks.append(
    #           fetch_news(
    #                url=site["url"],
    #                base_url=site["base_url"],
    #                provider=provider,
    #                api_token=api_token,
    #                # Optional: Adjust max_items per source if needed
    #                # max_items=site.get("max_items", 10),
    #           )
    #      )
    # results = await asyncio.gather(*tasks, return_exceptions=True)

    # Sequential execution (as in original code)
    results = []
    for site in enabled_sources:
         site_name = site.get("name", "Unknown Source")
         site_url = site.get("url", "No URL")
         logger.info("Fetching news from %s (%s)", site_name, site_url)
         try:
             items = await fetch_news(
                 url=site["url"],
                 base_url=site["base_url"],
                 provider=provider,
                 api_token=api_token,
                 # Optional: Adjust max_items per source if needed
                 # max_items=site.get("max_items", 10),
             )
             results.append((site, items)) # Store site info with results
         except Exception as exc:
             logger.error("Unhandled error during fetch_news for %s: %s", site_name, exc, exc_info=True)
             results.append((site, exc)) # Store exception if fetch failed


    # Process results (whether sequential or concurrent)
    for i, result_info in enumerate(results):
        # site = enabled_sources[i] # Use if using asyncio.gather
        site, result_data = result_info # Use if sequential with tuple stored
        site_name = site.get("name", "Unknown Source")

        if isinstance(result_data, Exception):
            logger.error("Fetch task for %s failed with exception: %s", site_name, result_data)
            continue # Skip this source if the fetch itself failed

        if result_data:
             logger.info("Successfully scraped %d raw items from %s", len(result_data), site_name)
             # --- Ensure correct source name is assigned ---
             for art in result_data:
                 art["source"] = site_name # Overwrite or set the source name definitively
             # ----------------------------------------------
             all_items.extend(result_data)
        else:
             logger.warning("No news items returned from fetch_news for %s", site_name)


    logger.info("Finished fetching from all sources. Total items collected: %d", len(all_items))
    return all_items