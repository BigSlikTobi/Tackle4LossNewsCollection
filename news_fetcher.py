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

# Define JSON schema for news items
class NewsItem(BaseModel):
    uniqueName: str = Field(..., alias="id", description="Unique ID (lowercase with hyphens)")
    source: str = Field(..., description="Website domain or brand name")
    headline: str = Field(..., description="Extracted headline text")
    href: str = Field(..., description="Relative URL (href) from the anchor tag")
    url: str = Field(..., description="Full URL of the news item")
    publishedAt: str = Field(..., alias="published_at", description="Publication date in YYYY-MM-DD format")
    isProcessed: bool = Field(default=False, description="Flag indicating if the item has been processed")

# Map provider names to what Crawl4AI expects
PROVIDER_MAP = {
    LLMSelector.OPENAI: "openai",
    LLMSelector.GEMINI: "google",
    LLMSelector.ANTHROPIC: "anthropic",
    LLMSelector.DEEPSEEK: "deepseek",  # Changed to directly use 'deepseek' as provider
}

# Model names for the Crawl4AI LLMExtractionStrategy
MODEL_MAP = {
    LLMSelector.OPENAI: "gpt-4o-mini",
    LLMSelector.GEMINI: "gemini-pro",
    LLMSelector.ANTHROPIC: "claude-3-sonnet",
    LLMSelector.DEEPSEEK: "deepseek-reasoner",  
}

async def fetch_news(
    url: str,
    base_url: str,
    provider: str,
    api_token: str,
    schema: Type[BaseModel] = NewsItem,
    max_items: int = 10,
    time_period: str = "last 24 hours"
) -> List[Dict[str, Any]]:
    """
    Fetch news from a specific website using AI-powered extraction.
    
    Args:
        url: The URL to scrape news from
        base_url: The base URL for constructing complete URLs
        provider: The LLM provider (e.g., "openai", "gemini", "deepseek")
        api_token: The API token for the LLM provider
        schema: Pydantic schema for the news items
        max_items: Maximum number of items to fetch
        time_period: Time period for news articles
    
    Returns:
        List of news items as dictionaries
    """
    
    # Enhanced LLM extraction strategy with improved prompt
    instruction = f"""
    Extract sports news articles from the given webpage with these rules:
    1. Look for anchor tags containing article links and extract their text content as headlines
    2. The 'headline' field MUST contain the actual text of the news headline from the anchor tag or article title
    3. For the 'href' field, extract ONLY the href attribute value from anchor tags (no base URL)
    4. For the 'url' field, combine the base URL ({base_url}) with the 'href' value to form a complete URL
    5. Set 'source' to the website domain name (e.g., 'nfl.com', 'espn.com')
    6. For 'id', create a slug from the headline (lowercase, replace spaces with hyphens, no special chars)
    7. For 'published_at', use the date from the article or current date in YYYY-MM-DD format
    8. Only include articles from the {time_period}
    9. Return at most {max_items} items
    10. Focus on NFL news articles and ignore non-news content like ads or navigation links
    11. Headlines should be complete, meaningful sentences or phrases, not just single words
    
    Pay special attention to elements with classes containing 'article', 'headline', 'news', etc.
    Extract headlines from both anchor text and any nearby heading elements (h1, h2, h3) associated with the article.
    """
    
    is_github_actions = os.getenv("GITHUB_ACTIONS", "").lower() == "true"
    
    try:
        # Use a longer timeout and more retries in GitHub Actions environment
        timeout_ms = 60000 if is_github_actions else 30000  # 60 seconds for GitHub Actions
        max_retries = 3 if is_github_actions else 1
        
        # Use a more forgiving strategy in GitHub Actions
        wait_for_selector = "body" if is_github_actions else "a[href*='/news/'], a[href*='/story/'], article, .article"
        
        # Map the provider to the format expected by Crawl4AI
        crawl4ai_provider = PROVIDER_MAP.get(provider, "openai")
        model = MODEL_MAP.get(provider, "gpt-4o-mini")
        
        # Configure the LLM extraction strategy differently based on provider
        strategy_params = {
            "llm_provider": crawl4ai_provider,
            "llm_api_token": api_token,
            "cache_mode": CacheMode.DISABLED,
            "schema": schema.schema_json(),
            "extraction_type": "schemas",
            "instruction": instruction,
            "temperature": 0.2,
            "model": model,
        }
        
        # Special handling for DeepSeek
        if provider == LLMSelector.DEEPSEEK:
            # Configure environment for LiteLLM to use DeepSeek
            # These environment variables will be picked up by LiteLLM
            os.environ["DEEPSEEK_API_KEY"] = api_token
            os.environ["LITELLM_MODEL_ALIAS"] = f"{model}=deepseek/{model}"
            
            # Update parameters specific to DeepSeek
            strategy_params.update({
                "litellm_params": {
                    "model": f"deepseek/{model}",
                    "api_base": "https://api.deepseek.com/v1"
                }
            })
            
            logger.info(f"Using DeepSeek with model: {model} and specific configuration")
        
        # Create the strategy with the appropriate parameters
        strategy = LLMExtractionStrategy(**strategy_params)
        
        logger.info(f"Using LLM provider: {provider} with model: {model}")
        
        result = None
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                async with AsyncWebCrawler(verbose=False) as crawler:
                    result = await crawler.arun(
                        url=url,
                        word_count_threshold=1,
                        extraction_strategy=strategy,
                        cache_mode=CacheMode.DISABLED,
                        wait_for_selector=wait_for_selector,
                        timeout=timeout_ms
                    )
                if result and result.extracted_content:
                    break
                retry_count += 1
                if retry_count <= max_retries:
                    logger.info(f"Retry {retry_count}/{max_retries} for {url}")
                    await asyncio.sleep(2)  # Wait 2 seconds before retrying
            except Exception as e:
                logger.error(f"Error during crawling attempt {retry_count}: {e}")
                retry_count += 1
                if retry_count <= max_retries:
                    logger.info(f"Retry {retry_count}/{max_retries} for {url}")
                    await asyncio.sleep(2)
        
        # If we have no result after all retries or extraction failed
        if result is None or result.extracted_content is None:
            logger.error(f"Error: No content extracted from {url}")
            return []
        
        # Handle potential encoding issues
        try:
            if isinstance(result.extracted_content, str):
                decoded_content = result.extracted_content
            else:
                decoded_content = result.extracted_content.decode('utf-8', 'replace')
        except Exception as e:
            logger.error(f"Decoding error: {e}")
            try:
                decoded_content = result.extracted_content.encode('latin-1', 'replace').decode('utf-8', 'replace')
            except:
                # Last resort fallback
                decoded_content = str(result.extracted_content)
        
        # Check again if decoded_content is None before parsing JSON
        if not decoded_content:
            logger.error(f"Error: Failed to decode content from {url}")
            return []
        
        # Try to parse the JSON
        try:
            # Remove any leading/trailing non-JSON content
            json_start = decoded_content.find('[')
            json_end = decoded_content.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = decoded_content[json_start:json_end]
                extracted_data = json.loads(json_content)
            else:
                # Try to parse the whole content as JSON
                extracted_data = json.loads(decoded_content)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.debug(f"Raw content: {decoded_content[:200]}...")  # First 200 chars for debugging
            return []
        
        # Process the extracted data
        if not extracted_data:
            logger.warning(f"No data extracted from {url}")
            return []
            
        if not isinstance(extracted_data, list):
            logger.warning(f"Unexpected data format: {type(extracted_data)}")
            # Try to convert to list if it's not already
            if isinstance(extracted_data, dict):
                extracted_data = [extracted_data]
            else:
                return []
        
        # Clean and process the data
        cleaned_data = []
        for item in extracted_data:
            if not isinstance(item, dict):
                continue
                
            # Skip items missing required fields or with empty headlines
            if not all(key in item for key in ["headline", "href"]) or not item.get("headline"):
                continue
                
            # Clean and normalize the headline
            headline = item.get("headline", "").strip()
            if not headline:
                continue
                
            # Create the ID slug from the headline
            item["id"] = re.sub(r'[^\w\-]', '', headline.lower().replace(" ", "-"))[:100]  # Limit length to 100 chars
            
            # Handle URL construction
            href = item.get("href", "")
            if href.startswith("http"):
                item["url"] = href
            else:
                # Ensure the href starts with a slash if needed
                if not href.startswith('/') and not base_url.endswith('/'):
                    href = '/' + href
                item["url"] = base_url + href
            
            # Clean URL
            item["url"] = clean_url(item["url"])
            
            # Ensure isProcessed is set to False for new articles
            item["isProcessed"] = False
            
            cleaned_data.append(item)
        
        return cleaned_data
        
    except Exception as e:
        logger.error(f"Error during scraping {url}: {e}")
        return []

async def fetch_from_all_sources(
    sources: List[Dict[str, Any]], 
    provider: str, 
    api_token: str
) -> List[Dict[str, Any]]:
    """
    Fetch news from all configured sources.
    
    Args:
        sources: List of source configurations with name, url, and base_url
        provider: LLM provider identifier
        api_token: API token for the LLM provider
        
    Returns:
        Combined list of all fetched news items
    """
    all_news_items = []
    
    for site in sources:
        if site.get("execute", True):
            try:
                logger.info(f"Fetching news from {site['name']}")
                news_items = await fetch_news(
                    url=site["url"],
                    base_url=site["base_url"],
                    provider=provider,
                    api_token=api_token
                )
                
                if news_items:
                    # Make sure source is set to the site name
                    for item in news_items:
                        item["source"] = site["name"]
                    
                    all_news_items.extend(news_items)
                    logger.info(f"Scraped {len(news_items)} news items from {site['name']}")
                else:
                    logger.warning(f"No news items scraped from {site['name']}")
            except Exception as e:
                logger.error(f"Error scraping {site['name']}: {e}")
    
    return all_news_items