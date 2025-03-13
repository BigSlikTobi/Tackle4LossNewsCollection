"""
Main script for running the news fetching pipeline.
"""
import asyncio
import logging
import os
import sys
import json
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Local imports
from news_fetcher import fetch_from_all_sources
from source_manager import get_default_sources
from utils import clean_url, create_slug, extract_source_domain, format_href
from llm_selector import LLMSelector, get_available_llm_providers
from db_operations import DatabaseManager
from database_functions import SupabaseClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class NewsFetchingPipeline:
    """Coordinates the news fetching pipeline."""
    
    def __init__(self, sources: Optional[List[Dict[str, Any]]] = None, llm_provider: Optional[str] = None, llm_model: Optional[str] = None):
        """
        Initialize the news fetching pipeline.
        
        Args:
            sources: List of news sources to fetch from (uses defaults if None)
            llm_provider: The LLM provider to use (e.g., "openai", "gemini")
            llm_model: Specific model to use (optional)
        """
        self.sources = sources
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "openai")
        self.llm_model = llm_model
        self.llm_selector = LLMSelector(provider=self.llm_provider, model=self.llm_model)
        
        # Initialize database components
        try:
            self.supabase_client = SupabaseClient()
            self.db_manager = DatabaseManager(supabase_client=self.supabase_client)
            logger.info("Successfully initialized Supabase client and database manager")
        except Exception as e:
            logger.error(f"Failed to initialize database components: {e}")
            self.supabase_client = None
            self.db_manager = DatabaseManager()
            
        logger.info(f"News fetching pipeline initialized with LLM: {self.llm_selector.get_llm_info()}")
    
    def validate_environment(self) -> bool:
        """Validate that all required environment variables are set."""
        required_vars = [
            "SUPABASE_URL",
            "SUPABASE_KEY"
        ]
        
        for var in required_vars:
            if not os.getenv(var):
                logger.error(f"Missing required environment variable: {var}")
                return False
        return True

    async def run(self) -> int:
        """Run the news fetching pipeline."""
        if not self.validate_environment():
            return 1
            
        try:
            # Load sources if not provided in constructor
            if self.sources is None:
                self.sources = await get_default_sources()
                
            if not self.sources:
                logger.error("No sources configured")
                return 1
                
            api_token = self.llm_selector.get_api_token()
            
            news_items = await fetch_from_all_sources(
                sources=self.sources,
                provider=self.llm_provider,
                api_token=api_token
            )
            
            if not news_items:
                logger.warning("No news items fetched")
                return 1
                
            logger.info(f"Successfully fetched {len(news_items)} news items") 
            
            # Format and store news items in the database
            formatted_items = []
            for item in news_items:
                news_item = {
                    "uniqueName": item.get("id", ""),
                    "source": item.get("source", ""),
                    "headline": item.get("headline", ""),
                    "href": item.get("href", ""),
                    "url": item.get("url", ""),
                    "publishedAt": item.get("published_at", ""),
                    "isProcessed": False
                }
                formatted_items.append(news_item)
                # Log each news item for debugging
                logger.info(f"NewsItem: {json.dumps(news_item, indent=3, ensure_ascii=False)}")
                
            # Store all articles in the database
            stored_count = self.db_manager.store_articles(formatted_items)
            logger.info(f"Stored {stored_count} articles in the SourceArticles database table")
                
            return 0
          
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return 1

async def main():
    """Main entry point for the news fetching pipeline."""
    pipeline = NewsFetchingPipeline()
    return await pipeline.run()

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))