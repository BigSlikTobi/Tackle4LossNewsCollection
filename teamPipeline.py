"""
Script for running the team news fetching pipeline.
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
from source_manager import get_team_news_sources
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

class TeamNewsFetchingPipeline:
    """Coordinates the team news fetching pipeline."""
    
    def __init__(self, sources: Optional[List[Dict[str, Any]]] = None, llm_provider: Optional[str] = None, llm_model: Optional[str] = None):
        """
        Initialize the team news fetching pipeline.
        
        Args:
            sources: List of team news sources to fetch from (uses defaults if None)
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
            logger.info("Successfully initialized Supabase client")
        except Exception as e:
            logger.error(f"Failed to initialize database components: {e}")
            self.supabase_client = None
            
        logger.info(f"Team news fetching pipeline initialized with LLM: {self.llm_selector.get_llm_info()}")
    
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
    
    def store_team_article(self, article: Dict[str, Any], source_id: int) -> Optional[Dict[str, Any]]:
        """
        Store an article in the TeamSourceArticles table.
        
        Args:
            article: Article data dictionary
            source_id: The ID of the source
            
        Returns:
            The stored article data if successful, None otherwise
        """
        try:
            # Prepare the data according to the TeamSourceArticles table format
            data = {
                "uniqueName": article.get("uniqueName", ""),
                "source": source_id,  # Use the source ID directly
                "headline": article.get("headline", ""),
                "href": article.get("href", ""),
                "url": article.get("url", ""),
                "publishedAt": article.get("publishedAt", ""),
                "isProcessed": False,
                "contentType": None,
                "author": None,
                "content": None,
                "isArticleCreated": False
            }

            # Insert into Supabase TeamSourceArticles table
            result = self.supabase_client.client.table('TeamSourceArticles').insert(data).execute()
            logger.info(f"Successfully stored team article: {data['headline']}")
            return result.data[0] if result.data else None

        except Exception as e:
            logger.error(f"Error posting to TeamSourceArticles table: {e}")
            return None
    
    def store_team_articles(self, articles: List[Dict[str, Any]], source_id_map: Dict[str, int]) -> int:
        """
        Store multiple articles in the TeamSourceArticles table.
        
        Args:
            articles: List of article data dictionaries
            source_id_map: Mapping of source names to source IDs
            
        Returns:
            Number of articles successfully stored
        """
        stored_count = 0
        for article in articles:
            try:
                source_name = article.get("source", "")
                source_id = source_id_map.get(source_name)
                
                if not source_id:
                    logger.warning(f"No source ID found for source name: {source_name}")
                    continue
                
                self.store_team_article(article, source_id)
                stored_count += 1
            except Exception as e:
                logger.error(f"Failed to store team article {article.get('headline', 'Unknown')}: {e}")
        
        return stored_count
    
    async def run(self) -> int:
        """Run the team news fetching pipeline."""
        if not self.validate_environment():
            return 1
            
        try:
            # Load team sources if not provided in constructor
            if self.sources is None:
                self.sources = await get_team_news_sources()
                
            if not self.sources:
                logger.error("No team news sources configured")
                return 1
            
            # Create a mapping of source names to source IDs
            source_id_map = {str(source['team']): source['id'] for source in self.sources}
            
            # Format sources to match expected structure in fetch_from_all_sources
            formatted_sources = [{
                "name": str(source['team']),  # Use just the team number as the name
                "url": source["url"],
                "base_url": source["base_url"],
                "execute": source["execute"],
                "primary_source": False,
                # Include team info for later reference
                "team": source["team"]
            } for source in self.sources]
            
            # Only use sources where execute is True
            active_sources = [source for source in formatted_sources if source["execute"]]
            
            if not active_sources:
                logger.warning("No active team news sources found (all have isExecute=false)")
                return 1
                
            api_token = self.llm_selector.get_api_token()
            
            news_items = await fetch_from_all_sources(
                sources=active_sources,
                provider=self.llm_provider,
                api_token=api_token
            )
            
            if not news_items:
                logger.warning("No team news items fetched")
                return 1
                
            logger.info(f"Successfully fetched {len(news_items)} team news items") 
            
            # Format news items for database
            formatted_items = []
            for item in news_items:
                news_item = {
                    "uniqueName": item.get("id", ""),
                    "source": item.get("source", ""),
                    "headline": item.get("headline", ""),
                    "href": item.get("href", ""),
                    "url": item.get("url", ""),
                    "publishedAt": item.get("published_at", "")
                }
                formatted_items.append(news_item)
            
            # Print the results to the terminal
            print("\n===== TEAM NEWS RESULTS =====")
            print(json.dumps(formatted_items, indent=4))
            print(f"Total team news items fetched: {len(formatted_items)}")
            print("=============================\n")
            
            # Store in database if we have a valid Supabase client
            if self.supabase_client:
                stored_count = self.store_team_articles(formatted_items, source_id_map)
                logger.info(f"Stored {stored_count} articles in the TeamSourceArticles database table")
            else:
                logger.warning("No Supabase client available, skipping database storage")
            
            return 0
          
        except Exception as e:
            logger.error(f"Team pipeline error: {e}")
            return 1

async def main():
    """Main entry point for the team news fetching pipeline."""
    pipeline = TeamNewsFetchingPipeline()
    return await pipeline.run()

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))