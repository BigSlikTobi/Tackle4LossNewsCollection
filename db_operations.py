"""
Database operations for the news fetching pipeline.
"""
import logging
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Handle database operations for news articles."""
    def __init__(self, supabase_client=None):
        """
        Initialize the database manager.
        If a Supabase client is provided, it will be used for database operations.
        If not, it will operate in a mock mode, printing to console instead of storing in the database.
        Args:
            supabase_client: SupabaseClient instance for database operations
        Returns:
            None
        """
        self.supabase_client = supabase_client
        if supabase_client:
            logger.info("Initialized DatabaseManager with Supabase client")
        else:
            logger.info("Initialized DatabaseManager with console output mode (mock)")
    
    def store_article(self, article: Dict[str, Any]) -> bool:
        """
        Store a single article in the database.
        This method retrieves the source ID for the article's source and stores the article data in the SourceArticles table.
        If the Supabase client is not provided, it will print the article data to the console.  
        Args:
            article: Article data dictionary with keys like 'source', 'uniqueName', 'id', etc.      
        Returns:
            True if successful, False otherwise
        """
        article_name = article.get("uniqueName", article.get("id", "Unknown"))
        
        if self.supabase_client:
            try:
                result = self.supabase_client.store_article_to_source_articles(article)
                return result is not None
            except Exception as e:
                logger.error(f"Error storing article {article_name}: {e}")
                return False
        else:
            # Mock storage with console output
            logger.info(f"[MOCK DB] Would store article: {article_name}")
            logger.info(f"Article content: {article}")
            return True
    
    def store_articles(self, articles: List[Dict[str, Any]]) -> int:
        """
        Store multiple articles in the database.
        This method iterates over a list of article dictionaries and stores each one in the SourceArticles table.
        If the Supabase client is not provided, it will print the articles to the console.
        Args:
            articles: List of article data dictionaries with keys like 'source', 'uniqueName', 'id', etc.
        Returns:
            Number of articles successfully processed
        """
        if self.supabase_client:
            try:
                return self.supabase_client.store_articles_to_source_articles(articles)
            except Exception as e:
                logger.error(f"Error storing articles batch: {e}")
                return 0
        else:
            # Mock storage with console output
            logger.info(f"[MOCK DB] Would store {len(articles)} articles")
            for article in articles:
                self.store_article(article)
            return len(articles)
        
    def get_existing_articles(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve existing articles from the database.
        This method is a placeholder for future implementation.
        If the Supabase client is not provided, it will return an empty list.
        If implemented, it should return a list of article data dictionaries.
        Args:
            limit: Maximum number of articles to retrieve
        If the Supabase client is not provided, it will return an empty list.
        If implemented, it should return a list of article data dictionaries.   
        Returns:
            List of article data dictionaries
        If the Supabase client is not provided, it will return an empty list.
        If implemented, it should return a list of article data dictionaries.
        """
        logger.info(f"Would retrieve up to {limit} articles")
        # For now, just return an empty list
        # TODO: Implement actual retrieval from Supabase
        return []