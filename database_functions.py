import os
import logging
from supabase import create_client, Client  # Import Client
from typing import Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO)

class SupabaseClient:
    def __init__(self) -> None:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            raise EnvironmentError("Supabase credentials are not set in the environment.")
        self.client = create_client(supabase_url, supabase_key)
        self.source_id_map = {}
        self._load_source_ids()

    def _load_source_ids(self) -> None:
        """Load source IDs from the NewsSource table."""
        try:
            result = self.client.table('NewsSource').select('id,Name').execute()
            if result.data:
                self.source_id_map = {source['Name'].upper(): source['id'] for source in result.data}
                logging.info(f"Loaded source IDs: {self.source_id_map}")
        except Exception as e:
            logging.error(f"Error loading source IDs: {e}")
            raise

    def get_source_id(self, source_name: str) -> Optional[int]:
        """Get the source ID for a given source name."""
        return self.source_id_map.get(source_name.upper())

    def store_article_to_source_articles(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Store an article in the SourceArticles table.
        
        Args:
            article: Article data dictionary
            
        Returns:
            The stored article data if successful, None otherwise
        """
        try:
            source_name = article.get("source", "")
            source_id = self.get_source_id(source_name)
            
            if not source_id:
                raise ValueError(f"No source ID found for source name: {source_name}")

            # Prepare the data according to the SourceArticles table format
            data = {
                "uniqueName": article.get("uniqueName", article.get("id", "")),
                "source": source_id,  # Use the numeric source ID
                "headline": article.get("headline"),
                "href": article.get("href"),
                "url": article.get("url"),
                "publishedAt": article.get("publishedAt", article.get("published_at")),
                "isProcessed": False
            }

            # Insert into Supabase SourceArticles table
            result = self.client.table('SourceArticles').insert(data).execute()
            logging.info(f"Successfully stored article in SourceArticles: {data['headline']}")
            return result.data[0] if result.data else None

        except Exception as e:
            logging.error(f"Error posting to SourceArticles table: {e}")
            raise
            
    def store_articles_to_source_articles(self, articles: List[Dict[str, Any]]) -> int:
        """
        Store multiple articles in the SourceArticles table.
        
        Args:
            articles: List of article data dictionaries
            
        Returns:
            Number of articles successfully stored
        """
        stored_count = 0
        for article in articles:
            try:
                self.store_article_to_source_articles(article)
                stored_count += 1
            except Exception as e:
                logging.error(f"Failed to store article {article.get('headline', 'Unknown')}: {e}")
        
        return stored_count