# Description: Manages the news sources for the application.
from typing import Any, Dict, List
import logging
import os
import httpx

# Set up logging
logger = logging.getLogger(__name__)

async def get_default_sources() -> List[Dict[str, Any]]:
    """
    Fetch news sources from Supabase database.
    This function retrieves the list of news sources from the NewsSource table in Supabase.
    It returns a list of dictionaries, each containing the source name, URL, base URL,
    execution status, and whether it is a primary source.
    Args:
        None
    Returns:
        List of news source configurations
    """
    try:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase credentials not found in environment variables")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{supabase_url}/rest/v1/NewsSource",
                headers={
                    "apikey": supabase_key,
                    "Authorization": f"Bearer {supabase_key}"
                },
                params={
                    "select": "*",
                    "isExecute": "eq.true"
                }
            )
            response.raise_for_status()
            sources = response.json()
            
            # Transform the data to match the expected format
            formatted_sources = [{
                "name": source["Name"],
                "url": source["url"],
                "base_url": source["baseUrl"],
                "execute": source["isExecute"],
                "primary_source": source.get("isPrimarySource", False)
            } for source in sources]
            
            logger.info(f"Successfully loaded {len(formatted_sources)} sources from Supabase")
            return formatted_sources
            
    except Exception as e:
        logger.error(f"Error loading sources from Supabase: {e}")
        return []  # Return empty list if sources cannot be loaded

async def get_team_news_sources() -> List[Dict[str, Any]]:
    """
    Fetch team news sources from Supabase database.
    This function retrieves the list of team news sources from the TeamNewsSource table in Supabase.
    It returns a list of dictionaries, each containing the source ID, creation date, URL,
    base URL, team name, and execution status.
    Args:
        None  
    Returns:
        List of team news source configurations
    """
    try:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase credentials not found in environment variables")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{supabase_url}/rest/v1/TeamNewsSource",
                headers={
                    "apikey": supabase_key,
                    "Authorization": f"Bearer {supabase_key}"
                },
                params={
                    "select": "*",
                    "isExecute": "eq.true"  # Only fetch sources where isExecute is true
                }
            )
            response.raise_for_status()
            sources = response.json()
            
            # Transform the data to match the expected format
            formatted_sources = [{
                "id": source["id"],
                "created_at": source["created_at"],
                "url": source["url"],
                "base_url": source["baseUrl"],
                "team": source["Team"],
                "execute": source["isExecute"]
            } for source in sources]
            
            logger.info(f"Successfully loaded {len(formatted_sources)} team news sources from Supabase")
            return formatted_sources
            
    except Exception as e:
        logger.error(f"Error loading team news sources from Supabase: {e}")
        return []  # Return empty list if sources cannot be loaded