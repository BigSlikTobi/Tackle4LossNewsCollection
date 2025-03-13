# Description: Manages the news sources for the application.
from typing import Any, Dict, List
import logging
import os
import httpx
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def get_default_sources() -> List[Dict[str, Any]]:
    """
    Fetch news sources from Supabase database.
    
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