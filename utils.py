"""
URL utilities for cleaning and processing URLs in the news fetching pipeline.
"""
import os
import re
import urllib.parse
from urllib.parse import urlparse
import unicodedata
import logging

# Set up logging
logger = logging.getLogger(__name__)

def remove_control_chars(s: str) -> str:
    """Remove all Unicode control characters from a string."""
    return ''.join(ch for ch in s if not unicodedata.category(ch).startswith('C'))

def build_url_from_parts(parts: urllib.parse.ParseResult) -> str:
    """Rebuild the URL from its parts, stripping extra whitespace and control characters."""
    scheme = parts.scheme.strip()
    netloc = parts.netloc.strip()
    path_segments = [segment.strip() for segment in parts.path.split('/') if segment.strip()]
    path = '/' + '/'.join(path_segments) if path_segments else ''

    query_params_list = [param.strip() for param in parts.query.split('&') if param.strip()]
    processed_query_params = []
    for param_group in query_params_list:
        if '=' in param_group:
            key, value = param_group.split('=', 1)
            processed_query_params.append(f"{key.strip()}={value.strip()}")
        else:
            # Handle parameters without values, e.g., 'flag' in 'query?flag'
            processed_query_params.append(param_group.strip())
    query = '&'.join(processed_query_params)

    fragment = parts.fragment.strip()
    return urllib.parse.urlunparse((scheme, netloc, path, parts.params, query, fragment))

def clean_url(url: str) -> str:
    """Clean URL by removing control characters and performing percent-encoding."""
    if not url:
        return url
    
    # Remove Unicode control characters and trim
    url = remove_control_chars(url).strip()
    
    # Special handling for GitHub Actions environment
    if os.getenv("GITHUB_ACTIONS", "").lower() == "true":
        logger.debug("Detected GitHub Actions environment; rebuilding URL from parts.")
        parts = urllib.parse.urlparse(url)
        url = build_url_from_parts(parts)
        logger.debug(f"Rebuilt URL: {url}")
    else:
        # Standard cleaning for local environments
        url = url.replace('\n', '').replace('\r', '')
        url = ''.join(char for char in url if 32 <= ord(char) <= 126)
        url = re.sub(r'\s+', '-', url.strip())
        
        try:
            parts = urllib.parse.urlparse(url)
            path = urllib.parse.quote(parts.path)
            query = urllib.parse.quote_plus(parts.query, safe='=&')
            url = urllib.parse.urlunparse((
                parts.scheme,
                parts.netloc,
                path,
                parts.params,
                query,
                parts.fragment
            ))
        except Exception as e:
            logger.warning(f"Error cleaning URL {url}: {e}")
            return url
    
    # Allow ':', '?', '=', '&' to be percent-encoded by removing them from the safe set.
    # Retain '/', '%', '#' as safe.
    final_url = urllib.parse.quote(url, safe="/%#")
    return final_url

def is_valid_url(url: str) -> bool:
    """Validate if a URL is properly formatted."""
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False
    
def create_slug(headline: str) -> str:
    """
    Create a URL-friendly slug from a headline.
    
    Args:
        headline: The headline to convert to a slug
        
    Returns:
        A URL-friendly slug string
    """
    # Convert to lowercase and replace spaces with hyphens
    slug = headline.lower()
    # Remove special characters and replace spaces with hyphens
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug).strip('-')
    return slug

def extract_source_domain(url: str) -> str:
    """Extract the source domain from a URL without www. prefix"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        return domain.replace('www.', '')
    except:
        return ''

def format_href(url: str) -> str:
    """Format the href path from the full URL"""
    try:
        parsed = urlparse(url)
        return parsed.path
    except:
        return ''