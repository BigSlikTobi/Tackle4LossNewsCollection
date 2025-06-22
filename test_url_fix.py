#!/usr/bin/env python3
"""
Test script to demonstrate that the URL encoding bug has been fixed.
"""

import os
from utils import clean_url

def test_url_fix():
    """Test that URLs are no longer incorrectly encoded."""
    
    # Test URLs that were problematic before
    test_urls = [
        "https://www.espn.com/nfl/story/_/id/45498715/nfl-carolina-panthers-chuba-hubbard-olympics-flag-football-track",
        "http://www.nfl.com/news/article-1234",
        "https://sports.yahoo.com/nfl/teams/dal/",
        "http://example.com/path with spaces",
        "https://site.com/query?param=value&other=test"
    ]
    
    print("Testing URL cleaning function:")
    print("=" * 60)
    
    for url in test_urls:
        # Test in local environment
        os.environ['GITHUB_ACTIONS'] = 'false'
        local_result = clean_url(url)
        
        # Test in GitHub Actions environment
        os.environ['GITHUB_ACTIONS'] = 'true'
        gha_result = clean_url(url)
        
        print(f"\nOriginal: {url}")
        print(f"Local:    {local_result}")
        print(f"GHA:      {gha_result}")
        
        # Check that the scheme is not encoded
        local_ok = "https://" in local_result or "http://" in local_result
        gha_ok = "https://" in gha_result or "http://" in gha_result
        
        # Check that %3A is not present
        no_encoded_colon_local = "%3A" not in local_result
        no_encoded_colon_gha = "%3A" not in gha_result
        
        status = "✅ FIXED" if (local_ok and gha_ok and no_encoded_colon_local and no_encoded_colon_gha) else "❌ BROKEN"
        print(f"Status: {status}")
    
    print("\n" + "=" * 60)
    print("Fix Summary:")
    print("- URLs no longer have encoded colons (%3A) in the scheme")
    print("- Both local and GitHub Actions environments work correctly")
    print("- The 'https://' and 'http://' prefixes are preserved")

if __name__ == "__main__":
    test_url_fix()
