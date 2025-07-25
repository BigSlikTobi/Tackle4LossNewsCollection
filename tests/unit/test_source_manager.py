import pytest
import os
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
import logging # Added for caplog.set_level

# Module to test
from source_manager import get_default_sources

MOCK_SUPABASE_URL = "http://mock.supabase.co"
MOCK_SUPABASE_KEY = "mock_supabase_key"

# Sample raw responses from Supabase
SAMPLE_NEWS_SOURCE_RAW = [
    {"id": 1, "Name": "Source Alpha", "url": "http://alpha.com", "baseUrl": "http://alpha.com", "isExecute": True, "isPrimarySource": True},
    {"id": 2, "Name": "Source Beta", "url": "http://beta.com", "baseUrl": "http://beta.com", "isExecute": False, "isPrimarySource": False},
]
EXPECTED_DEFAULT_SOURCES_FORMATTED = [
    {"name": "Source Alpha", "url": "http://alpha.com", "base_url": "http://alpha.com", "execute": True, "primary_source": True},
    {"name": "Source Beta", "url": "http://beta.com", "base_url": "http://beta.com", "execute": False, "primary_source": False},
]

SAMPLE_TEAM_NEWS_SOURCE_RAW = [
    {"id": 10, "created_at": "2023-01-01T00:00:00Z", "url": "http://team1.com/news", "baseUrl": "http://team1.com", "Team": "Team1", "isExecute": True},
    {"id": 11, "created_at": "2023-01-02T00:00:00Z", "url": "http://team2.com/feed", "baseUrl": "http://team2.com", "Team": "Team2", "isExecute": False},
]
EXPECTED_TEAM_NEWS_SOURCES_FORMATTED = [
    {"id": 10, "created_at": "2023-01-01T00:00:00Z", "url": "http://team1.com/news", "base_url": "http://team1.com", "team": "Team1", "execute": True},
    {"id": 11, "created_at": "2023-01-02T00:00:00Z", "url": "http://team2.com/feed", "base_url": "http://team2.com", "team": "Team2", "execute": False},
]


@pytest.fixture(autouse=True)
def manage_supabase_env_vars(monkeypatch):
    """Fixture to set and clean up Supabase environment variables for tests.
    This ensures that the tests can run without needing to set these variables manually.
    Args:
        monkeypatch: pytest fixture to modify environment variables.
    Returns:
        None
    """
    monkeypatch.setenv("SUPABASE_URL", MOCK_SUPABASE_URL)
    monkeypatch.setenv("SUPABASE_KEY", MOCK_SUPABASE_KEY)
    yield
    # Monkeypatch automatically undoes its changes

@pytest.mark.asyncio
class TestGetDefaultSources:
    async def test_success_fetches_and_transforms_data(self, caplog):
        """Test that get_default_sources fetches data from Supabase and transforms it correctly.
        This verifies that the function can successfully retrieve and format news sources.
        Args:
            caplog: pytest fixture to capture log messages.
        Returns:
            None
        """
        caplog.set_level(logging.INFO)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_NEWS_SOURCE_RAW

        mock_async_client = AsyncMock(spec=httpx.AsyncClient)
        mock_async_client.__aenter__.return_value.get.return_value = mock_response

        with patch('httpx.AsyncClient', return_value=mock_async_client) as mock_httpx_constructor:
            sources = await get_default_sources()
            assert sources == EXPECTED_DEFAULT_SOURCES_FORMATTED
            mock_httpx_constructor.assert_called_once() # Ensure client was created
            mock_async_client.__aenter__.return_value.get.assert_called_once_with(
                f"{MOCK_SUPABASE_URL}/rest/v1/NewsSource",
                headers={"apikey": MOCK_SUPABASE_KEY, "Authorization": f"Bearer {MOCK_SUPABASE_KEY}"},
                params={"select": "*", "isExecute": "eq.true"}
            )
            assert f"Successfully loaded {len(EXPECTED_DEFAULT_SOURCES_FORMATTED)} sources from Supabase" in caplog.text

    async def test_empty_list_from_api(self, caplog):
        """Test that get_default_sources handles an empty list from Supabase.
        This verifies that the function can gracefully handle cases where no sources are returned.
        Args:
            caplog: pytest fixture to capture log messages.
        Returns:
            None
        """
        caplog.set_level(logging.INFO)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = [] # API returns empty list

        mock_async_client = AsyncMock(spec=httpx.AsyncClient)
        mock_async_client.__aenter__.return_value.get.return_value = mock_response

        with patch('httpx.AsyncClient', return_value=mock_async_client):
            sources = await get_default_sources()
            assert sources == []
            assert "Successfully loaded 0 sources from Supabase" in caplog.text

    async def test_missing_supabase_credentials_logs_error_returns_empty(self, monkeypatch, caplog):
        """Test that get_default_sources logs an error and returns an empty list if Supabase credentials are missing.
        This verifies that the function handles missing environment variables correctly.
        Args:
            monkeypatch: pytest fixture to modify environment variables.
            caplog: pytest fixture to capture log messages.
        Returns:
            None
        """
        caplog.set_level(logging.ERROR)
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_KEY", raising=False)

        # No httpx.AsyncClient should be called if env vars are missing
        with patch('httpx.AsyncClient') as mock_httpx_constructor:
            sources = await get_default_sources()
            assert sources == []
            assert "Supabase credentials not found in environment variables" in caplog.text
            mock_httpx_constructor.assert_not_called()

    async def test_httpx_http_status_error_logs_and_returns_empty(self, caplog):
        """Test that get_default_sources logs an error and returns an empty list on HTTP status errors.
        This verifies that the function handles HTTP errors gracefully and logs appropriate messages.
        Args:
            caplog: pytest fixture to capture log messages.       
        Returns:        
            None
        """
        caplog.set_level(logging.ERROR)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401 # Unauthorized
        mock_response.request = MagicMock(spec=httpx.Request)
        mock_response.request.url = f"{MOCK_SUPABASE_URL}/rest/v1/NewsSource"

        # Configure raise_for_status on the response mock itself
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Client Error", request=mock_response.request, response=mock_response
        )

        mock_async_client = AsyncMock(spec=httpx.AsyncClient)
        mock_async_client.__aenter__.return_value.get.return_value = mock_response

        with patch('httpx.AsyncClient', return_value=mock_async_client):
            sources = await get_default_sources()
            assert sources == []
            assert "Error loading sources from Supabase: Client Error" in caplog.text # Message from HTTPStatusError

    async def test_httpx_request_error_logs_and_returns_empty(self, caplog):
        """Test that get_default_sources logs an error and returns an empty list on request errors.
        This verifies that the function handles network-related errors gracefully.
        Args:
            caplog: pytest fixture to capture log messages.
        Returns:
            None
        """
        caplog.set_level(logging.ERROR)
        mock_async_client = AsyncMock(spec=httpx.AsyncClient)
        mock_async_client.__aenter__.return_value.get.side_effect = httpx.RequestError(
            "Network Error", request=MagicMock(spec=httpx.Request)
        )

        with patch('httpx.AsyncClient', return_value=mock_async_client):
            sources = await get_default_sources()
            assert sources == []
            assert "Error loading sources from Supabase: Network Error" in caplog.text

