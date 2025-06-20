import pytest
import logging # Added for caplog.set_level
from unittest.mock import MagicMock, patch

from db_operations import DatabaseManager
# from database_functions import SupabaseClient # Avoid direct import if not strictly needed

@pytest.fixture
def mock_supabase_client_for_ops(): # Renamed for clarity within this test file
    # This is a mock of our own SupabaseClient from database_functions.py
    client = MagicMock(name="MockSupabaseClientForDbOps")
    # Set default return values for its methods that DatabaseManager will call
    client.store_article_to_source_articles.return_value = {"id": 1, "headline": "Stored via SupabaseClient"}
    client.store_articles_to_source_articles.return_value = 1 # Default count of stored articles
    return client

@pytest.fixture
def db_manager_with_mock_supabase(mock_supabase_client_for_ops): # Renamed
    return DatabaseManager(supabase_client=mock_supabase_client_for_ops)

@pytest.fixture
def db_manager_console_mode(): # Renamed
    return DatabaseManager(supabase_client=None)


class TestDatabaseManagerInitialization:
    def test_init_with_supabase_client(self, mock_supabase_client_for_ops, caplog):
        caplog.set_level(logging.INFO)
        manager = DatabaseManager(supabase_client=mock_supabase_client_for_ops)
        assert manager.supabase_client == mock_supabase_client_for_ops
        assert "Initialized DatabaseManager with Supabase client" in caplog.text

    def test_init_console_mode(self, caplog): # Renamed
        caplog.set_level(logging.INFO)
        manager = DatabaseManager(supabase_client=None)
        assert manager.supabase_client is None
        assert "Initialized DatabaseManager with console output mode (mock)" in caplog.text

class TestDatabaseManagerStoreArticle:
    def test_store_article_with_supabase_success(self, db_manager_with_mock_supabase, mock_supabase_client_for_ops):
        article = {"uniqueName": "test-article", "headline": "Test"}
        # mock_supabase_client_for_ops.store_article_to_source_articles already configured for success

        result = db_manager_with_mock_supabase.store_article(article)

        assert result is True # As store_article_to_source_articles returns a dict
        mock_supabase_client_for_ops.store_article_to_source_articles.assert_called_once_with(article)

    def test_store_article_with_supabase_failure_returns_false(self, db_manager_with_mock_supabase, mock_supabase_client_for_ops, caplog): # Renamed
        article = {"uniqueName": "test-article", "headline": "Test"}
        # Simulate the SupabaseClient's method returning None (indicating failure)
        mock_supabase_client_for_ops.store_article_to_source_articles.return_value = None

        result = db_manager_with_mock_supabase.store_article(article)

        assert result is False # store_article should return False if the underlying call returned None
        # db_operations.store_article itself doesn't log here if SupabaseClient returns None.
        # It only logs if SupabaseClient raises an exception.

    def test_store_article_with_supabase_exception_returns_false_and_logs(self, db_manager_with_mock_supabase, mock_supabase_client_for_ops, caplog): # Renamed
        article = {"uniqueName": "test-article-exc", "headline": "Test Exc"}
        mock_supabase_client_for_ops.store_article_to_source_articles.side_effect = Exception("Supabase Is Down")

        result = db_manager_with_mock_supabase.store_article(article)

        assert result is False
        assert "Error storing article test-article-exc: Supabase Is Down" in caplog.text

    def test_store_article_console_mode_logs_and_succeeds(self, db_manager_console_mode, caplog): # Renamed
        caplog.set_level(logging.INFO)
        article = {"uniqueName": "mock-article", "id": "fallback-id", "headline": "Mock Test"}
        result = db_manager_console_mode.store_article(article)
        assert result is True
        assert "[MOCK DB] Would store article: mock-article" in caplog.text # Prefers uniqueName
        assert f"Article content: {article}" in caplog.text


class TestDatabaseManagerStoreArticles:
    def test_store_articles_with_supabase_success(self, db_manager_with_mock_supabase, mock_supabase_client_for_ops):
        articles = [{"uniqueName": "batch-1"}, {"uniqueName": "batch-2"}]
        # Configure SupabaseClient's batch method to return a count
        mock_supabase_client_for_ops.store_articles_to_source_articles.return_value = 2

        count = db_manager_with_mock_supabase.store_articles(articles)

        assert count == 2
        mock_supabase_client_for_ops.store_articles_to_source_articles.assert_called_once_with(articles)

    def test_store_articles_with_supabase_exception_returns_zero_and_logs(self, db_manager_with_mock_supabase, mock_supabase_client_for_ops, caplog): # Renamed
        articles = [{"uniqueName": "batch-exc-1"}]
        mock_supabase_client_for_ops.store_articles_to_source_articles.side_effect = Exception("Batch Insert Failed Miserably")

        count = db_manager_with_mock_supabase.store_articles(articles)

        assert count == 0
        assert "Error storing articles batch: Batch Insert Failed Miserably" in caplog.text

    def test_store_articles_console_mode_logs_individually_and_succeeds(self, db_manager_console_mode, caplog): # Renamed
        caplog.set_level(logging.INFO)
        articles = [{"uniqueName": "mock-batch-1"}, {"id": "mock-batch-2", "headline": "HB2"}] # Test with 'id' as well

        # The console mode store_articles calls its own store_article, which logs.
        count = db_manager_console_mode.store_articles(articles)

        assert count == 2 # Processed both
        assert "[MOCK DB] Would store 2 articles" in caplog.text # Overall log
        # Check for individual logs from the internal calls to store_article
        assert "[MOCK DB] Would store article: mock-batch-1" in caplog.text
        assert "[MOCK DB] Would store article: mock-batch-2" in caplog.text # store_article uses uniqueName or id for log

class TestDatabaseManagerGetExistingArticles:
    def test_get_existing_articles_console_mode_logs_and_returns_empty(self, db_manager_console_mode, caplog): # Renamed
        caplog.set_level(logging.INFO)
        # This function is currently a placeholder
        result = db_manager_console_mode.get_existing_articles(limit=50)
        assert result == []
        assert "Would retrieve up to 50 articles" in caplog.text

    def test_get_existing_articles_with_supabase_logs_and_returns_empty(self, db_manager_with_mock_supabase, caplog): # Renamed
        caplog.set_level(logging.INFO)
        # This function is currently a placeholder even when a Supabase client is present
        # TODO: When implemented, add mock for self.supabase_client.table(...).select(...).limit(...).execute()
        result = db_manager_with_mock_supabase.get_existing_articles(limit=75)
        assert result == []
        assert "Would retrieve up to 75 articles" in caplog.text
        # No actual DB call is made by the current implementation, so no Supabase mock interaction needed yet.
