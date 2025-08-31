import pytest
import os
from unittest.mock import patch, MagicMock
import asyncio

# Module to test
from llm_selector import LLMSelector, get_available_llm_providers, create_llm_selector

# Test data
MOCK_OPENAI_API_KEY = "sk-openai_mock_key"
# Removed other provider mock keys; only OpenAI supported

@pytest.fixture(autouse=True)
def clear_env_vars():
    # Clear relevant env vars before each test to ensure isolation
    vars_to_clear = ["LLM_PROVIDER", "OPENAI_API_KEY"]
    original_values = {var: os.environ.get(var) for var in vars_to_clear}
    for var in vars_to_clear:
        if var in os.environ:
            del os.environ[var]
    yield
    # Restore original values
    for var, val in original_values.items():
        if val is not None:
            os.environ[var] = val
        elif var in os.environ: # Ensure it's removed if it wasn't there originally
            del os.environ[var]

@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = "OpenAI mock response"
    return client

@pytest.fixture
def mock_gemini_client():
    # google.generativeai.GenerativeModel returns a model instance, then model_instance.generate_content
    model_instance_mock = MagicMock()
    model_instance_mock.generate_content.return_value.text = "Gemini mock response"

    client = MagicMock()
    client.GenerativeModel.return_value = model_instance_mock
    return client

@pytest.fixture
def mock_anthropic_client():
    client = MagicMock()
    # response.content[0].text
    mock_message = MagicMock()
    mock_message.text = "Anthropic mock response"
    client.messages.create.return_value.content = [mock_message]
    return client

@pytest.fixture
def mock_deepseek_client(): # DeepSeek uses OpenAI's client structure
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = "DeepSeek mock response"
    return client


class TestLLMSelectorInitialization:
    def test_default_provider_openai_no_env_key(self):
        """Test that LLMSelector defaults to OpenAI when no API key is set.
            This verifies that the default provider is OpenAI and no client is initialized
            when no API key is provided.
        Args:
            None
        Returns:
            None
        """
        selector = LLMSelector()
        assert selector.provider == LLMSelector.OPENAI
        assert selector.api_key is None
        assert selector.client is None # No key, no client

    def test_default_provider_openai_with_env_key(self, monkeypatch):
        """Test that LLMSelector initializes OpenAI client when API key is set in environment.
            This verifies that the OpenAI client is initialized correctly when the API key is provided
            through an environment variable.
        Args:
            monkeypatch: pytest fixture to set environment variables.
        Returns:
            None    
        """
        monkeypatch.setenv("OPENAI_API_KEY", MOCK_OPENAI_API_KEY)
        with patch('openai.OpenAI') as mock_openai_constructor:
            selector = LLMSelector()
            assert selector.provider == LLMSelector.OPENAI
            assert selector.api_key == MOCK_OPENAI_API_KEY
            mock_openai_constructor.assert_called_once_with(api_key=MOCK_OPENAI_API_KEY)
            assert selector.client is not None

    # Removed tests for unsupported providers (Gemini, Anthropic, DeepSeek, env override)

    def test_unsupported_provider(self, caplog):
        """Test that LLMSelector raises an error for unsupported providers.
            This verifies that the LLMSelector logs an error message and does not initialize a client
            when an unsupported provider is specified.
        Args:
            caplog: pytest fixture to capture log messages.
        Returns:
            None
        """
        selector = LLMSelector(provider="unsupported_provider")
        assert selector.provider == "unsupported_provider"
        assert selector.api_key is None
        assert selector.client is None
        assert "Unsupported LLM provider: unsupported_provider" in caplog.text # This is logged by _get_api_key and _initialize_client
        assert "API key not found for provider unsupported_provider" in caplog.text # from _get_api_key
        # The specific message from _initialize_client for an unknown provider is actually "Unsupported LLM provider: {self.provider}"
        # not "Error initializing LLM client for..."
        # So the first assertion "Unsupported LLM provider: unsupported_provider" covers this.
        # Let's ensure the more generic one isn't there if the specific one is.
        assert "Error initializing LLM client for unsupported_provider" not in caplog.text
        # And confirm the specific one from _initialize_client (which is the same as from constructor)
        assert f"Unsupported LLM provider: {selector.provider}" in caplog.text

    def test_initialization_with_specific_model(self, monkeypatch):
        """Test that LLMSelector can be initialized with a specific model.
            This verifies that the model can be set during initialization and is correctly assigned.
        Args:
            monkeypatch: pytest fixture to set environment variables.
        Returns:
            None
        """
        monkeypatch.setenv("OPENAI_API_KEY", MOCK_OPENAI_API_KEY)
        with patch('openai.OpenAI'):
            selector = LLMSelector(provider=LLMSelector.OPENAI, model="gpt-custom")
            assert selector.model == "gpt-custom"

    def test_api_key_not_found_logs_error(self, caplog):
        """Test that LLMSelector logs an error when API key is not found for OpenAI.
            This verifies that the LLMSelector logs an appropriate error message when the API key
            is not set for OpenAI and does not initialize the client.
        Args:
            caplog: pytest fixture to capture log messages.
        Returns:
            None
        """
        # Test with OpenAI, no key set
        with patch('openai.OpenAI') as mock_openai_constructor:
             selector = LLMSelector(provider=LLMSelector.OPENAI)
             assert selector.api_key is None
             assert "API key not found for provider openai" in caplog.text # from _get_api_key
             assert "OpenAI API key not available" in caplog.text      # from _initialize_openai
             mock_openai_constructor.assert_not_called() # Client should not be initialized
             assert selector.client is None

class TestLLMSelectorInfoAndDefaults:
    def test_get_api_token(self, monkeypatch):
        """Test that LLMSelector retrieves the API token correctly for OpenAI.
            This verifies that the get_api_token method returns the API key set in the environment.
        Args:       
            monkeypatch: pytest fixture to set environment variables.
        Returns:    
            None
        """
        monkeypatch.setenv("OPENAI_API_KEY", MOCK_OPENAI_API_KEY)
        with patch('openai.OpenAI'):
            selector = LLMSelector(provider=LLMSelector.OPENAI)
            assert selector.get_api_token() == MOCK_OPENAI_API_KEY

    def test_get_llm_info_openai_with_key_and_client(self, monkeypatch):
        """Test that LLMSelector returns correct info for OpenAI when API key is set and client is initialized.
            This verifies that the get_llm_info method returns the expected information about the LLM client.   
        Args:
            monkeypatch: pytest fixture to set environment variables.
        Returns:
            None
        """
        monkeypatch.setenv("OPENAI_API_KEY", MOCK_OPENAI_API_KEY)
        mock_client_instance = MagicMock()
        with patch('openai.OpenAI', return_value=mock_client_instance) as mock_constructor:
            selector = LLMSelector(provider=LLMSelector.OPENAI, model="gpt-4")
            info = selector.get_llm_info()
            assert info == {
                "provider": LLMSelector.OPENAI,
                "model": "gpt-4",
                "api_key_available": True,
                "client_initialized": True
            }
            mock_constructor.assert_called_once() # ensure client was attempted to be initialized

    def test_get_llm_info_no_key_no_client(self):
        """Test that LLMSelector returns correct info when no API key is set and client is not initialized.
            This verifies that the get_llm_info method returns the expected information when no API key is available.
        Args:
            None
        Returns:
            None
        """
        selector = LLMSelector(provider=LLMSelector.OPENAI) # No key set
        info = selector.get_llm_info()
        assert info == {
            "provider": LLMSelector.OPENAI,
            "model": LLMSelector.DEFAULT_MODELS[LLMSelector.OPENAI]["chat"], # Default model
            "api_key_available": False,
            "client_initialized": False # No key, so client should not be initialized
        }

    def test_get_default_model(self):
        """Test that LLMSelector returns the default model for a given provider.
            This verifies that the get_default_model method returns the correct default model
            based on the provider specified.
        Args:
            None
        Returns:
            None
        """ 
        selector = LLMSelector(provider=LLMSelector.OPENAI)
        assert selector.get_default_model("chat") == LLMSelector.DEFAULT_MODELS[LLMSelector.OPENAI]["chat"]

    selector_unknown = LLMSelector(provider="unknown")
    assert selector_unknown.get_default_model("chat") == ""


class TestClientInitializationFailures:
    def test_openai_import_error(self, monkeypatch, caplog):
        """Test that LLMSelector handles OpenAI import errors gracefully.
            This verifies that the LLMSelector logs an error message when the OpenAI package cannot be imported
            and does not initialize the client. 
        Args:
            monkeypatch: pytest fixture to set environment variables.
            caplog: pytest fixture to capture log messages. 
        Returns:
            None
        """
        monkeypatch.setenv("OPENAI_API_KEY", MOCK_OPENAI_API_KEY)
        with patch('openai.OpenAI', side_effect=ImportError("Mocked OpenAI import error")):
            selector = LLMSelector(provider=LLMSelector.OPENAI)
            assert selector.client is None
            assert "OpenAI package not installed" in caplog.text # Specific message from _initialize_openai
            assert "Failed to import required libraries for openai" not in caplog.text # Ensure generic isn't there

    # Removed import error tests for other providers

@pytest.mark.asyncio
class TestLLMSelectorGenerateText:
    async def test_generate_text_openai_success(self, monkeypatch, mock_openai_client):
        """Test that LLMSelector can generate text using OpenAI client successfully.    
            This verifies that the generate_text method calls the OpenAI client correctly
            and returns the expected response.
        Args:
            monkeypatch: pytest fixture to set environment variables.
            mock_openai_client: Mocked OpenAI client instance.
        Returns:
            None
        """
        monkeypatch.setenv("OPENAI_API_KEY", MOCK_OPENAI_API_KEY)

        # Simulate that 'openai.OpenAI' returns our mock_openai_client
        with patch('openai.OpenAI', return_value=mock_openai_client), \
             patch('asyncio.to_thread', new_callable=MagicMock) as mock_to_thread:

            # Configure mock_to_thread to immediately return the result of the sync function call
            mock_to_thread.side_effect = lambda func, *args, **kwargs: asyncio.get_event_loop().run_in_executor(None, func, *args) # More realistic to_thread

            selector = LLMSelector(provider=LLMSelector.OPENAI)
            # Ensure the client is our mock
            assert selector.client == mock_openai_client

            prompt = "Test prompt for OpenAI"
            # Since mock_to_thread now returns a future, we await it if it's a coroutine,
            # but our side_effect makes it return the direct result of func.
            # The generate_text method itself is async.

            # Let's make the to_thread mock directly return the result of the sync function for simplicity in test
            async def immediate_sync_executor(func, *args, **kwargs):
                return func(*args, **kwargs)
            mock_to_thread.side_effect = immediate_sync_executor


            response = await selector.generate_text(prompt)

            assert response == "OpenAI mock response"
            # Check that to_thread was called with the client's method
            mock_to_thread.assert_called_once()
            call_args = mock_to_thread.call_args[0] # First argument is the function
            assert call_args[0] == mock_openai_client.chat.completions.create
            # Check some of the params passed to the actual client method
            passed_kwargs = mock_to_thread.call_args[1] # Second argument is kwargs
            assert passed_kwargs['model'] == LLMSelector.DEFAULT_MODELS[LLMSelector.OPENAI]["chat"]
            assert passed_kwargs['messages'] == [{"role": "user", "content": prompt}]

    # Removed generate_text tests for other providers

    async def test_generate_text_no_client(self, caplog):
        """Test that LLMSelector returns None when no client is available.
            This verifies that the generate_text method returns None and logs an error message
            when no LLM client is available (e.g., no API key set).
        Args:
            caplog: pytest fixture to capture log messages.
        Returns:
            None
        """
        selector = LLMSelector(provider=LLMSelector.OPENAI) # No API key, so no client
        assert selector.client is None
        response = await selector.generate_text("A prompt")
        assert response is None
        assert "No LLM client available" in caplog.text

    async def test_generate_text_client_api_error(self, monkeypatch, mock_openai_client, caplog):
        """Test that LLMSelector handles API errors from the client gracefully.
            This verifies that the generate_text method returns None and logs an error message
            when the client raises an exception during text generation.
        Args:
            monkeypatch: pytest fixture to set environment variables.
            mock_openai_client: Mocked OpenAI client instance.
            caplog: pytest fixture to capture log messages.
        Returns:
            None
        """
        monkeypatch.setenv("OPENAI_API_KEY", MOCK_OPENAI_API_KEY)

        mock_openai_client.chat.completions.create.side_effect = Exception("Mock API Error")

        async def immediate_sync_executor(func, *args, **kwargs):
            # This will re-raise the exception from the mocked func
            return func(*args, **kwargs)

        with patch('openai.OpenAI', return_value=mock_openai_client), \
             patch('asyncio.to_thread', new_callable=MagicMock, side_effect=immediate_sync_executor) as mock_to_thread:

            selector = LLMSelector(provider=LLMSelector.OPENAI)
            response = await selector.generate_text("A prompt")

            assert response is None
            assert "Error generating text with openai: Mock API Error" in caplog.text

class TestLLMSelectorUtilityFunctions:
    def test_get_available_llm_providers_none(self):
        """Test that get_available_llm_providers returns an empty list when no API keys are set.
            This verifies that the function correctly identifies when no LLM providers are available.
        Args:
            None
        Returns:
            None
        """
        assert get_available_llm_providers() == []

    def test_get_available_llm_providers_openai_only(self, monkeypatch):
        """Test that get_available_llm_providers returns OpenAI when only its API key is set.
            This verifies that the function correctly identifies OpenAI as the only available provider
            when its API key is set in the environment.
        Args:
            monkeypatch: pytest fixture to set environment variables.
        Returns:
            None
        """
        monkeypatch.setenv("OPENAI_API_KEY", MOCK_OPENAI_API_KEY)
        assert get_available_llm_providers() == [LLMSelector.OPENAI]

    def test_get_available_llm_providers_multiple(self, monkeypatch):
        """Test that get_available_llm_providers returns multiple providers when their API keys are set.
            This verifies that the function correctly identifies all available LLM providers based on the
            API keys set in the environment.
        Args:
            monkeypatch: pytest fixture to set environment variables.
        Returns:
            None
        """
        monkeypatch.setenv("OPENAI_API_KEY", MOCK_OPENAI_API_KEY)
    providers = get_available_llm_providers()  # Only OpenAI considered
    assert providers == [LLMSelector.OPENAI]

    def test_create_llm_selector_default(self, monkeypatch):
        """Test that create_llm_selector creates an LLMSelector with default provider OpenAI.
            This verifies that the create_llm_selector function initializes an LLMSelector with OpenAI
            as the default provider when no provider is specified and the OpenAI API key is set.
        Args:
            monkeypatch: pytest fixture to set environment variables.
        Returns:
            None
        """
        monkeypatch.setenv("OPENAI_API_KEY", MOCK_OPENAI_API_KEY) # Needed for client init
        with patch('openai.OpenAI'):
            selector = create_llm_selector()
            assert isinstance(selector, LLMSelector)
            assert selector.provider == LLMSelector.OPENAI # Default

    # Removed create_llm_selector test for other providers

