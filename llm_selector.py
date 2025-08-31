"""
LLM selector module for choosing and configuring different LLM providers.
"""
import os
import logging
from typing import Dict, Any, Optional, Tuple, List

# Set up logging
logger = logging.getLogger(__name__)

class LLMSelector:
    """Simplified selector supporting only OpenAI provider."""

    OPENAI = "openai"

    DEFAULT_MODELS = {
        OPENAI: {"chat": "gpt-5-nano"}
    }

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLM selector.
        This method sets the provider and model based on the input parameters.
        If no provider is specified, it will default to the environment variable LLM_PROVIDER.

        Args:
            provider: The LLM provider to use (only 'openai')
            model: Specific model to use (if not provided, will use default)
        Returns:
            None
        Raises:
            ValueError: If the provider is not supported or API key is not set
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", self.OPENAI)
        self.model = model
        self.client = None
        self.api_key = None

        # Get API key based on provider
        self._get_api_key()
        # Initialize the selected LLM client
        self._initialize_client()

    def _get_api_key(self) -> None:
        """
        Get the appropriate API key based on the selected provider.
        This method checks environment variables for the API key corresponding to the provider.
        If the API key is not found, it logs an error.
        Args:
            None
        Returns:
            None
        Raises:
            ValueError: If the provider is not supported or API key is not set
        """
        if self.provider == self.OPENAI:
            self.api_key = os.getenv("OPENAI_API_KEY")
        else:
            logger.error(f"Unsupported LLM provider: {self.provider}")
            self.api_key = None

        if not self.api_key:
            logger.error(f"API key not found for provider {self.provider}")

    def _initialize_client(self) -> None:
        """
        Initialize the LLM client based on the selected provider.
        This method imports the necessary libraries and sets up the client for the specified provider.
        If the provider is not supported, it logs an error.
        Args:
            None
        Returns:
            None
        Raises:
            ImportError: If the required library for the provider is not installed
            Exception: If there is an error during client initialization
        """
        try:
            if self.provider == self.OPENAI:
                self._initialize_openai()
            else:
                logger.error(f"Unsupported LLM provider: {self.provider}")
        except ImportError as e:
            logger.error(f"Failed to import required libraries for {self.provider}: {e}")
            logger.info("Make sure to install the required packages for the selected provider.")
        except Exception as e:
            logger.error(f"Error initializing LLM client for {self.provider}: {e}")

    def _initialize_openai(self) -> None:
        """Initialize the OpenAI client.
        This method sets up the OpenAI client using the API key from the environment variable.
        If the API key is not set, it logs an error.
        Args:
            None
        Returns:
            None
        Raises:
            ImportError: If the OpenAI package is not installed
        """
        try:
            from openai import OpenAI
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized successfully")
            else:
                logger.error("OpenAI API key not available")
        except ImportError:
            logger.error("OpenAI package not installed. Run 'pip install openai'")

    # Removed other provider initializers (Gemini, Anthropic, DeepSeek)

    def get_api_token(self) -> Optional[str]:
        """
        Get the API token for the current provider.
        This method returns the API key if available, or None if not set.
        Args:
            None
        Returns:
            API token if available, None otherwise  
        """
        return self.api_key

    def get_llm_info(self) -> Dict[str, str]:
        """
        Get information about the current LLM configuration.
        This method returns a dictionary containing the provider, model, and whether the API key is available.
        Args:
            None
        Returns:
            Dictionary containing provider and model information
        """
        model = self.model or self.get_default_model("chat")
        return {
            "provider": self.provider,
            "model": model,
            "api_key_available": self.api_key is not None,
            "client_initialized": self.client is not None
        }

    def get_default_model(self, task_type: str = "chat") -> str:
        """
        Get the default model for the current provider and task.
        This method returns the default model name based on the provider and task type.
        If the provider or task type is not recognized, it returns an empty string.
        Args:
            task_type: Type of task ("chat")
        Returns:
            Model name for the specified task
        """
        if self.provider in self.DEFAULT_MODELS and task_type in self.DEFAULT_MODELS[self.provider]:
            return self.DEFAULT_MODELS[self.provider][task_type]
        return ""

    async def generate_text(self, prompt: str, max_tokens: int = 150, temperature: float = 0.5) -> Optional[str]:
        """
        Generate text using the selected LLM provider.
        This method uses the provider's API to generate text based on the input prompt.
        It supports asynchronous calls for non-blocking operations.
        If the provider is not recognized, it logs an error and returns None.
        Args:
            prompt: The input prompt for the LLM
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0.0-1.0)
        Returns:
            Generated text or None if generation fails
        """
        import asyncio

        if not self.client:
            logger.error("No LLM client available")
            return None

        try:
            if self.provider == self.OPENAI:
                model = self.model or self.get_default_model("chat")
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()

            logger.error(f"Text generation not implemented for provider {self.provider}")
            return None

        except Exception as e:
            logger.error(f"Error generating text with {self.provider}: {e}")
            return None

def get_available_llm_providers() -> List[str]:
    """
    Get a list of all available LLM providers based on environment variables.
    This method checks for the presence of API keys in the environment variables
    and returns a list of providers that have API keys configured.
    Args:
        None
    Returns:
        List of provider names that have API keys configured
    """
    available_providers = []

    if os.getenv("OPENAI_API_KEY"):
        available_providers.append(LLMSelector.OPENAI)

    # Other providers removed

    return available_providers

def create_llm_selector(provider: Optional[str] = None, model: Optional[str] = None) -> LLMSelector:
    """
    Create an LLM selector instance with the specified provider and model.
    This method initializes the LLMSelector with the given provider and model.
    If no provider is specified, it defaults to the environment variable LLM_PROVIDER. 
    Args:
    provider: The LLM provider to use (only 'openai')
        model: Specific model to use
    Returns:
        Initialized LLMSelector instance
    """
    return LLMSelector(provider=provider, model=model)