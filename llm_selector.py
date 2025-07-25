"""
LLM selector module for choosing and configuring different LLM providers.
"""
import os
import logging
from typing import Dict, Any, Optional, Tuple, List

# Set up logging
logger = logging.getLogger(__name__)

class LLMSelector:
    """
    A class for selecting and configuring different LLM providers.
    Supports OpenAI, Google (Gemini), Anthropic, and DeepSeek.
    """

    # Supported LLM providers
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"

    # Default models for each provider
    DEFAULT_MODELS = {
        OPENAI: {
            "chat": "gpt-4o-mini"
        },
        GEMINI: {
            "chat": "gemini-pro"
        },
        ANTHROPIC: {
            "chat": "claude-3-sonnet-20240229"
        },
        DEEPSEEK: {
            "chat": "deepseek-reasoner"
        }
    }

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLM selector.
        This method sets the provider and model based on the input parameters.
        If no provider is specified, it will default to the environment variable LLM_PROVIDER.

        Args:
            provider: The LLM provider to use (openai, gemini, anthropic, deepseek)
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
        elif self.provider == self.GEMINI:
            self.api_key = os.getenv("GEMINI_API_KEY")
        elif self.provider == self.ANTHROPIC:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        elif self.provider == self.DEEPSEEK:
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
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
            elif self.provider == self.GEMINI:
                self._initialize_gemini()
            elif self.provider == self.ANTHROPIC:
                self._initialize_anthropic()
            elif self.provider == self.DEEPSEEK:
                self._initialize_deepseek()
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

    def _initialize_gemini(self) -> None:
        """Initialize the Gemini (Google) client.
        This method sets up the Gemini client using the API key from the environment variable.
        If the API key is not set, it logs an error.
        Args:
            None
        Returns:
            None
        Raises:
            ImportError: If the Google Generative AI package is not installed
        """
        try:
            import google.generativeai as genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.client = genai
                logger.info("Gemini client initialized successfully")
            else:
                logger.error("Gemini API key not available")
        except ImportError:
            logger.error("Google Generative AI package not installed. Run 'pip install google-generativeai'")

    def _initialize_anthropic(self) -> None:
        """Initialize the Anthropic client.
        This method sets up the Anthropic client using the API key from the environment variable.
        If the API key is not set, it logs an error.
        Args:
            None
        Returns:
            None
        Raises:
            ImportError: If the Anthropic package is not installed
        """
        try:
            import anthropic
            if self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("Anthropic client initialized successfully")
            else:
                logger.error("Anthropic API key not available")
        except ImportError:
            logger.error("Anthropic package not installed. Run 'pip install anthropic'")

    def _initialize_deepseek(self) -> None:
        """Initialize the DeepSeek client.
        This method sets up the DeepSeek client using the OpenAI-compatible API.
        If the API key is not set, it logs an error.
        Args:
            None
        Returns:
            None
        Raises:
            ImportError: If the OpenAI package is not installed
            Exception: If there is an error during client initialization
        """
        try:
            # DeepSeek uses the OpenAI client with a different base URL
            from openai import OpenAI
            if self.api_key:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com/v1"
                )
                logger.info("DeepSeek client initialized successfully")
            else:
                logger.error("DeepSeek API key not available")
        except ImportError:
            logger.error("OpenAI package not installed. Run 'pip install openai'")

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
        If the provider is not recognized, it returns an empty string.
        If the task type is not recognized, it returns an empty string.
        Returns:
            Model name for the specified task
        If the provider is not recognized, it returns an empty string.
        If the task type is not recognized, it returns an empty string.     
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
                # Use asyncio.to_thread for non-blocking OpenAI API calls
                model = self.model or self.get_default_model("chat")
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()

            elif self.provider == self.GEMINI:
                model = self.model or self.get_default_model("chat")
                model_instance = self.client.GenerativeModel(model_name=model)
                response = await asyncio.to_thread(
                    model_instance.generate_content,
                    prompt,
                    generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
                )
                return response.text.strip()

            elif self.provider == self.ANTHROPIC:
                model = self.model or self.get_default_model("chat")
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()

            elif self.provider == self.DEEPSEEK:
                # DeepSeek uses OpenAI-compatible API
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

    if os.getenv("GEMINI_API_KEY"):
        available_providers.append(LLMSelector.GEMINI)

    if os.getenv("ANTHROPIC_API_KEY"):
        available_providers.append(LLMSelector.ANTHROPIC)

    if os.getenv("DEEPSEEK_API_KEY"):
        available_providers.append(LLMSelector.DEEPSEEK)

    return available_providers

def create_llm_selector(provider: Optional[str] = None, model: Optional[str] = None) -> LLMSelector:
    """
    Create an LLM selector instance with the specified provider and model.
    This method initializes the LLMSelector with the given provider and model.
    If no provider is specified, it defaults to the environment variable LLM_PROVIDER. 
    Args:
        provider: The LLM provider to use (openai, gemini, anthropic, deepseek)
        model: Specific model to use
    Returns:
        Initialized LLMSelector instance
    """
    return LLMSelector(provider=provider, model=model)