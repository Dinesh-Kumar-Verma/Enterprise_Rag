import os
# Set LLM framework to LangChain to support google engine seamlessly
os.environ["NEMOGUARDRAILS_LLM_FRAMEWORK"] = "langchain"

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Monkeypatch ChatGoogleGenerativeAI to translate max_tokens to max_output_tokens
    # preventing Pydantic ValidationError in newer versions of langchain-google-genai.
    original_prepare_request = ChatGoogleGenerativeAI._prepare_request

    def patched_prepare_request(self, messages, stop=None, generation_config=None, **kwargs):
        if "max_tokens" in kwargs:
            max_tokens = kwargs.pop("max_tokens")
            if "max_output_tokens" not in kwargs:
                kwargs["max_output_tokens"] = max_tokens
        if generation_config and "max_tokens" in generation_config:
            max_tokens = generation_config.pop("max_tokens")
            if "max_output_tokens" not in generation_config:
                generation_config["max_output_tokens"] = max_tokens
        return original_prepare_request(self, messages, stop=stop, generation_config=generation_config, **kwargs)

    ChatGoogleGenerativeAI._prepare_request = patched_prepare_request
except ImportError:
    pass

from pathlib import Path
from loguru import logger
from config.settings import get_settings

try:
    from nemoguardrails import RailsConfig, LLMRails
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

settings = get_settings()

class RAGGuardrails:
    def __init__(self):
        self.enabled = False
        self.rails = None

        if not NEMO_AVAILABLE:
            logger.warning("nemoguardrails is not installed. Guardrails will be disabled. Run `pip install nemoguardrails` to enable.")
            return

        # Set LLM framework to LangChain to support google engine seamlessly
        os.environ["NEMOGUARDRAILS_LLM_FRAMEWORK"] = "langchain"

        # NeMo Guardrails expects GOOGLE_API_KEY for the Gemini engine
        if "GOOGLE_API_KEY" not in os.environ and settings.gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key

        config_dir = Path(__file__).parents[2] / "config" / "guardrails"
        
        if config_dir.exists():
            try:
                logger.info(f"Loading NeMo Guardrails configuration from: {config_dir}")
                config = RailsConfig.from_path(str(config_dir))
                self.rails = LLMRails(config)
                self.enabled = True
            except Exception as e:
                logger.error(f"Failed to load NeMo Guardrails: {e}")
        else:
            logger.warning(f"Guardrails config directory not found at {config_dir}. Guardrails are disabled.")

    async def check_input(self, query: str) -> str | None:
        """
        Verify the user input against input rails.
        Returns a refusal message if blocked, or None if the query is safe.
        """
        if not self.enabled or not self.rails:
            return None
            
        try:
            # Generate response via guardrails logic
            response = await self.rails.generate_async(prompt=query)
            # If NeMo Guardrails triggers a block flow (e.g. refuse off topic or jailbreak),
            # it returns the pre-defined bot refusal string instead of letting the query proceed.
            if response in [
                "I am sorry, but I can only answer questions related to the enterprise knowledge base.",
                "I cannot assist with that request. I must adhere to my safety guidelines."
            ]:
                return response
        except Exception as e:
            logger.error(f"Error executing input guardrails: {e}")
            
        return None

    def check_input_sync(self, query: str) -> str | None:
        """
        Synchronously verify user input against input rails.
        """
        if not self.enabled or not self.rails:
            return None
            
        try:
            response = self.rails.generate(prompt=query)
            if response in [
                "I am sorry, but I can only answer questions related to the enterprise knowledge base.",
                "I cannot assist with that request. I must adhere to my safety guidelines."
            ]:
                return response
        except Exception as e:
            logger.error(f"Error executing sync input guardrails: {e}")
            
        return None

    async def check_output(self, query: str, response_text: str) -> str:
        """
        Verify the bot output against output rails.
        """
        if not self.enabled or not self.rails:
            return response_text

        try:
            # Verify output content safety
            checked_response = await self.rails.generate_async(
                messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response_text}
                ]
            )
            return checked_response
        except Exception as e:
            logger.error(f"Error executing output guardrails: {e}")
            return response_text

    def check_output_sync(self, query: str, response_text: str) -> str:
        """
        Synchronously verify the bot output against output rails.
        """
        if not self.enabled or not self.rails:
            return response_text

        try:
            checked_response = self.rails.generate(
                messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response_text}
                ]
            )
            return checked_response
        except Exception as e:
            logger.error(f"Error executing sync output guardrails: {e}")
            return response_text

