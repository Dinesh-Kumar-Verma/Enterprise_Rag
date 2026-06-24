import os

# NeMo Guardrails needs LangChain framework for non-native providers like Groq
os.environ["NEMOGUARDRAILS_LLM_FRAMEWORK"] = "langchain"

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

        # NeMo Guardrails expects GROQ_API_KEY for the Groq engine
        if "GROQ_API_KEY" not in os.environ and settings.groq_api_key:
            os.environ["GROQ_API_KEY"] = settings.groq_api_key

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
