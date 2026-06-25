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

# Refusal keywords — NeMo returns these phrases when it blocks a query.
REFUSAL_KEYWORDS = [
    "i can only answer questions related to",
    "i cannot assist with that request",
    "i must adhere to my safety guidelines",
    "i am sorry, but i can only",
    "i'm sorry, but i can only",
]


def _normalize_response(response) -> str:
    """NeMo Guardrails can return either a str or a dict like {'role': 'assistant', 'content': '...'}."""
    if isinstance(response, dict):
        return response.get("content", str(response))
    return str(response)


def _is_blocked(response) -> bool:
    """Check if a NeMo Guardrails response is a refusal/block."""
    text = _normalize_response(response)
    lowered = text.strip().lower()
    return any(kw in lowered for kw in REFUSAL_KEYWORDS)


def _is_same_content(original: str, guarded: str) -> bool:
    """Check if the guarded response is substantively the same as the original.

    NeMo Guardrails re-generates the entire answer when checking output,
    which strips formatting. We only accept the guarded version if it's
    actually blocking/flagging unsafe content — not if it's just a
    reformulation of the same safe answer.
    """
    if _is_blocked(guarded):
        return False  # genuinely blocked — use the guarded version

    # If the guarded answer is just a re-phrased version of the same content,
    # the original (with formatting) is always better.
    # Heuristic: if the guarded answer is shorter by >40%, it was likely truncated.
    orig_len = len(original.strip())
    guard_len = len(guarded.strip())

    if orig_len > 0 and guard_len / orig_len < 0.6:
        return False  # significant content was lost — use guarded as it may have removed unsafe parts

    return True  # same content — keep the original with its formatting


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
            response = await self.rails.generate_async(prompt=query)
            if _is_blocked(response):
                logger.warning(f"Input blocked by NeMo Guardrails: {query[:80]}")
                response_text = _normalize_response(response)
                return response_text
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
            if _is_blocked(response):
                logger.warning(f"Input blocked by NeMo Guardrails: {query[:80]}")
                response_text = _normalize_response(response)
                return response_text
        except Exception as e:
            logger.error(f"Error executing sync input guardrails: {e}")

        return None

    async def check_output(self, query: str, response_text: str) -> str:
        """
        Verify the bot output against output rails.
        Returns the original response if safe, or a refusal if blocked.

        Key design: NeMo Guardrails re-generates the answer when checking,
        which strips markdown formatting (headings, lists, etc.). We only
        replace the original answer when content is actually unsafe — not
        when the LLM just paraphrases a safe answer.
        """
        if not self.enabled or not self.rails:
            return response_text

        try:
            checked_response = await self.rails.generate_async(
                messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response_text}
                ]
            )
            checked_text = _normalize_response(checked_response)

            # Only replace the original if it was actually blocked/flagged
            if _is_blocked(checked_text):
                logger.warning(f"Output blocked by NeMo Guardrails: {response_text[:80]}")
                return checked_text

            # Safe content — always keep the original with its formatting
            return response_text

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
            checked_text = _normalize_response(checked_response)

            # Only replace the original if it was actually blocked/flagged
            if _is_blocked(checked_text):
                logger.warning(f"Output blocked by NeMo Guardrails: {response_text[:80]}")
                return checked_text

            # Safe content — always keep the original with its formatting
            return response_text

        except Exception as e:
            logger.error(f"Error executing sync output guardrails: {e}")
            return response_text
