"""LLM service connecting to NVIDIA Nemotron via OpenAI-compatible SDK."""
import time
import structlog
try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - optional dependency
    AsyncOpenAI = None
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class LLMService:
    """Wrapper around NVIDIA's OpenAI-compatible API for Nemotron."""

    def __init__(self):
        self.model = settings.nvidia_model
        if not settings.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set. Add it to backend/.env before calling the Nemotron API.")
        if AsyncOpenAI is None:
            raise RuntimeError("The openai package is unavailable. Install backend dependencies before using NVIDIA Nemotron.")
        self.client = AsyncOpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.6,
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> dict:
        """Send a chat completion request to NVIDIA Nemotron."""
        if self.client is None:
            raise RuntimeError("LLM client unavailable. Install the required runtime dependencies.")
        start = time.monotonic()
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            latency_ms = (time.monotonic() - start) * 1000
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else None
            logger.info(
                "llm_response",
                model=self.model,
                latency_ms=round(latency_ms, 1),
                tokens=tokens,
                chars=len(content),
            )
            return {
                "content": content,
                "tokens_used": tokens,
                "latency_ms": latency_ms,
                "model": self.model,
            }
        except Exception as e:
            logger.error(
                "llm_error",
                error_type=type(e).__name__,
                status_code=getattr(e, "status_code", None),
                model=self.model,
            )
            raise

    async def stream_chat(self, messages: list[dict], temperature: float = 0.6, max_tokens: int = 1024):
        """Stream chat completions — yields content chunks."""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    async def extract_structured(self, prompt: str, system: str = "") -> str:
        """Helper for structured extraction tasks."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = await self.chat(messages, temperature=0.1, max_tokens=4096)
        return result["content"]


_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
