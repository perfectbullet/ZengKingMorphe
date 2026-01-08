"""
Ollama service keep-alive mechanism.
Prevents model from being unloaded by sending periodic requests.
"""
import asyncio
import httpx
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OllamaKeepAlive:
    """
    Background service to keep Ollama models loaded in memory.
    Sends periodic lightweight requests to prevent model unloading.
    """

    def __init__(self):
        """Initialize keep-alive service using settings."""
        from app.core.config import settings

        self.interval = settings.ollama_keep_alive_interval
        self._task: asyncio.Task | None = None
        self._running = False
        self._client: httpx.AsyncClient | None = None

    async def start(self):
        """Start the background keep-alive task."""
        if self._running:
            logger.warning("Ollama keep-alive already running")
            return

        if not settings.use_ollama:
            logger.info("Ollama not enabled, skipping keep-alive")
            return

        if self.interval <= 0:
            logger.info("Ollama keep-alive disabled by configuration (interval=0)")
            return

        self._running = True
        self._client = httpx.AsyncClient(timeout=30.0)
        self._task = asyncio.create_task(self._keep_alive_loop())
        logger.info("Ollama keep-alive service started", interval_seconds=self.interval)

    async def stop(self):
        """Stop the background keep-alive task."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()

        logger.info("Ollama keep-alive service stopped")

    async def _keep_alive_loop(self):
        """Background loop that sends periodic keep-alive requests."""
        try:
            while self._running:
                await asyncio.sleep(self.interval)

                if not self._running:
                    break

                await self._send_keep_alive()
        except asyncio.CancelledError:
            logger.info("Keep-alive loop cancelled")
            raise
        except Exception as e:
            logger.error("Unexpected error in keep-alive loop", error=str(e), exc_info=True)

    async def _send_keep_alive(self):
        """
        Send a lightweight keep-alive request to Ollama.

        Uses /api/tags endpoint which doesn't require model loading.
        """
        try:
            if not self._client:
                return

            url = f"{settings.ollama_base_url}/api/tags"
            response = await self._client.get(url)

            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "unknown") for m in models]
                logger.debug("Ollama keep-alive ping successful", models=model_names)
            else:
                logger.warning(
                    "Ollama keep-alive ping failed",
                    status_code=response.status_code
                )

        except httpx.TimeoutError:
            logger.error("Ollama keep-alive request timed out")
        except Exception as e:
            logger.error("Ollama keep-alive request failed", error=str(e))

    async def force_load_model(self):
        """
        Force load the configured model by sending a minimal completion request.

        Useful for cold start optimization.
        """
        try:
            if not self._client or not settings.use_ollama:
                return

            logger.info("Force loading Ollama model", model=settings.ollama_model)

            url = f"{settings.ollama_base_url}/api/generate"
            payload = {
                "model": settings.ollama_model,
                "prompt": "hi",
                "stream": False,
                "keep_alive": -1  # Tell Ollama to keep model loaded indefinitely
            }

            response = await self._client.post(url, json=payload)
            if response.status_code == 200:
                logger.info("Ollama model force loaded successfully")
            else:
                logger.warning(
                    "Failed to force load Ollama model",
                    status_code=response.status_code
                )

        except Exception as e:
            logger.error("Error force loading Ollama model", error=str(e), exc_info=True)


# Global singleton
ollama_keep_alive = OllamaKeepAlive()
