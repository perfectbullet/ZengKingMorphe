"""Coordinate concurrent display persistence and HTTP speech streaming."""

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import suppress
from typing import Any

from app.core.logging import get_logger
from app.repositories.stream_chunk_repository import StreamChunkRepository
from app.services.chat.answer_events import AnswerEvent, ChatContext
from app.services.chat.chat_orchestrator import ChatOrchestrator
from app.services.chat.display_pipeline import DisplayPipeline
from app.services.chat.speech_pipeline import SpeechPipeline
from app.services.chat.sse_writer import SSEWriter

logger = get_logger(__name__)

_QUEUE_END = object()
DatabaseProvider = Callable[[], Awaitable[Any]]
SpeechPipelineFactory = Callable[[], SpeechPipeline]


class ChatStreamService:
    """Dispatch one answer stream to Mongo display and SSE speech outputs."""

    def __init__(
        self,
        *,
        orchestrator: ChatOrchestrator | None = None,
        database_provider: DatabaseProvider | None = None,
        speech_pipeline_factory: SpeechPipelineFactory = SpeechPipeline,
    ) -> None:
        self.orchestrator = orchestrator or ChatOrchestrator()
        if database_provider is None:
            from app.core.database import get_database

            database_provider = get_database
        self.database_provider = database_provider
        self.speech_pipeline_factory = speech_pipeline_factory

    async def stream(self, context: ChatContext) -> AsyncGenerator[str, None]:
        writer = SSEWriter(context)
        queue: asyncio.Queue[AnswerEvent | Exception | object] = asyncio.Queue(maxsize=32)
        producer: asyncio.Task | None = None
        display: DisplayPipeline | None = None
        done_metadata: dict[str, Any] = {}

        try:
            db = await self.database_provider()
            repository = StreamChunkRepository(
                db,
                chat_id=context.chat_id,
                session_id=context.session_id,
                user_id=context.user_id,
                employee_id=context.employee_id,
            )
            display = DisplayPipeline(context, repository)
            speech = self.speech_pipeline_factory()
            await display.start()
            yield writer.start()

            async def produce() -> None:
                display_text_parts: list[str] = []
                try:
                    async for event in self.orchestrator.stream(context):
                        await display.handle(event)
                        if event.event_type == "content":
                            display_text_parts.append(event.content)
                        elif event.event_type == "done":
                            event.metadata["display_text"] = "".join(display_text_parts)
                            finish_payload = writer.finish_payload(event.metadata)
                            await display.finish(
                                finish_payload,
                                event.metadata.get("conversation_id", ""),
                            )
                        await queue.put(event)
                except Exception as exc:
                    await queue.put(exc)
                finally:
                    await queue.put(_QUEUE_END)

            producer = asyncio.create_task(produce())

            while True:
                item = await queue.get()
                if item is _QUEUE_END:
                    break
                if isinstance(item, Exception):
                    raise item

                event = item
                if event.event_type == "done":
                    done_metadata = dict(event.metadata)

                async for speech_event in speech.handle(event):
                    yield writer.encode(speech_event)

            await producer
            yield writer.finish(done_metadata)

        except Exception as exc:
            logger.error(
                f"OpenAI stream v2 generation error | error={str(exc)}",
                exc_info=True,
            )
            encoded_error = writer.error(exc)
            if display is not None:
                try:
                    await display.error(
                        json.loads(encoded_error),
                        done_metadata.get("conversation_id", ""),
                    )
                except Exception as db_error:
                    logger.error(
                        f"Failed to save error chunk to DB | error={str(db_error)}",
                        exc_info=True,
                    )
            yield encoded_error
        finally:
            if producer is not None and not producer.done():
                producer.cancel()
                with suppress(asyncio.CancelledError):
                    await producer

        yield "[DONE]"
