"""MongoDB persistence for frontend display stream chunks."""

from datetime import datetime
from typing import Any, Optional

from app.models.database import StreamChunkModel


class StreamChunkRepository:
    """Persist legacy-compatible ``stream_chunks`` documents with ordered IDs."""

    def __init__(
        self,
        db: Any,
        *,
        chat_id: str,
        session_id: str,
        user_id: str,
        employee_id: str,
    ) -> None:
        self._collection = db.stream_chunks
        self.chat_id = chat_id
        self.session_id = session_id
        self.user_id = user_id
        self.employee_id = employee_id
        self.sequence = 0

    async def save(
        self,
        *,
        chunk_type: str,
        chunk_data: dict[str, Any],
        conversation_id: Optional[str] = None,
    ) -> StreamChunkModel:
        self.sequence += 1
        now = datetime.utcnow()
        chunk = StreamChunkModel(
            chunk_id=f"{self.chat_id}_chunk_{self.sequence}",
            conversation_id=conversation_id,
            session_id=self.session_id,
            user_id=self.user_id,
            employee_id=self.employee_id,
            chat_id=self.chat_id,
            chunk_type=chunk_type,
            chunk_data=chunk_data,
            sequence=self.sequence,
            timestamp=now,
            created_at=now,
        )
        await self._collection.insert_one(chunk.model_dump())
        return chunk

