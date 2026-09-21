"""Protocol objects between answer generation and output delivery."""

from dataclasses import dataclass, field
from typing import Any, Optional

from app.models.schemas import OpenAIChatRequest


@dataclass(slots=True)
class ChatContext:
    """Resolved request identity and immutable stream metadata."""

    request: OpenAIChatRequest
    user_query: str
    session_id: str
    chat_id: str
    created: int
    user_id: str
    employee_id: str
    team_id: Optional[str] = None
    channel_name: Optional[str] = None
    user_name: Optional[str] = None
    head_url: Optional[str] = None

    @classmethod
    def from_request(
        cls,
        request: OpenAIChatRequest,
        *,
        user_query: str,
        session_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        created: Optional[int] = None,
    ) -> "ChatContext":
        import hashlib
        import time

        resolved_session_id = session_id or request.session_id
        if not resolved_session_id:
            seed = f"{request.user_id}_{time.time()}"
            digest = hashlib.md5(seed.encode()).hexdigest()[:12]
            resolved_session_id = f"sess_{digest}"

        resolved_created = created if created is not None else int(time.time())
        resolved_chat_id = chat_id
        if not resolved_chat_id:
            seed = f"{resolved_session_id}_{time.time()}"
            digest = hashlib.md5(seed.encode()).hexdigest()[:12]
            resolved_chat_id = f"chatcmpl-{digest}"

        return cls(
            request=request,
            user_query=user_query,
            session_id=resolved_session_id,
            chat_id=resolved_chat_id,
            created=resolved_created,
            user_id=request.user_id,
            employee_id=request.employee_id,
            team_id=request.team_id,
            channel_name=request.channel_name,
            user_name=request.user_name,
            head_url=request.head_url,
        )


@dataclass(slots=True)
class AnswerEvent:
    """A protocol-neutral fact emitted while an answer is being produced."""

    event_type: str
    content: str = ""
    content_type: str = "text"
    sequence: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpeechEvent:
    """A TTS-ready text fragment, independent of HTTP serialization."""

    event_type: str = "content"
    content: str = ""
    sequence: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnswerResult:
    """Final answer aggregate used for metadata and conversation persistence."""

    display_text: str = ""
    speech_text: Optional[str] = None
    answer_mode: str = ""
    model_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

