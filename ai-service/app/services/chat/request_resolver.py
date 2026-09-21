"""Resolve v2 request identity fields without coupling the endpoint to precedence rules."""

import re

from app.core.logging import get_logger
from app.models.schemas import OpenAIChatRequest
from app.services.chat.answer_events import ChatContext

logger = get_logger(__name__)


class RequestResolver:
    """Apply the legacy body/channel/``extra_body`` precedence rules."""

    def resolve(self, request: OpenAIChatRequest) -> ChatContext:
        extra = request.extra_body or {}
        values = {
            "team_id": request.team_id,
            "user_id": request.user_id,
            "employee_id": request.employee_id,
            "channel_name": request.channel_name,
            "user_name": request.user_name,
            "head_url": request.head_url,
            "session_id": request.session_id,
        }

        extra_channel = extra.get("channel_name")
        if extra_channel and not values["channel_name"]:
            values["channel_name"] = extra_channel
            self._fill_from_channel_name(values, extra_channel)

        # Direct extra_body fields retain the highest priority.
        for field_name in (
            "session_id",
            "team_id",
            "user_id",
            "employee_id",
            "user_name",
            "head_url",
        ):
            if extra.get(field_name):
                values[field_name] = extra[field_name]

        if extra_channel and not values["channel_name"]:
            values["channel_name"] = extra_channel

        resolved_request = request.model_copy(update=values)
        user_query = self._last_user_query(resolved_request)
        user_query = re.sub(r"^[，。！？、；：,.?!;:\s]+", "", user_query).lstrip()
        return ChatContext.from_request(resolved_request, user_query=user_query)

    @staticmethod
    def _fill_from_channel_name(values: dict, channel_name: str) -> None:
        parts = channel_name.split("_")
        if len(parts) < 4 or parts[0] != "employee":
            logger.error(
                f"Invalid channel_name format: {channel_name!r}, expected "
                "'employee_<team_id>_<user_id>_<employee_id>'"
            )
            return

        parsed = {
            "team_id": parts[1],
            "user_id": parts[2],
            "employee_id": parts[3],
            "user_name": parts[4] if len(parts) > 4 else None,
            "head_url": parts[5] if len(parts) > 5 else None,
        }
        for field_name, value in parsed.items():
            if not values.get(field_name) and value:
                values[field_name] = value

    @staticmethod
    def _last_user_query(request: OpenAIChatRequest) -> str:
        for message in reversed(request.messages):
            if message.role == "user":
                return message.content
        return request.messages[-1].content if request.messages else ""

