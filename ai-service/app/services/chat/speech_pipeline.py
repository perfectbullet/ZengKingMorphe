"""Transform answer events into streaming TTS-friendly semantic chunks."""

import re
from collections.abc import AsyncIterator, Awaitable, Callable

from app.services.chat.answer_events import AnswerEvent, SpeechEvent
from app.services.revise_llm import (
    convert_formula_to_voice,
    convert_math_sentence_to_voice,
)
from app.utils.sentence_buffer import SentenceBuffer, has_latex_formula
from app.utils.tts_formatter import convert_markdown_to_voice_text

TextConverter = Callable[[str], Awaitable[str]]


class SpeechPipeline:
    """Buffer semantic units and emit only text suitable for speech synthesis."""

    _MATH_EXPRESSION = re.compile(
        r"(?:[A-Za-z0-9)²³]\s*[=+×÷≤≥≠≈∈∪∩]\s*[A-Za-z0-9(])|[√∞π]"
    )

    def __init__(
        self,
        *,
        formula_converter: TextConverter = convert_formula_to_voice,
        math_sentence_converter: TextConverter = convert_math_sentence_to_voice,
    ) -> None:
        self._buffer = SentenceBuffer(
            max_chars=200,
            max_wait_seconds=0.5,
            comma_split_threshold=80,
        )
        self._formula_converter = formula_converter
        self._math_sentence_converter = math_sentence_converter
        self._sequence = 0
        self._used_teaching_script_tts = False

    async def handle(self, event: AnswerEvent) -> AsyncIterator[SpeechEvent]:
        if event.event_type == "status" and event.content:
            yield self._speech_event(event.content, event.metadata)
            return

        if event.event_type == "content":
            direct_match = event.metadata.get("direct_match") or {}
            teaching_script_tts = direct_match.get("teaching_script_tts")
            if teaching_script_tts:
                if self._used_teaching_script_tts:
                    return
                self._used_teaching_script_tts = True
                async for speech_event in self._flush_pending(event.metadata):
                    yield speech_event
                for segment in self._split_prebuilt_speech(teaching_script_tts):
                    yield self._speech_event(segment, event.metadata)
                return

            segment = self._buffer.add(event.content)
            if segment:
                rendered = await self._render_segment(segment)
                if rendered:
                    yield self._speech_event(rendered, event.metadata)
            return

        if event.event_type == "done":
            async for speech_event in self._flush_pending(event.metadata):
                yield speech_event

    async def _flush_pending(
        self,
        metadata: dict,
    ) -> AsyncIterator[SpeechEvent]:
        remaining = await self._buffer.flush(is_final=True)
        if remaining and remaining.content:
            rendered = await self._render_segment(remaining.content)
            if rendered:
                yield self._speech_event(rendered, metadata)

    async def _render_segment(self, text: str) -> str:
        if has_latex_formula(text):
            text = await self._formula_converter(text)
        elif self._MATH_EXPRESSION.search(text):
            text = await self._math_sentence_converter(text)

        return convert_markdown_to_voice_text(
            text,
            strip_bold=True,
            strip_code=True,
            strip_links=True,
        )

    def _speech_event(self, content: str, metadata: dict) -> SpeechEvent:
        self._sequence += 1
        return SpeechEvent(
            content=content,
            sequence=self._sequence,
            metadata={
                "model_name": metadata.get("model_name"),
                "finish_reason": metadata.get("finish_reason"),
            },
        )

    @staticmethod
    def _split_prebuilt_speech(text: str) -> list[str]:
        parts = re.split(r"([。！？\n])", text)
        return [part for part in SentenceBuffer._merge_punctuation_segments(parts) if part]
