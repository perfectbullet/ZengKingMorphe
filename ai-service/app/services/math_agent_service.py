"""
数学模型 / 数学 Agent 兼容层。

目标：
1. 保持上层仍可使用 `astream(messages)` / `ainvoke(messages)`。
2. 根据配置动态切换：
   - llm: 直接使用 ChatOpenAI
   - cot: 使用 Qwen-Agent Assistant(function_list=[])
   - tir: 使用 Qwen-Agent TIRMathAgent
3. 将 Qwen-Agent 的“全量流式输出”适配成更接近 OpenAI / LangChain 的增量流式接口。
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.logging import get_logger

logger = get_logger(__name__)

# 兜底最大运行时间（秒）。数学问题长时间思考是正常现象，默认值设得较长，
# 仅用于防止永久挂死；可通过环境变量 MATH_AGENT_MAX_RUNTIME_SECONDS 覆盖。
_DEFAULT_MAX_RUNTIME_SECONDS = 600


def _resolve_max_runtime_seconds() -> int:
    raw = os.getenv("MATH_AGENT_MAX_RUNTIME_SECONDS", str(_DEFAULT_MAX_RUNTIME_SECONDS))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            f"[MathAgentStreamingAdapter] Invalid MATH_AGENT_MAX_RUNTIME_SECONDS={raw!r}, "
            f"fallback to {_DEFAULT_MAX_RUNTIME_SECONDS}"
        )
        return _DEFAULT_MAX_RUNTIME_SECONDS
    return value if value > 0 else _DEFAULT_MAX_RUNTIME_SECONDS


TIR_SYSTEM_EN = (
    "Please integrate natural language reasoning with programs to solve "
    "the problem above, and put your final answer within \\boxed{}."
)
COT_SYSTEM_EN = "Please reason step by step, and put your final answer within \\boxed{}."

TIR_SYSTEM_ZH = "请结合自然语言推理和程序来解决上述问题，并将最终答案放在 \\boxed{} 中。请全程使用中文作答。"
COT_SYSTEM_ZH = "请逐步推理，并将最终答案放在 \\boxed{} 中。请全程使用中文作答。"

SYSTEM_PROMPTS = {
    ("tir", "zh"): TIR_SYSTEM_ZH,
    ("tir", "en"): TIR_SYSTEM_EN,
    ("cot", "zh"): COT_SYSTEM_ZH,
    ("cot", "en"): COT_SYSTEM_EN,
}


@dataclass
class MathRuntimeConfig:
    base_url: str
    model: str
    api_key: str = "dummy-key"
    temperature: float | None = 0.6
    top_p: float | None = 0.95
    max_tokens: int = 10240
    streaming: bool = True


class MathAgentService:
    """动态创建数学 LLM / 数学 Agent 的统一入口。"""

    def __init__(self, config: MathRuntimeConfig):
        self.config = config

    def create_streaming_interface(self, mode: str = "llm", lang: str = "zh"):
        """返回统一流式接口对象。

        返回值始终兼容：
        - `await obj.ainvoke(messages)`
        - `async for chunk in obj.astream(messages): ...`
        """
        if mode == "llm":
            return self.create_chat_openai()
        return MathAgentStreamingAdapter(service=self, mode=mode, lang=lang)

    def create_chat_openai(self) -> ChatOpenAI:
        """创建 OpenAI 兼容数学模型。"""
        kwargs = {
            "base_url": self.config.base_url,
            "api_key": self.config.api_key,
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "streaming": self.config.streaming,
        }
        kwargs.update(self._sampling_kwargs())
        return ChatOpenAI(**kwargs)

    def create_qwen_agent(self, mode: str = "cot", lang: str = "zh"):
        """创建 Qwen-Agent 数学 Agent。"""
        if mode not in {"cot", "tir"}:
            raise ValueError("mode must be 'cot', 'tir', or 'llm'")
        if lang not in {"zh", "en"}:
            raise ValueError("lang must be 'zh' or 'en'")

        from qwen_agent.agents import Assistant, TIRMathAgent

        generate_cfg = {
            "max_tokens": self.config.max_tokens,
        }
        generate_cfg.update(self._sampling_kwargs())

        llm_cfg = {
            "model": self.config.model,
            "model_server": self.config.base_url,
            "api_key": self.config.api_key,
            "generate_cfg": generate_cfg,
        }
        system_message = SYSTEM_PROMPTS[(mode, lang)]

        if mode == "cot":
            return Assistant(
                llm=llm_cfg,
                name=self.config.model,
                system_message=system_message,
                function_list=[],
            )

        return TIRMathAgent(
            llm=llm_cfg,
            name=self.config.model,
            system_message=system_message,
        )

    def uses_model_generation_defaults(self) -> bool:
        """Qwen3-32B 使用模型服务端 generation_config.json 的采样参数。"""
        return self.is_model_generation_default(self.config.model)

    def _sampling_kwargs(self) -> dict[str, float]:
        if self.uses_model_generation_defaults():
            return {}

        kwargs: dict[str, float] = {}
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        return kwargs

    @staticmethod
    def is_model_generation_default(model: str | None) -> bool:
        if not model:
            return False
        normalized = model.lower().replace("_", "-")
        return "qwen3-32b" in normalized

    @staticmethod
    def resolve_lang(query: str | None, fallback: str = "zh") -> str:
        """根据 query 粗略推断语言；未命中时回退 fallback。"""
        text = (query or "").strip()
        if not text:
            return fallback
        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
            return "zh"
        return "en"

    @staticmethod
    def extract_full_text(response) -> str:
        """从 Qwen-Agent response 中提取 assistant 最终文本。"""
        if not response:
            return ""

        if isinstance(response, list):
            for msg in reversed(response):
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(item["text"])
                    return "\n".join(parts)
                return content or ""

        if isinstance(response, dict):
            return response.get("content", "") or ""

        return ""


class MathAgentStreamingAdapter:
    """把 Qwen-Agent 包装成类似 LangChain ChatModel 的接口。"""

    def __init__(self, service: MathAgentService, mode: str = "cot", lang: str = "zh"):
        self.service = service
        self.mode = mode
        self.lang = lang
        self.model_name = service.config.model
        self.base_url = service.config.base_url
        self.openai_api_base = service.config.base_url

    async def astream(self, messages: list[Any]) -> AsyncIterator[AIMessageChunk]:
        """返回增量文本流，兼容上层 `async for chunk in llm.astream(...)`。

        取消/关闭感知设计：
        - 通过 ``stop_event`` 通知后台 worker 线程尽快停止投递。
        - 所有 ``loop.call_soon_threadsafe`` 调用都经由 ``safe_put`` 封装，
          event loop 关闭或调用失败时不再二次异常。
        - async 侧收到 ``CancelledError`` 时设置 ``stop_event`` 并重新抛出。
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()
        stop_event = threading.Event()
        max_runtime = _resolve_max_runtime_seconds()
        start_time = time.monotonic()
        logger.info(
            f"[MathAgentStreamingAdapter] Worker start | mode={self.mode} "
            f"lang={self.lang} max_runtime={max_runtime}s"
        )

        def safe_put(item: tuple[str, str | None]) -> bool:
            """线程安全地向 event loop 投递消息；取消/loop 已关闭时安全跳过。"""
            if stop_event.is_set():
                return False

            if loop.is_closed():
                logger.warning(
                    f"[MathAgentStreamingAdapter] Event loop closed, drop worker message | kind={item[0]}"
                )
                return False

            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
                return True
            except RuntimeError:
                logger.exception(
                    f"[MathAgentStreamingAdapter] Failed to put worker message into event loop queue | kind={item[0]}"
                )
                return False

        def worker():
            previous_text = ""
            try:
                bot = self.service.create_qwen_agent(mode=self.mode, lang=self.lang)
                agent_messages = self._to_agent_messages(messages)
                for response in bot.run(agent_messages):
                    if stop_event.is_set():
                        logger.info(
                            f"[MathAgentStreamingAdapter] Worker cancelled before processing response | mode={self.mode}"
                        )
                        return

                    full_text = self.service.extract_full_text(response)
                    if not full_text:
                        continue

                    if full_text.startswith(previous_text):
                        delta = full_text[len(previous_text):]
                    else:
                        delta = full_text
                    previous_text = full_text

                    if delta:
                        if not safe_put(("chunk", delta)):
                            return

                if not safe_put(("done", None)):
                    return
                logger.info(
                    f"[MathAgentStreamingAdapter] Worker finished | mode={self.mode}"
                )
            except Exception as exc:  # pragma: no cover
                logger.exception(
                    f"[MathAgentStreamingAdapter] Worker failed | mode={self.mode}"
                )
                safe_put(("error", str(exc)))

        thread = threading.Thread(
            target=worker,
            name=f"MathAgentStreamingAdapter-{self.mode}",
            daemon=True,
        )
        thread.start()

        try:
            while True:
                # 用「剩余预算」作为 queue.get() 的等待上限。这样即使 worker 长时间
                # 静默（例如数学题首个 token 之前的长时间思考），也能在超过 max_runtime
                # 后兜底退出；正常产出时 wait_for 会立即返回，不会误判长思考为失败。
                remaining = max_runtime - (time.monotonic() - start_time)
                if remaining <= 0:
                    stop_event.set()
                    logger.error(
                        f"[MathAgentStreamingAdapter] Max runtime exceeded, abort stream | "
                        f"mode={self.mode} max_runtime={max_runtime}s"
                    )
                    raise TimeoutError(
                        f"math agent stream exceeded max runtime {max_runtime}s"
                    )

                try:
                    kind, payload = await asyncio.wait_for(queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    stop_event.set()
                    logger.error(
                        f"[MathAgentStreamingAdapter] Max runtime exceeded, abort stream | "
                        f"mode={self.mode} max_runtime={max_runtime}s"
                    )
                    raise TimeoutError(
                        f"math agent stream exceeded max runtime {max_runtime}s"
                    )

                if kind == "chunk":
                    yield AIMessageChunk(content=payload or "")
                elif kind == "error":
                    raise RuntimeError(payload or "math agent stream failed")
                elif kind == "done":
                    break
        except asyncio.CancelledError:
            stop_event.set()
            logger.info(
                f"[MathAgentStreamingAdapter] Stream cancelled; stop worker delivery | mode={self.mode}"
            )
            raise
        finally:
            stop_event.set()

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        """一次性返回完整答案。"""
        text = await asyncio.to_thread(self._invoke_sync, messages)
        return AIMessage(content=text)

    def _invoke_sync(self, messages: list[Any]) -> str:
        bot = self.service.create_qwen_agent(mode=self.mode, lang=self.lang)
        agent_messages = self._to_agent_messages(messages)
        last_response = None
        for response in bot.run(agent_messages):
            last_response = response
        return self.service.extract_full_text(last_response)

    @staticmethod
    def _to_agent_messages(messages: list[Any]) -> list[dict]:
        """把 LangChain / OpenAI 风格消息转换成 Qwen-Agent 可接受格式。"""
        converted: list[dict] = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            elif isinstance(msg, HumanMessage):
                role = "user"
                content = msg.content
            elif isinstance(msg, SystemMessage):
                role = "system"
                content = msg.content
            elif isinstance(msg, BaseMessage):
                role = getattr(msg, "type", "user")
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                content = msg.content
            else:
                role = "user"
                content = str(msg)
            converted.append({"role": role, "content": content})
        return converted
