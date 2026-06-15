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
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


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
        """返回增量文本流，兼容上层 `async for chunk in llm.astream(...)`。"""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()

        def worker():
            previous_text = ""
            try:
                bot = self.service.create_qwen_agent(mode=self.mode, lang=self.lang)
                agent_messages = self._to_agent_messages(messages)
                for response in bot.run(agent_messages):
                    full_text = self.service.extract_full_text(response)
                    if not full_text:
                        continue

                    if full_text.startswith(previous_text):
                        delta = full_text[len(previous_text):]
                    else:
                        delta = full_text
                    previous_text = full_text

                    if delta:
                        loop.call_soon_threadsafe(queue.put_nowait, ("chunk", delta))

                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as exc:  # pragma: no cover
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

        while True:
            event, payload = await queue.get()
            if event == "chunk":
                yield AIMessageChunk(content=payload or "")
            elif event == "error":
                raise RuntimeError(payload or "math agent stream failed")
            elif event == "done":
                break

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
