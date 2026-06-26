"""
数学模型调用调试 dump 工具。

当请求进入数学模型分支（``streaming_type == "math_llm"``）时，把本次数学模型
调用的输入 / 输出 / 异常 / 耗时保存为本地 JSON 文件，便于排查数学模型输入
prompt、当前问题、模型输出全文、异常和耗时。

设计原则：
1. 每次数学模型调用保存一个 JSON 文件，按时间命名；
2. 先保存输入，状态为 ``running``；正常结束后更新为 ``completed``；
   异常时为 ``error``；取消时为 ``cancelled``；
3. 所有写操作都是 ``safe`` 的——保存失败只打日志，绝不影响数学回答主流程。
"""

import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# 默认调试文件保存目录（相对于服务运行时的工作目录，通常是 ai-service/）
DEFAULT_MATH_DEBUG_DIR = "logs/math_model_debug"


def is_math_debug_dump_enabled() -> bool:
    """是否启用数学模型调试 dump（默认启用）。"""
    value = os.getenv("MATH_DEBUG_DUMP_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def get_math_debug_dump_dir() -> Path:
    """获取调试文件保存目录，支持用 MATH_DEBUG_DUMP_DIR 覆盖。"""
    path = os.getenv("MATH_DEBUG_DUMP_DIR", DEFAULT_MATH_DEBUG_DIR).strip()
    return Path(path)


def _safe_filename_part(value: Any, max_len: int = 64) -> str:
    """把任意值转换为安全的文件名片段（仅保留字母数字与 -_）。"""
    text = str(value or "unknown")
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    result = "".join(safe).strip("_")
    return (result or "unknown")[:max_len]


def serialize_message(message: Any) -> dict[str, Any]:
    """序列化 LangChain message 或 dict message。"""
    if isinstance(message, dict):
        return {
            "type": message.get("type") or message.get("role") or "dict",
            "role": message.get("role"),
            "content": message.get("content"),
        }

    return {
        "type": message.__class__.__name__,
        "role": getattr(message, "type", None),
        "content": getattr(message, "content", None),
    }


def serialize_messages(messages: list[Any] | None) -> list[dict[str, Any]]:
    """批量序列化 messages。"""
    return [serialize_message(msg) for msg in (messages or [])]


def create_math_debug_file(
    *,
    user_id: Any,
    employee_id: Any,
    session_id: Any,
    chat_id: Any,
    runtime_mode: Any,
) -> Path:
    """生成调试 JSON 文件路径（按时间 + 关键标识命名），并确保目录存在。"""
    dump_dir = get_math_debug_dump_dir()
    dump_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]

    filename = (
        f"{timestamp}"
        f"_user-{_safe_filename_part(user_id)}"
        f"_emp-{_safe_filename_part(employee_id)}"
        f"_sess-{_safe_filename_part(session_id, 40)}"
        f"_mode-{_safe_filename_part(runtime_mode)}"
        f"_chat-{_safe_filename_part(chat_id, 32)}"
        f".json"
    )

    return dump_dir / filename


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """原子写 JSON：先写 .tmp 再 replace，避免半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    tmp_path.replace(path)


def safe_write_math_debug(path: Optional[Path], data: dict[str, Any]) -> None:
    """安全写调试 JSON：失败只记日志，绝不抛出。"""
    if not path:
        return

    try:
        atomic_write_json(path, data)
    except Exception:
        logger.exception(f"[MathDebugDump] Failed to write debug json | path={path}")


def build_base_debug_payload(
    *,
    status: str,
    user_id: Any,
    employee_id: Any,
    session_id: Any,
    chat_id: Any,
    runtime_mode: Any,
    model_name: Any,
    base_url: Any,
    user_query: str,
    messages: list[Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """构建初始调试 payload（状态 running，输出/耗时为空待后续填充）。"""
    now = datetime.now().isoformat(timespec="milliseconds")

    return {
        "status": status,
        "created_at": now,
        "updated_at": now,
        "request": {
            "user_id": user_id,
            "employee_id": employee_id,
            "session_id": session_id,
            "chat_id": chat_id,
        },
        "math_runtime": {
            "mode": runtime_mode,
            "model_name": model_name,
            "base_url": base_url,
        },
        "query": {
            "user_query": user_query,
            "classification_label": state.get("classification_label"),
            "classification_confidence": state.get("classification_confidence"),
            "classification_reason": state.get("classification_reason"),
            "answer_mode": state.get("answer_mode"),
            "is_math_problem": state.get("is_math_problem"),
        },
        "input": {
            "messages": serialize_messages(messages),
            "messages_count": len(messages or []),
        },
        "output": {
            "content": "",
            "output_chars": 0,
        },
        "timing": {
            "started_at": now,
            "finished_at": None,
            "duration_ms": None,
        },
        "error": None,
    }


def mark_completed(
    payload: dict[str, Any],
    *,
    output_content: str,
    duration_ms: int | None,
) -> dict[str, Any]:
    """把 payload 标记为 completed，填充输出与耗时。"""
    now = datetime.now().isoformat(timespec="milliseconds")
    payload["status"] = "completed"
    payload["updated_at"] = now
    payload["output"] = {
        "content": output_content,
        "output_chars": len(output_content or ""),
    }
    payload["timing"]["finished_at"] = now
    payload["timing"]["duration_ms"] = duration_ms
    payload["error"] = None
    return payload


def mark_error(
    payload: dict[str, Any],
    *,
    exc: BaseException,
    output_content: str,
    duration_ms: int | None,
) -> dict[str, Any]:
    """把 payload 标记为 error，保存异常类型/消息/traceback。

    必须在 except 块内调用，traceback.format_exc() 才能取到当前异常堆栈。
    """
    now = datetime.now().isoformat(timespec="milliseconds")
    payload["status"] = "error"
    payload["updated_at"] = now
    payload["output"] = {
        "content": output_content,
        "output_chars": len(output_content or ""),
    }
    payload["timing"]["finished_at"] = now
    payload["timing"]["duration_ms"] = duration_ms
    payload["error"] = {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    return payload


def mark_cancelled(
    payload: dict[str, Any],
    *,
    output_content: str,
    duration_ms: int | None,
) -> dict[str, Any]:
    """把 payload 标记为 cancelled（客户端断开 / 任务取消）。"""
    now = datetime.now().isoformat(timespec="milliseconds")
    payload["status"] = "cancelled"
    payload["updated_at"] = now
    payload["output"] = {
        "content": output_content,
        "output_chars": len(output_content or ""),
    }
    payload["timing"]["finished_at"] = now
    payload["timing"]["duration_ms"] = duration_ms
    payload["error"] = {
        "type": "CancelledError",
        "message": "Math streaming was cancelled",
    }
    return payload
