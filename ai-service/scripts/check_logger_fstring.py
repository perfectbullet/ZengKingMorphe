#!/usr/bin/env python3
"""Check that logger.* calls do not use parameterized message arguments.

All ``logger.*(...)`` calls in this repository MUST use f-strings.
This script flags any call that passes positional message arguments
(e.g. ``logger.info("x=%s", x)`` or ``logger.info("x={}", x)``), including
``logger.bind(...).info(...)`` style calls. It only inspects the first
positional argument and the call arity — it does not execute any code.

Run::

    cd ai-service
    python scripts/check_logger_fstring.py

Exit code is 0 when no violations are found, 1 otherwise.
"""

import ast
import sys
from pathlib import Path

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    "dist", "build", ".pytest_cache", ".mypy_cache"
}

LOGGER_METHODS = {
    "trace", "debug", "info", "success", "warning", "warn",
    "error", "exception", "critical"
}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def _tail_name(node: ast.AST) -> str | None:
    """Return the attribute name at the end of a ``foo.bar.baz`` style access."""
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _root_is_logger(node: ast.AST) -> bool:
    """Walk down func.value chain (e.g. logger.bind(x=1).info) → True if root is ``logger``."""
    cur = node
    # Drill through Attribute.value / Call.func to find the base Name.
    while True:
        if isinstance(cur, ast.Call):
            cur = cur.func
            continue
        if isinstance(cur, ast.Attribute):
            cur = cur.value
            continue
        break
    return isinstance(cur, ast.Name) and cur.id == "logger"


def is_logger_call(node: ast.Call) -> bool:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr not in LOGGER_METHODS:
        return False
    # Either ``logger.<method>`` directly, or ``logger.<chain>.<method>``
    # (e.g. ``logger.bind(...).<method>``). Also accept generic attribute roots
    # so chained calls like ``logger.bind(x).info(...)`` are caught.
    if isinstance(func.value, ast.Name) and func.value.id == "logger":
        return True
    return _root_is_logger(func.value)


def main() -> int:
    violations = []

    for path in sorted(Path(".").rglob("*.py")):
        if should_skip(path):
            continue

        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except SyntaxError as exc:
            violations.append(f"{path}: parse failed: {exc}")
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not is_logger_call(node):
                continue

            # Parameterized: >= 2 positional args, first is a plain string constant.
            if (
                len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                violations.append(
                    f"{path}:{node.lineno}:{node.col_offset} "
                    "logger call uses parameterized arguments; use an f-string"
                )

    if violations:
        print("Logger f-string check failed:", file=sys.stderr)
        for item in violations:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print("Logger f-string check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
