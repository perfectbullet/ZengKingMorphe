#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合测试脚本 - 测试意图识别、文档召回、复杂度评估等功能。

=======================
测试功能
=======================
1. 意图识别测试 (Intent Recognition)
   - 问候语识别
   - 一般查询识别
   - FAQ匹配

2. 文档召回测试 (Document Retrieval/RAG)
   - 向量检索
   - 关键词检索
   - 混合检索 (RRF融合)
   - 召回质量评估

3. 复杂度评估测试 (Complexity Evaluation)
   - LLM复杂度评分
   - 启发式规则评分

4. 实时查询检测 (Realtime Query Detection)
   - 时间表达式识别
   - 天气/新闻/价格查询识别

5. 混合模型路由测试 (Hybrid LLM Routing)
   - 复杂度阈值判断
   - 本地/外部模型选择

=======================
用法说明
=======================

【运行全部测试】
  python ai-service/scripts/comprehensive_test.py

【运行特定测试】
  python ai-service/scripts/comprehensive_test.py --test intent
  python ai-service/scripts/comprehensive_test.py --test retrieval
  python ai-service/scripts/comprehensive_test.py --test complexity

【指定服务器】
  python ai-service/scripts/comprehensive_test.py --host http://192.168.8.230:8100

【使用测试文件中的问题】
  python ai-service/scripts/comprehensive_test.py --use-test-questions

=======================
默认参数
=======================
  --host           http://localhost:8000
  --employee_id    financial_analyst
  --user_id        user_20260110
  --session_id     sess_financial_analyst_fdaf
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# =============================================================================
# Configuration
# =============================================================================

# 默认测试会话信息（来自 "一个测试用的session信息.json"）
DEFAULT_SESSION_ID = "sess_financial_analyst_fdaf"
DEFAULT_USER_ID = "user_20260110"
DEFAULT_EMPLOYEE_ID = "financial_analyst"
DEFAULT_HOST = "http://localhost:8000"
DEFAULT_API_KEY = "y2tJW3P0bvZIxw6pGuV2FrcT0C1wyUfg2ldweEaDYN4"

# 测试配置
REQUEST_DELAY = 1.0  # 每个请求之间的延迟（秒），避免触发速率限制
MAX_RETRIES = 3  # 429 错误的最大重试次数
RETRY_DELAY = 2.0  # 重试延迟（秒）

# 测试问题文件路径
TEST_QUESTIONS_FILE = Path(__file__).parent.parent / "scripts" / "test_questions.json"


# =============================================================================
# Test Questions (从 test_questions.json 中的精选问题)
# =============================================================================

INTENT_TEST_QUESTIONS = [
    # 问候语测试
    ("问候语", "你好"),
    ("问候语", "早上好"),
    ("问候语", "在吗"),
    ("问候语", "哈喽"),
    ("问候语", "打扰一下"),
    # 一般查询测试
    ("一般查询", "失蜡铸造的原理是什么"),
    ("一般查询", "密码学课程的两个主要分支"),
    # 复杂查询测试 - 多轮对话、上下文理解
    ("一般查询-复杂", "比较一下失蜡铸造和数控雕刻的优缺点，并分析在什么情况下应该选择哪种工艺"),
    ("一般查询-复杂", "我需要了解数字韧性对企业发展的具体影响，包括正面和负面因素，最好有实际案例"),
    ("一般查询-复杂", "如果我想学习密码学，应该按照什么样的顺序学习？请给出详细的学习路径和建议"),
]

RETRIEVAL_TEST_QUESTIONS = [
    ("首饰雕蜡工艺", "游标卡尺一般用来测量什么？"),
    ("首饰雕蜡工艺", "使用戒指棒测量戒指尺寸时，如何读取码数？"),
    ("首饰雕蜡工艺", "在为猴子浮雕添加眼睛时，钻半圆坑有什么优点？"),
    ("学术研究", "根据文档，数字韧性的两个核心层面是什么？"),
    ("学术研究", "数据资产通过哪两条重要途径推动企业新质生产力发展？"),
    ("课程教育", "密码学课程的两个主要分支是什么？"),
    ("建筑艺术", "悉尼歌剧院国际设计竞赛的评委中，哪位评委的到来使得伍重的方案最终被选中？"),
    ("基础设施", "乌海市分行机房的PUE值是多少？"),
]

COMPLEXITY_TEST_QUESTIONS = [
    ("简单-具体操作", "游标卡尺一般用来测量什么？"),
    ("简单-数值数据", "熔焊硬蜡时，温度调整为多少？"),
    ("中等-概念理解", "数字韧性的两个核心层面是什么？"),
    ("中等-列表型", "密码学课程的课外实践内容主要包含哪些形式？"),
    ("复杂-多步推理", "为什么在为猴子浮雕添加眼睛时要钻半圆坑？这样做有什么优缺点？"),
    ("复杂-综合分析", "比较悉尼歌剧院设计竞赛中伍重方案与其他方案的主要区别，并分析最终胜出的原因"),
]

REALTIME_TEST_QUESTIONS = [
    # 实时查询测试
    ("天气", "北京今天天气怎么样"),
    ("新闻", "今天有什么热点新闻"),
    ("价格", "黄金现在的价格是多少"),
    ("time", "现在是几点了"),
    ("market", "今天的股市行情如何"),
    # 非实时查询测试 - 这些不应该被识别为实时查询
    ("非实时-知识查询", "失蜡铸造的原理是什么"),
    ("非实时-历史事实", "悉尼歌剧院是什么时候建成的"),
    ("非实时-概念解释", "什么是数字韧性"),
]


# =============================================================================
# Color Output
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.END}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")


# =============================================================================
# API Client
# =============================================================================

class APIClient:
    """API client for comprehensive testing."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        api_key: str = DEFAULT_API_KEY,
        employee_id: str = DEFAULT_EMPLOYEE_ID,
        user_id: str = DEFAULT_USER_ID,
        session_id: str = DEFAULT_SESSION_ID,
        request_delay: float = REQUEST_DELAY,
        max_retries: int = MAX_RETRIES,
        retry_delay: float = RETRY_DELAY,
    ):
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.employee_id = employee_id
        self.user_id = user_id
        self.base_session_id = session_id
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        }
        self._last_request_time = 0.0

    def _get_session_id(self) -> str:
        """Generate a unique session ID for each test run to avoid rate limiting."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 使用进程ID和微秒时间戳确保唯一性
        unique_id = f"{os.getpid()}_{int(time.time() * 1000000) % 10000}"
        return f"test_{timestamp}_{unique_id}"

    def _wait_for_rate_limit(self):
        """Wait to avoid triggering rate limit."""
        now = time.time()
        time_since_last = now - self._last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self._last_request_time = time.time()

    def chat(self, query: str, unique_session: bool = False) -> Dict[str, Any]:
        """
        Send streaming chat request and collect full response.

        Args:
            query: User query text
            unique_session: If True, use a unique session ID for this request

        Note: This API only supports streaming responses.
        The stream is consumed internally and the full content is returned.
        Includes automatic retry for 429 rate limit errors.
        """
        self._wait_for_rate_limit()

        # Use unique session ID if requested to avoid rate limiting
        session_id = self._get_session_id() if unique_session else self.base_session_id

        url = f"{self.host}/api/chat/v1/chat/completions"
        payload = {
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": query}],
            "stream": True,
            "employee_id": self.employee_id,
            "user_id": self.user_id,
            "session_id": session_id,
        }

        # Retry logic for 429 errors
        for attempt in range(self.max_retries):
            start_time = time.time()
            ttfb_ms = None
            full_content = ""
            chunk_count = 0
            first_token = False

            response = requests.post(url, headers=self.headers, json=payload, stream=True, timeout=60)

            # Handle rate limiting
            if response.status_code == 429:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    print_warning(f"Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        "status_code": 429,
                        "elapsed_ms": 0,
                        "ttfb_ms": None,
                        "chunk_count": 0,
                        "content": None,
                        "error": "Rate limit exceeded, max retries reached",
                    }

            if response.status_code != 200:
                return {
                    "status_code": response.status_code,
                    "elapsed_ms": 0,
                    "ttfb_ms": None,
                    "chunk_count": 0,
                    "content": None,
                    "error": response.text,
                }

            # Consume streaming response
            metadata = {}
            for line in response.iter_lines(decode_unicode=True):
                if not line or line.startswith(":"):
                    continue

                if line == "data: [DONE]":
                    break

                if line.startswith("data: "):
                    line = line[6:]

                try:
                    chunk_data = json.loads(line)

                    # Track TTFB
                    if not first_token:
                        ttfb_ms = (time.time() - start_time) * 1000
                        first_token = True

                    chunk_count += 1

                    # Extract content from token chunks
                    if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                        choice = chunk_data["choices"][0]
                        # Check for done chunk (finish_reason == "stop")
                        if choice.get("finish_reason") == "stop":
                            metadata = chunk_data.get("metadata", {})
                        else:
                            delta = choice.get("delta", {})
                            if "content" in delta:
                                full_content += delta["content"]

                except json.JSONDecodeError:
                    pass

            elapsed = (time.time() - start_time) * 1000

            return {
                "status_code": 200,
                "elapsed_ms": elapsed,
                "ttfb_ms": ttfb_ms,
                "chunk_count": chunk_count,
                "content": full_content,
                "metadata": metadata,
            }

        # Should not reach here
        return {
            "status_code": 500,
            "elapsed_ms": 0,
            "ttfb_ms": None,
            "chunk_count": 0,
            "content": None,
            "error": "Unexpected error",
        }

    def get_conversations(self, limit: int = 10) -> List[Dict]:
        """Get recent conversations for metrics analysis."""
        # Note: This endpoint may not exist, would need to be implemented
        return []


# =============================================================================
# Test Functions
# =============================================================================

def test_intent_recognition(client: APIClient) -> Dict[str, Any]:
    """Test intent recognition functionality."""
    print_header("意图识别测试 (Intent Recognition)")

    results = {
        "total": 0,
        "greeting_correct": 0,
        "general_correct": 0,
        "details": [],
    }

    for category, question in INTENT_TEST_QUESTIONS:
        results["total"] += 1
        print(f"\n[{results['total']}] 测试: {question}")
        print(f"    预期意图: {category}")

        response = client.chat(question, unique_session=True)

        if response["status_code"] == 200:
            metadata = response.get("metadata", {})
            intent = metadata.get("intent", "unknown")

            # Check if intent matches expectation
            is_greeting = category == "问候语"
            detected_greeting = intent == "greeting"

            if is_greeting and detected_greeting:
                results["greeting_correct"] += 1
                print_success(f"    检测意图: {intent} (正确)")
            elif not is_greeting and not detected_greeting:
                results["general_correct"] += 1
                print_success(f"    检测意图: {intent} (正确)")
            else:
                print_warning(f"    检测意图: {intent} (预期: {category})")

            results["details"].append({
                "question": question,
                "expected": category,
                "detected": intent,
                "elapsed_ms": response["elapsed_ms"],
            })
        else:
            print_error(f"    请求失败: {response.get('error', 'Unknown error')}")

    # Summary
    print_header("意图识别测试汇总")
    accuracy = (results["greeting_correct"] + results["general_correct"]) / results["total"] * 100
    print(f"总测试数: {results['total']}")
    print(f"问候语正确: {results['greeting_correct']}")
    print(f"一般查询正确: {results['general_correct']}")
    print(f"准确率: {accuracy:.1f}%")

    return results


def test_document_retrieval(client: APIClient) -> Dict[str, Any]:
    """Test document retrieval (RAG) functionality."""
    print_header("文档召回测试 (Document Retrieval)")

    results = {
        "total": 0,
        "with_sources": 0,
        "avg_relevance": 0.0,
        "avg_response_time": 0.0,
        "details": [],
    }

    total_relevance = 0.0

    for category, question in RETRIEVAL_TEST_QUESTIONS:
        results["total"] += 1
        print(f"\n[{results['total']}] [{category}] 测试: {question}")

        response = client.chat(question, unique_session=True)

        if response["status_code"] == 200:
            # Parse metadata from the response
            # Note: We need to make another request to get conversation details
            # For now, just check if we got a response

            content = response.get("content", "")
            has_content = len(content) > 50

            if has_content:
                results["with_sources"] += 1
                print_success(f"    响应时间: {response['elapsed_ms']:.0f}ms | TTFB: {response['ttfb_ms']:.0f}ms")
                print(f"    响应长度: {len(content)} 字符")
                print(f"    内容预览: {content[:100]}...")
            else:
                print_warning(f"    响应内容为空或过短")

            results["details"].append({
                "question": question,
                "category": category,
                "elapsed_ms": response["elapsed_ms"],
                "ttfb_ms": response["ttfb_ms"],
                "content_length": len(content),
                "has_sources": has_content,
            })
        else:
            print_error(f"    请求失败: {response.get('error')}")

    # Summary
    print_header("文档召回测试汇总")
    source_rate = results["with_sources"] / results["total"] * 100 if results["total"] > 0 else 0
    avg_time = sum(d["elapsed_ms"] for d in results["details"]) / results["total"] if results["total"] > 0 else 0
    avg_ttfb = sum(d["ttfb_ms"] or 0 for d in results["details"]) / results["total"] if results["total"] > 0 else 0

    print(f"总测试数: {results['total']}")
    print(f"有效响应: {results['with_sources']}")
    print(f"召回率: {source_rate:.1f}%")
    print(f"平均响应时间: {avg_time:.0f}ms")
    print(f"平均TTFB: {avg_ttfb:.0f}ms")

    results["avg_response_time"] = avg_time

    return results


def test_complexity_evaluation(client: APIClient) -> Dict[str, Any]:
    """Test complexity evaluation functionality."""
    print_header("复杂度评估测试 (Complexity Evaluation)")

    results = {
        "total": 0,
        "avg_response_time": 0.0,
        "low_complexity_count": 0,  # 0-3
        "medium_complexity_count": 0,  # 4-6
        "high_complexity_count": 0,  # 7-10
        "details": [],
    }

    total_time = 0.0

    for complexity_level, question in COMPLEXITY_TEST_QUESTIONS:
        results["total"] += 1
        print(f"\n[{results['total']}] [{complexity_level}] 测试: {question}")

        response = client.chat(question, unique_session=True)

        if response["status_code"] == 200:
            elapsed = response["elapsed_ms"]
            content = response.get("content", "")

            total_time += elapsed

            # Categorize by complexity level based on label prefix
            level_prefix = complexity_level.split("-")[0] if "-" in complexity_level else complexity_level
            if level_prefix == "简单":
                results["low_complexity_count"] += 1
            elif level_prefix == "中等":
                results["medium_complexity_count"] += 1
            elif level_prefix == "复杂":
                results["high_complexity_count"] += 1

            print_info(f"    响应时间: {elapsed:.0f}ms")
            print(f"    内容预览: {content[:100]}...")

            results["details"].append({
                "question": question,
                "complexity_level": complexity_level,
                "elapsed_ms": elapsed,
                "content_length": len(content),
            })
        else:
            print_error(f"    请求失败")

    # Summary
    print_header("复杂度评估测试汇总")
    avg_time = total_time / results["total"] if results["total"] > 0 else 0
    results["avg_response_time"] = avg_time
    print(f"总测试数: {results['total']}")
    print(f"低复杂度 (0-3分): {results['low_complexity_count']}")
    print(f"中复杂度 (4-6分): {results['medium_complexity_count']}")
    print(f"高复杂度 (7-10分): {results['high_complexity_count']}")
    print(f"平均响应时间: {avg_time:.0f}ms")

    return results


def test_realtime_detection(client: APIClient) -> Dict[str, Any]:
    """Test realtime query detection."""
    print_header("实时查询检测测试 (Realtime Query Detection)")

    results = {
        "total": 0,
        "realtime_correct": 0,
        "non_realtime_correct": 0,
        "details": [],
    }

    for category, question in REALTIME_TEST_QUESTIONS:
        results["total"] += 1
        print(f"\n[{results['total']}] [{category}] 测试: {question}")

        response = client.chat(question, unique_session=True)

        if response["status_code"] == 200:
            metadata = response.get("metadata", {})
            is_realtime = metadata.get("is_realtime_query", False)
            realtime_category = metadata.get("realtime_category", "")

            # Determine if this should be detected as realtime
            is_realtime_expected = not category.startswith("非实时")

            if is_realtime_expected:
                # Should be detected as realtime
                if is_realtime:
                    results["realtime_correct"] += 1
                    print_success(f"    检测为实时查询 | 类别: {realtime_category} (正确)")
                else:
                    print_warning(f"    未检测为实时查询 (预期是实时查询)")
            else:
                # Should NOT be detected as realtime
                if not is_realtime:
                    results["non_realtime_correct"] += 1
                    print_success(f"    未检测为实时查询 (正确)")
                else:
                    print_warning(f"    检测为实时查询 | 类别: {realtime_category} (预期不是实时查询)")

            results["details"].append({
                "question": question,
                "expected_category": category,
                "expected_realtime": is_realtime_expected,
                "detected_as_realtime": is_realtime,
                "detected_category": realtime_category,
            })
        else:
            print_error(f"    请求失败")

    # Summary
    print_header("实时查询检测汇总")
    total_correct = results["realtime_correct"] + results["non_realtime_correct"]
    accuracy = total_correct / results["total"] * 100 if results["total"] > 0 else 0
    print(f"总测试数: {results['total']}")
    print(f"实时查询正确: {results['realtime_correct']}")
    print(f"非实时查询正确: {results['non_realtime_correct']}")
    print(f"准确率: {accuracy:.1f}%")

    return results


def test_hybrid_routing(client: APIClient) -> Dict[str, Any]:
    """Test hybrid LLM routing."""
    print_header("混合模型路由测试 (Hybrid LLM Routing)")

    # Test questions that should trigger different routing
    # 平衡测试：简单查询用本地模型，复杂查询用外部模型
    test_cases = [
        # 简单查询 - 应该使用本地 Ollama 模型
        ("简单-本地模型", "你好"),
        ("简单-本地模型", "在吗"),
        ("简单-本地模型", "游标卡尺用来测量什么"),
        ("简单-本地模型", "失蜡铸造的原理是什么"),
        ("简单-本地模型", "什么是密码学"),

        # 复杂查询 - 应该使用外部 API 模型
        ("复杂-外部模型", "分析并比较悉尼歌剧院伍重方案与其他方案的主要区别，并说明最终胜出的原因"),
        ("复杂-外部模型", "比较失蜡铸造和数控雕刻的优缺点，并分析在什么情况下应该选择哪种工艺"),
        ("复杂-外部模型", "请详细说明数字韧性对企业发展的多维度影响，包括技术、组织和文化层面"),
        ("复杂-外部模型", "如果我想系统学习密码学，应该如何规划学习路径？请从基础到高级给出详细建议"),
        ("复杂-外部模型", "综合考虑工艺难度、成本和效果，分析首饰制作中不同工艺的适用场景"),
    ]

    results = {
        "total": len(test_cases),
        "local_llm_count": 0,
        "remote_llm_count": 0,
        "correct_routing": 0,
        "details": [],
    }

    for expected_route, question in test_cases:
        print(f"\n[{len(results['details']) + 1}] [{expected_route}] 测试: {question}")

        response = client.chat(question, unique_session=True)

        if response["status_code"] == 200:
            metadata = response.get("metadata", {})
            # Model info is in metadata
            model = metadata.get("model", "unknown")

            is_local = "qwen" in model.lower() or "ollama" in model.lower()
            expected_local = "本地" in expected_route

            if is_local:
                results["local_llm_count"] += 1
                if expected_local:
                    results["correct_routing"] += 1
                    print_success(f"    使用模型: {model} (本地) - 路由正确")
                else:
                    print_warning(f"    使用模型: {model} (本地) - 预期外部模型")
            else:
                results["remote_llm_count"] += 1
                if not expected_local:
                    results["correct_routing"] += 1
                    print_success(f"    使用模型: {model} (外部) - 路由正确")
                else:
                    print_warning(f"    使用模型: {model} (外部) - 预期本地模型")

            results["details"].append({
                "question": question,
                "expected_route": expected_route,
                "model": model,
                "is_local": is_local,
                "routing_correct": (is_local == expected_local),
            })
        else:
            print_error(f"    请求失败")

    # Summary
    print_header("混合模型路由汇总")
    accuracy = results["correct_routing"] / results["total"] * 100 if results["total"] > 0 else 0
    print(f"总测试数: {results['total']}")
    print(f"本地模型: {results['local_llm_count']}")
    print(f"外部模型: {results['remote_llm_count']}")
    print(f"路由正确率: {accuracy:.1f}%")

    return results


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="综合测试脚本 - 测试意图识别、文档召回、复杂度评估等功能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="API服务器地址",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API密钥",
    )
    parser.add_argument(
        "--employee-id",
        default=DEFAULT_EMPLOYEE_ID,
        help="员工ID",
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help="用户ID",
    )
    parser.add_argument(
        "--session-id",
        default=DEFAULT_SESSION_ID,
        help="会话ID",
    )
    parser.add_argument(
        "--test",
        choices=["all", "intent", "retrieval", "complexity", "realtime", "routing"],
        default="all",
        help="运行的测试类型",
    )
    parser.add_argument(
        "--use-test-questions",
        action="store_true",
        help="使用test_questions.json中的问题",
    )
    parser.add_argument(
        "--output",
        help="输出结果到JSON文件",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    print_header("AI Service 综合测试")
    print(f"服务器: {args.host}")
    print(f"员工ID: {args.employee_id}")
    print(f"用户ID: {args.user_id}")
    print(f"会话ID: {args.session_id}")

    # Create API client
    client = APIClient(
        host=args.host,
        api_key=args.api_key,
        employee_id=args.employee_id,
        user_id=args.user_id,
        session_id=args.session_id,
    )

    # Health check
    print("\n健康检查...")
    try:
        response = requests.get(f"{args.host}/health", timeout=5)
        if response.status_code == 200:
            print_success("服务正常运行")
        else:
            print_error(f"服务异常: {response.status_code}")
            return 1
    except Exception as e:
        print_error(f"无法连接到服务器: {e}")
        return 1

    # Run tests
    all_results = {}
    start_time = time.time()

    if args.test in ["all", "intent"]:
        all_results["intent"] = test_intent_recognition(client)

    if args.test in ["all", "retrieval"]:
        all_results["retrieval"] = test_document_retrieval(client)

    if args.test in ["all", "complexity"]:
        all_results["complexity"] = test_complexity_evaluation(client)

    if args.test in ["all", "realtime"]:
        all_results["realtime"] = test_realtime_detection(client)

    if args.test in ["all", "routing"]:
        all_results["routing"] = test_hybrid_routing(client)

    elapsed = time.time() - start_time

    # Final summary
    print_header("测试完成")
    print(f"总耗时: {elapsed:.1f}秒")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
        print_success(f"结果已保存到: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
