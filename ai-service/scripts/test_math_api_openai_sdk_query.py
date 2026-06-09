#!/usr/bin/env python3
"""
数学模型 API 测试脚本（OpenAI SDK版本）
测试远程服务器上的数学问题解答能力，支持流式输出，并将结果保存为 Markdown 文件
单独查询版本
"""

import json
import os
import time
from datetime import datetime
import re
from openai import OpenAI
from prompts.prompts import MATH_TEACHER_SYSTEM_PROMPT
# 远程服务器配置
SERVER = "192.168.8.233"
PORT = 8000
BASE_URL = f"http://{SERVER}:{PORT}/v1"

# 输出目录
OUTPUT_DIR = "math_responses_openai_sdk"

# 颜色输出
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
NC = "\033[0m"


def create_client():
    """创建 OpenAI 客户端"""
    return OpenAI(
        base_url=BASE_URL,
        api_key="dummy-key",  # vLLM 不需要真实的 API key
    )


def test_models(client):
    """测试模型列表端点（使用 OpenAI SDK）"""
    print(f"\n{BLUE}[*] 测试模型列表...{NC}")
    try:
        models = client.models.list()
        print(f"{GREEN}[+] 模型列表获取成功{NC}")
        for model in models.data:
            print(f"    - ID: {model.id}")
            print(f"      拥有者: {model.owned_by}")
        return True, models.data
    except Exception as e:
        print(f"{RED}[-] 模型列表获取失败: {e}{NC}")
        return False, []


def test_chat_completion_stream(
    client,
    question,
    model_id="/home/phi-4-mini-reasoning",
    temperature=0.6,
    max_tokens=4000,
):
    """测试聊天完成端点（流式输出）"""

    # SYSTEM_PROMPT = (
    #     "你是数学推理模型，一个专门从事数学和逻辑推理的强大 AI 模型。"
    #     "请一步步思考，并提供清晰、合理的答案。"
    # )

    SYSTEM_PROMPT = """你是一名数学老师。温暖鼓励型数学教师，面向初高中学生。举例减少需图形理解的例子。比喻贴近本质，举例≤2个（1知识+1题目）  
**开头肯定**：说出学生做对了什么（如“你能列出条件，思路很清楚”）。  
**节奏**：关键步骤前说“慢一点”、“重点来了”。    
**计算**：方程化简、代入等必须完整写出每步，不跳过。  
**易错点**：指出错误后说明后果（如“不检验会多出无效解”）。  
**过渡语**：讲到具体题目时，不说“例题”，要说“让我们来看一道题目”或“我们一起看个例子”等。

## 回答要点
单次回答(包含所有解释、推导和示例),总字数不得超过800字。
推导过程请只选择最关键的前3步进行说明。
每次回答问题,最多进行2轮额外的知识检索。
每次检索时,获取的相关实体不超过10个,关系链不超过3跳。
如果检索到的信息已经能够回答问题，立即停止检索并开始组织答案，不要试图穷尽所有相关信息。

## 任务

先判断“知识点”还是“题目”。
**知识点讲解（4段）**  

1. 一句话定义（可带比喻）+肯定。用生活例子或旧知识引出。  
2. 一个数字例子，引导参与（“你试试看”）。必须解释规律背后的定义或组合意义。  
3. 一个简单题目，用“让我们来看一道题目”引出，分步讲，计算步骤完整，包括组合数具体计算。  
4. 常见错误（原因+后果+检验方法）+小结+变式（先易后难）+鼓励。
   **题目讲解（5段）**  
5. 点明知识点+肯定。问“条件是什么？要求什么？”  
6. 一句话讲明白知识点
7. 给出最优解题思路，从定义出发解释“为什么这么做”。  
8. 分步求解：先/然后/最后。关键步放慢重复，穿插“你猜下一步？”解方程、代入等计算完整展示。  
9. 易错点（错误+原因+后果+检验）+变式思考（由浅入深）+鼓励收尾。

## 通用要求

口语化，少“嗯”“啊”。段落内用“首先/其次/最后”。体现“不着急”、“一步步来”。  
启发语示例：“你看是不是这样？”“试试看，如果…会怎样？”“你想一想，这个条件告诉我们什么？”  
禁止：图形描述、通篇举例、生硬序号、内部思考、跳过计算、空泛肯定（如“你很好”不说明原因）。  
输出纯口语，全文工整，保留空行，每段≤4行，末尾必有变式题(由浅入深),禁止大段文字。

## 示例（知识点：“什么是函数”）

你能主动问这个概念，特别好。函数就像自动售货机——按一个按钮只掉一种饮料。每个输入对应唯一输出。  

数字例子：y=x+1，输入2得3，输入5得6。你试试输入10？对，11。  

让我们来看一道题目：已知f(x)=x²，求f(3)和f(-3)。代入：3²=9，(-3)²=9。两个不同输入得相同输出，仍是函数。很多同学以为必须一一对应，你提前知道就不会错。  

小结：函数不能一对多。变式：如果f(1)有时等于2有时等于3，还是函数吗？继续努力你会越来越棒。

## 输出

直接输出讲解文本，每段后空一行，每段≤4行。不要额外说明。"""



    messages = [
        {"role": "system", "content": MATH_TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    try:
        # 记录开始时间
        start_time = time.time()

        # 创建流式响应
        stream = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens * 2,
            stream=True,  # 启用流式输出
            top_p=0.95,
            extra_body={
                # "repetition_penalty": 1.2,  # vLLM 特有参数
                # "reasoning_budget": 2048,
                # "reasoning_budget_message": "现在让我停止思考并回答问题。",  # 可选
            },
        )

        # 流式接收和显示响应
        full_response = ""
        thinking_tokens = 0
        answer_tokens = 0
        in_thinking = False
        think_start_time = None
        think_time = 0.0
        ttfb = None
        tag_buffer = ""  # 用于跨 chunk 检测 <think 标签
        print(f"\n{GREEN}[+] 开始接收流式响应:{NC}")

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                now = time.time()

                # TTFB：首字节时间
                if ttfb is None:
                    ttfb = now - start_time

                # 用 buffer 累积检测 <think / </think 标签（可能跨 chunk 拆分）
                tag_buffer += content
                if not in_thinking:
                    if "<think" in tag_buffer:
                        in_thinking = True
                        think_start_time = now
                        tag_buffer = ""
                else:
                    if "</think" in tag_buffer:
                        in_thinking = False
                        if think_start_time:
                            think_time += now - think_start_time
                        tag_buffer = ""
                    elif len(tag_buffer) > 50:
                        tag_buffer = tag_buffer[-20:]  # 只保留尾部防止无限增长

                # 统计 token 数
                if in_thinking:
                    thinking_tokens += 1
                else:
                    answer_tokens += 1

                print(content, end="", flush=True)
                full_response += content

        # 如果 thinking 未闭合
        if in_thinking and think_start_time:
            think_time += time.time() - think_start_time

        # 记录结束时间
        end_time = time.time()
        total_time = end_time - start_time
        answer_time = total_time - think_time

        # 获取使用情况（部分 API 可能不提供）
        usage = getattr(stream, "usage", None)

        timing_info = {
            "total_time": f"{total_time:.2f}s",
            "think_time": f"{think_time:.2f}s",
            "answer_time": f"{answer_time:.2f}s",
            "ttfb": f"{ttfb:.2f}s" if ttfb else "N/A",
            "thinking_tokens": thinking_tokens,
            "answer_tokens": answer_tokens,
        }
        print(f"\n{YELLOW}[计时] 思考: {think_time:.2f}s | 回答: {answer_time:.2f}s | 总计: {total_time:.2f}s | TTFB: {timing_info['ttfb']}{NC}")

        return {
            "success": True,
            "answer": full_response,
            "metadata": {
                "model": model_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **timing_info,
                "usage": usage or {"tokens_total": len(full_response.split())},
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _safe_filename(question: str, max_len: int = 80) -> str:
    """将问题文本转为安全的文件名，截取前 max_len 个字符"""
    name = re.sub(r'[\\/:*?"<>|]', '_', question)
    name = re.sub(r'\s+', '_', name).strip('_')
    return name[:max_len]


def test_single_question(question, output_dir="math_test_result"):
    """测试单个问题（流式输出），并将结果保存为 JSON"""
    print(f"\n{BLUE}{'=' * 60}{NC}")
    print(f"{BLUE}  Math LLM API 测试（OpenAI SDK版本）{NC}")
    print(f"{BLUE}  服务器: {BASE_URL}{NC}")
    print(f"{BLUE}{'=' * 60}{NC}")

    # 创建客户端
    client = create_client()

    # 打印模型列表
    print(f"\n{BLUE}[*] 可用模型列表:{NC}")
    models_ok, models_data = test_models(client)

    if not models_ok:
        print(f"\n{RED}[-] 无法获取模型列表{NC}")
        return

    # 使用第一个模型
    default_model = models_data[0].id if models_data else "/home/phi-4-mini-reasoning"
    print(f"\n{BLUE}[*] 使用模型: {default_model}{NC}")

    # 测试单个问题
    print(f"\n{BLUE}[*] 测试问题: {question}{NC}")

    result = test_chat_completion_stream(client, question, default_model)
    result["question"] = question

    # 保存结果到 JSON 文件
    os.makedirs(output_dir, exist_ok=True)
    filename = _safe_filename(question) + ".json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"{GREEN}[+] 结果已保存: {filepath}{NC}")

    if result["success"]:
        print(f"\n{GREEN}[+] 解答成功{NC}")
    else:
        print(f"\n{RED}[-] 解答失败: {result.get('error')}{NC}")


if __name__ == "__main__":
    questions = [
        "已知数列AN的前N项和SN等于N乘以四分之一加上N平方乘以三分之二加三，求通项公式AN，并判断数列是否为等差数列。",
        "求曲线y等于x的三次方减去3x，在点一负二处的切线方程。",
        "求椭圆X平方除以25，加上Y的平方除以9等于1的焦距和离心率。",
        "椭圆的离心率E的变，这个E是什么？E的取值范围是多少？E的大小如何影响椭圆的形状啊？为什么？",
        "在空间向量与立体几何中，我们学习了空间向量的数量及运算。请结合教材中的例题说明如何利用空间向量的数量及证明直线与平行、平面垂直的判定定理，并写出用向量法证明该定理的主要步骤。",
        "在实际问题中，常常需要求函数的闭区间上的最大值和最小值。已请以函数FX等于三分之一X三次方减去四X加四在区间零到三上为闭区间为例，说明求最值的步骤，并解释为什么需要将其值与端点值进行比较。",
        "求抛物线y的平方等于8x的交点坐标，准线方程，并求抛物线上与交点距离为6的点的坐标。",
        "已知圆C的方程为X平方加Y的平方减四X加六Y减十二等于零，求圆心坐标、半径，并判断点M为一负二时在圆内圆上还是圆外。",
        "已知双曲线X平方除以9减去y的平方除以16等于1，求它的实半周长、虚半周长、交点坐标、倍心率和渐近线方程。",
        "椭圆的标准方程是通过将椭圆的几何定义到两定点距离之和为常数，转化为代数方程，并化简得到的。请叙述这两个推导过程，并说明为什么要令B的平方等于A的平方减C的平方，以及A、B、C的几何意义。",
    ]
    for question in questions:
        test_single_question(question)
