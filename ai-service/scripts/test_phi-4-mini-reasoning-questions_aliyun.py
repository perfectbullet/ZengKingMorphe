#!/usr/bin/env python3
import requests
import json
import time

# 问题列表
QUESTIONS = [
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

# 根据实际情况修改
# SERVER_IP = "192.168.8.230"
# SERVER_IP = "192.168.8.235"
SERVER_IP = "39.97.230.23"
SERVER_PORT = 8000


# ----------------------------------------------------------------------------------------
# 获取模型的ID
# ----------------------------------------------------------------------------------------
def get_model_id(base_url="http://localhost:8000", index=0):
    response = requests.get(f"{base_url}/v1/models")
    if response.status_code != 200:
        print(f"获取模型列表失败：{response.status_code}")
        return None

    result = json.loads(response.content)
    if "data" not in result or len(result["data"]) == 0:
        print(json.dumps(result, indent=2))
        print("模型列表数据不存在")
        return None

    model_ids = [model["id"] for model in result["data"]]
    return model_ids[index] if index < len(model_ids) else None


# ----------------------------------------------------------------------------------------
# 从数据集中获取Q&A
# ----------------------------------------------------------------------------------------
def get_question_answer(index):
    if index < len(QUESTIONS):
        return QUESTIONS[index]
    return None


# ----------------------------------------------------------------------------------------
# 提交问题进行推理
# ----------------------------------------------------------------------------------------
def submit_question(
    question, model_id, index, base_url="http://localhost:8000", file_handle=None
):
    data = {
        "model": f"{model_id}",
        "messages": [{"role": "user", "content": f"{question}\n。用中文简洁地回答。"}],
        "temperature": 0.7,
        "top_p": 0.7,
        "repetition_penalty": 1.05,
        "max_tokens": 8192,
        "stream": True,
    }

    if file_handle is not None:
        file_handle.write("=" * 80)
        file_handle.write(f"\n")
        file_handle.write(f"Question {index}:\n{question}\n")
        file_handle.write("=" * 80)
        file_handle.write(f"\n")
        file_handle.flush()

    t1, result = time.time(), ""
    response = requests.post(f"{base_url}/v1/chat/completions", json=data, stream=True)
    # print(f"Response:")
    for line in response.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data_string = line[6:]
        if data_string == "[DONE]":
            continue
        try:
            data_json = json.loads(data_string)
            if "choices" in data_json and len(data_json["choices"]) > 0:
                delta = data_json["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
                    result += content
        except json.JSONDecodeError:
            pass
    delta_time = time.time() - t1

    # print("\n")
    if file_handle is not None:
        file_handle.write(result)
        file_handle.write(f"\n")
        file_handle.flush()

    return result, delta_time


# ----------------------------------------------------------------------------------------
# 程序主函数
# ----------------------------------------------------------------------------------------
def main(args=None):
    model_id = get_model_id(f"http://{SERVER_IP}:{SERVER_PORT}", 0)
    print(f"模型ID：{model_id}")
    if model_id is None:
        return

    output_file = f"questions-{model_id.split('/')[2]}.log"
    base_url = f"http://{SERVER_IP}:{SERVER_PORT}"
    total_token_count, elcipse_seconds, loop_count, index = 0, 0, 0, 0
    output_handle = open(output_file, "a", encoding="utf-8")

    while True:
        if index == 0:
            print("=" * 80)
            print(f"Loop {loop_count}")
            print("=" * 80)
            output_handle.write("=" * 80)
            output_handle.write("\n")
            output_handle.write(f"Loop {loop_count}")
            output_handle.write("=" * 80)
            output_handle.write("\n")
            output_handle.flush()

        question = QUESTIONS[index]
        if question is None:
            break

        print("=" * 80)
        print(f"Question {index}:\n{question}")
        print("=" * 80)

        tokens, delt_time = submit_question(
            question, model_id, index, base_url, output_handle
        )
        total_token_count += len(tokens)
        elcipse_seconds += delt_time
        index += 1

        token_speed = (
            total_token_count / elcipse_seconds if elcipse_seconds > 0 else 0.0
        )
        print("=" * 80)
        print(f"服务吞吐速率: {token_speed:.2f}tokens/s")
        print("=" * 80)
        output_handle.write("=" * 80)
        output_handle.write("\n")
        output_handle.write(f"服务吞吐速率: {token_speed:.2f}tokens/s\n")
        output_handle.write("=" * 80)
        output_handle.write("\n")
        output_handle.flush()

        if index == len(QUESTIONS):
            loop_count += 1
            index = 0

    output_handle.close()


# ----------------------------------------------------------------------------------------
# 程序入口
# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
