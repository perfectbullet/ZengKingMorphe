import time

import requests
import json

# 你的服务地址
BASE_URL = "http://192.168.8.222:8080"

def test_word2latex():
    url = f"{BASE_URL}/word2latex"
    data = {
        "text": "求函数f(x)等于x的立方的复合函数，在椭圆的交点坐标处求导数",
        "max_tokens": 250,
        "temperature": 0
    }
    start=time.time()
    response = requests.post(url, json=data)
    print(f"=== word2latex 结果 === time:{time.time()-start:.2f}")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))

def test_latex2word():
    url = f"{BASE_URL}/latex2word"
    data = {
        "text": r"这是什么，看不懂\int_{0}^{1} x^3 dx",
        "max_tokens": 250,
        "temperature": 0
    }
    start = time.time()
    response = requests.post(url, json=data)
    print(f"=== latex2word 结果 === time:{time.time() - start:.2f}")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))

# ======================
# 示例 3：健康检查
# ======================
def test_health():
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    print("\n=== 健康检查 ===")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    test_health()
    test_word2latex()
    test_latex2word()
