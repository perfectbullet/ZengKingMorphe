from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",  # vLLM默认无需认证
    base_url="http://localhost:8095/v1"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Math-7B-Instruct",
    messages=[
        {"role": "system", "content": "Please reason step by step, and put your final answer within boxed{}."},
        {"role": "user", "content": "AIME 2024 Problem 1: Find the number of..."}
    ],
    temperature=0.6,
    top_p=0.95,
    max_tokens=4096
)
print(response.choices[0].message.content)