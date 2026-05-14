import functools

from langchain_openai import ChatOpenAI


@functools.lru_cache(maxsize=4)
def get_vllm_first_model(base_url: str) -> str:
    """获取 vLLM 第一个模型 ID（结果按 base_url 缓存）

    Args:
        base_url: vLLM API base URL（如 http://192.168.8.235:8000/v1）

    Returns:
        第一个模型的 ID

    Raises:
        ValueError: 当模型列表为空时
        Exception: 当 API 调用失败时
    """
    try:
        client = ChatOpenAI(
            base_url=base_url,
            api_key="dummy-key",
        )
        # 访问底层的 OpenAI 客户端
        response = client.root_client.models.list()

        # 检查模型列表是否为空
        if not response.data:
            raise ValueError(f"未找到可用的模型，base_url={base_url}")

        # 返回第一个模型 ID
        return response.data[0].id

    except Exception as e:
        raise Exception(f"获取 vLLM 模型列表失败: {e}\n\nbase_url={base_url}") from e


if __name__ == "__main__":
    try:
        first_model = get_vllm_first_model("http://192.168.100.230:8000/v1")
        print(f"✓ 第一个模型 ID: {first_model}")
    except Exception as e:
        print(f"✗ 错误: {e}")
