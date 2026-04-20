from langchain_openai import ChatOpenAI


def get_vllm_first_model(base_url: str) -> str:
    """获取 vLLM 第一个模型 ID

    Args:
        client: ChatOpenAI 实例

    Returns:
        第一个模型的 ID

    Raises:
        ValueError: 当模型列表为空时
        Exception: 当 API 调用失败时
    """
    try:
        client = ChatOpenAI(
            base_url=base_url,  # 注意：base_url 不要加 /models
            api_key="dummy-key",
        )
        # 访问底层的 OpenAI 客户端
        response = client.root_client.models.list()

        # 检查模型列表是否为空
        if not response.data:
            raise ValueError(f"未找到可用的模型，base_url={client.openai_api_base}")

        # 返回第一个模型 ID
        return response.data[0].id

    except ValueError:
        raise  # 重新抛出 ValueError
    except Exception as e:
        raise Exception(f"获取 vLLM 模型列表失败: {e}") from e


if __name__ == "__main__":
    client = ChatOpenAI(
        base_url="http://192.168.8.235:8000/v1",  # 注意：不要加 /models
        api_key="dummy-key",
    )
    try:
        first_model = get_vllm_first_model(client)
        print(f"✓ 第一个模型 ID: {first_model}")
    except Exception as e:
        print(f"✗ 错误: {e}")
