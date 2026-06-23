import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.raganything_wrapper import (
    validate_required_env,
    check_raganything_services_health,
    get_raganything_instance,
    get_raganything_stream,
    reset_raganything_instance,
)


async def main():
    query = "请帮我讲解二项式定理"
    mode = "hybrid"

    print("1. 检查环境变量")
    validate_required_env()

    print("2. 检查依赖服务健康状态")
    health = await check_raganything_services_health()
    print(health)

    print("3. 初始化 RAGAnything")
    rag = await get_raganything_instance()
    print(f"RAGAnything initialized: {rag is not None}")

    print("4. 测试流式查询")
    merged_chunks = []
    async for chunk in get_raganything_stream(query=query, mode=mode):
        # type=chunk 是回答文本片段，累加后合并打印
        if chunk.get("type") == "chunk":
            content = chunk.get("content")
            if content:
                merged_chunks.append(content)
            continue
        # 其他类型（sources_info / sources / error 等）原样打印
        print(chunk)

    print("--- 合并后的 chunk 内容 ---")
    print("".join(merged_chunks))

    print("5. 重置/关闭 RAGAnything")
    await reset_raganything_instance()

    print("Done")


if __name__ == "__main__":
    asyncio.run(main())