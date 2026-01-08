"""
测试文件 MD5 计算性能
"""
import hashlib
import time
from pathlib import Path


def test_full_md5():
    """测试完整文件 MD5 计算"""
    file_path = Path(__file__).parent.parent.parent / "test_files" / "首饰雕蜡工艺-全本.pdf"

    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return

    file_size = file_path.stat().st_size
    print(f"文件大小: {file_size / 1024 / 1024:.2f} MB ({file_size:,} bytes)")

    # 测试完整文件 MD5
    start = time.time()
    content = file_path.read_bytes()
    full_md5 = hashlib.md5(content).hexdigest()
    elapsed = time.time() - start

    print(f"\n完整文件 MD5:")
    print(f"  MD5: {full_md5}")
    print(f"  计算时间: {elapsed:.3f} 秒")
    print(f"  速度: {file_size / 1024 / 1024 / elapsed:.2f} MB/s")

    # 测试部分文件 MD5（只读前 1MB）
    start = time.time()
    partial_content = file_path.read_bytes()[:1024 * 1024]  # 只读前 1MB
    partial_md5 = hashlib.md5(partial_content).hexdigest()
    elapsed = time.time() - start

    print(f"\n部分文件 MD5 (前 1MB):")
    print(f"  MD5: {partial_md5}")
    print(f"  计算时间: {elapsed:.3f} 秒")

    # 对比当前方案（只使用元信息）
    start = time.time()
    stat = file_path.stat()
    current_hash_input = f"{file_path.resolve()}_0_8_{stat.st_size}"
    current_md5 = hashlib.md5(current_hash_input.encode()).hexdigest()
    elapsed = time.time() - start

    print(f"\n当前方案 (元信息):")
    print(f"  MD5: {current_md5}")
    print(f"  计算时间: {elapsed:.6f} 秒")
    print(f"  速度提升: {elapsed} 秒 vs 0.XXX 秒")


if __name__ == "__main__":
    test_full_md5()
