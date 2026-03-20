"""
全局常量定义

集中管理项目中的魔法数字和常量
"""

from enum import IntEnum

# ========== Embedding 配置常量 ==========
class EmbeddingDefaults(IntEnum):
    """Embedding 默认值"""
    DEFAULT_DIMENSION = 1024  # BGE-M3 默认维度
    DEFAULT_BATCH_SIZE = 32   # 批量处理大小
    DEFAULT_TIMEOUT = 300     # 超时时间（秒）
    MIN_TEXT_LENGTH = 3      # 最小文本长度

# ========== Reranker 配置常量 ==========
class RerankerDefaults:
    """Reranker 默认值"""
    SCORE_MIN = -10.0         # 最小分数
    SCORE_MAX = 10.0          # 最大分数
    SCORE_RANGE = 20.0       # 分数范围 (max - min)

# ========== 分块配置常量 ==========
class ChunkingDefaults(IntEnum):
    """分块默认值"""
    MIN_CHUNK_SIZE = 100     # 最小分块大小
    MAX_CHUNK_SIZE = 1024    # 最大分块大小
    DEFAULT_OVERLAP = 50     # 默认重叠大小

# ========== 性能配置常量 ==========
class PerformanceDefaults(IntEnum):
    """性能相关默认值"""
    MAX_RETRIES = 3          # 最大重试次数
    RETRY_DELAY = 1          # 重试延迟（秒）
    REQUEST_TIMEOUT = 120    # 默认请求超时（秒）

# ========== 文本处理常量 ==========
class TextDefaults:
    """文本处理默认值"""
    DEFAULT_SEPARATOR = "\n\n"  # 默认分隔符
    MAX_TRUNCATION_LENGTH = 1000  # 截断文本最大长度（用于日志）

# ========== MinerU 分块配置常量 ==========
class MinerUChunkingDefaults(IntEnum):
    """MinerU 结构感知分块默认值"""
    MIN_CHUNK_SIZE_RATIO = 15  # min_chunk_size = max * 0.15
    MIN_VALID_CHUNK_LENGTH = 30  # 最小有效 chunk 长度
    TITLE_PREFIX_OVERHEAD = 2  # 标题前缀开销（换行符等）

# ========== 分块策略枚举 ==========
class ChunkingStrategy:
    """分块策略"""
    HYBRID = "hybrid"  # 混合策略（按标题分块，超长再分割）
