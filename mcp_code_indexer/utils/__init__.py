"""
工具函数包
提供各种辅助功能的工具函数
"""

# 导出主要函数，方便用户直接从包中导入
from .file_utils import (
    is_binary_file,
    get_file_language,
    normalize_path,
    get_relative_path,
    read_file_content,
    get_file_hash
)

from .embedding_utils import (
    create_embeddings,
    cosine_similarity,
    normalize_embeddings,
    batch_encode_texts
)

__all__ = [
    "is_binary_file",
    "get_file_language",
    "normalize_path",
    "get_relative_path",
    "read_file_content",
    "get_file_hash",
    "create_embeddings",
    "cosine_similarity",
    "normalize_embeddings",
    "batch_encode_texts"
]