"""
嵌入向量工具模块
提供嵌入向量生成、处理和比较的工具函数
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import math
from concurrent.futures import ThreadPoolExecutor
import os

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers 库未安装，某些嵌入功能将不可用")

def create_embeddings(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                     batch_size: int = 32) -> Optional[np.ndarray]:
    """
    使用指定模型为文本创建嵌入向量
    
    Args:
        texts: 文本列表
        model_name: 嵌入模型名称
        batch_size: 批处理大小
        
    Returns:
        嵌入向量数组，如果生成失败则返回None
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("无法创建嵌入：sentence-transformers 库未安装")
        return None
        
    if not texts:
        return np.array([])
        
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return embeddings
    except Exception as e:
        logger.error(f"创建嵌入向量失败: {str(e)}")
        return None

def batch_encode_texts(texts: List[str], model: Any, batch_size: int = 32) -> np.ndarray:
    """
    批量编码文本为嵌入向量
    
    Args:
        texts: 文本列表
        model: 嵌入模型对象
        batch_size: 批处理大小
        
    Returns:
        嵌入向量数组
    """
    if not texts:
        return np.array([])
        
    # 分批处理
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    all_embeddings = []
    
    for batch in batches:
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
    
    # 合并所有批次的结果
    return np.vstack(all_embeddings) if all_embeddings else np.array([])

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度
    
    Args:
        vec1: 第一个向量
        vec2: 第二个向量
        
    Returns:
        余弦相似度值，范围为[-1, 1]
    """
    if len(vec1.shape) > 1 or len(vec2.shape) > 1:
        raise ValueError("输入必须是一维向量")
        
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return np.dot(vec1, vec2) / (norm1 * norm2)

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    归一化嵌入向量
    
    Args:
        embeddings: 嵌入向量数组
        
    Returns:
        归一化后的嵌入向量数组
    """
    # 计算每个向量的范数
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # 避免除以零
    norms[norms == 0] = 1.0
    
    # 归一化
    return embeddings / norms

def parallel_encode(texts: List[str], model: Any, batch_size: int = 32, 
                   max_workers: Optional[int] = None) -> np.ndarray:
    """
    并行编码文本为嵌入向量
    
    Args:
        texts: 文本列表
        model: 嵌入模型对象
        batch_size: 批处理大小
        max_workers: 最大工作线程数，默认为CPU核心数
        
    Returns:
        嵌入向量数组
    """
    if not texts:
        return np.array([])
    
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    
    # 分批处理
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    results = [None] * len(batches)
    
    def encode_batch(batch_idx: int, batch: List[str]) -> Tuple[int, np.ndarray]:
        """编码单个批次并返回批次索引和结果"""
        embeddings = model.encode(batch, show_progress_bar=False)
        return batch_idx, embeddings
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(encode_batch, i, batch) for i, batch in enumerate(batches)]
        
        for future in futures:
            batch_idx, embeddings = future.result()
            results[batch_idx] = embeddings
    
    # 合并所有批次的结果
    return np.vstack(results) if results else np.array([])

def calculate_similarity_matrix(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    计算两组嵌入向量之间的相似度矩阵
    
    Args:
        embeddings1: 第一组嵌入向量
        embeddings2: 第二组嵌入向量
        
    Returns:
        相似度矩阵，形状为 (len(embeddings1), len(embeddings2))
    """
    # 归一化嵌入向量
    norm_embeddings1 = normalize_embeddings(embeddings1)
    norm_embeddings2 = normalize_embeddings(embeddings2)
    
    # 计算点积（即余弦相似度）
    return np.dot(norm_embeddings1, norm_embeddings2.T)

def find_top_k_similar(query_embedding: np.ndarray, corpus_embeddings: np.ndarray, 
                      k: int = 5) -> Tuple[List[int], List[float]]:
    """
    找出与查询向量最相似的k个语料向量
    
    Args:
        query_embedding: 查询嵌入向量
        corpus_embeddings: 语料库嵌入向量
        k: 返回的最相似向量数量
        
    Returns:
        元组(索引列表, 相似度列表)
    """
    # 确保查询向量是二维的
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # 计算相似度
    similarities = calculate_similarity_matrix(query_embedding, corpus_embeddings)[0]
    
    # 找出前k个最相似的索引
    k = min(k, len(similarities))
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_scores = similarities[top_k_indices]
    
    return top_k_indices.tolist(), top_k_scores.tolist()

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    将文本分割成重叠的块
    
    Args:
        text: 输入文本
        chunk_size: 块大小（字符数）
        chunk_overlap: 块重叠大小（字符数）
        
    Returns:
        文本块列表
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # 如果不是最后一块，尝试在空白处分割
        if end < text_len:
            # 在chunk_size范围内找最后一个空白字符
            while end > start + chunk_size - chunk_overlap and not text[end].isspace():
                end -= 1
            
            # 如果没找到合适的分割点，就直接在chunk_size处分割
            if end <= start + chunk_size - chunk_overlap:
                end = start + chunk_size
        
        chunks.append(text[start:end])
        start = end - chunk_overlap
    
    return chunks