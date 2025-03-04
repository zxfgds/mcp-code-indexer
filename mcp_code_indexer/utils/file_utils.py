"""
文件操作工具模块
提供文件读取、路径处理等工具函数
"""

import os
import hashlib
import logging
from typing import Optional, Tuple, List
from pathlib import Path
import mimetypes
import re

logger = logging.getLogger(__name__)

# 二进制文件的MIME类型前缀
BINARY_MIME_PREFIXES = [
    'image/',
    'audio/',
    'video/',
    'application/octet-stream',
    'application/zip',
    'application/x-tar',
    'application/x-rar-compressed',
    'application/x-7z-compressed',
    'application/pdf',
    'application/msword',
    'application/vnd.ms-',
    'application/vnd.openxmlformats-',
]

# 文件扩展名到语言的映射
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".rs": "rust",
    ".sh": "bash",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sql": "sql",
    ".md": "markdown",
    ".json": "json",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml"
}

def is_binary_file(file_path: str) -> bool:
    """
    检查文件是否为二进制文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        如果是二进制文件则返回True，否则返回False
    """
    # 检查文件扩展名
    _, ext = os.path.splitext(file_path.lower())
    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
               '.mp3', '.mp4', '.avi', '.mov', '.flv', '.wmv',
               '.zip', '.tar', '.gz', '.rar', '.7z',
               '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
               '.bin', '.exe', '.dll', '.so', '.dylib', '.class']:
        return True
    
    # 检查MIME类型
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        for prefix in BINARY_MIME_PREFIXES:
            if mime_type.startswith(prefix):
                return True
    
    # 读取文件头部检查是否包含空字节
    try:
        with open(file_path, 'rb') as f:
            header = f.read(1024)
            if b'\x00' in header:
                return True
    except Exception as e:
        logger.error(f"检查文件类型失败 {file_path}: {str(e)}")
        return True  # 如果无法读取，保守地认为是二进制文件
    
    return False

def get_file_language(file_path: str) -> str:
    """
    根据文件扩展名获取编程语言
    
    Args:
        file_path: 文件路径
        
    Returns:
        编程语言字符串，如果未知则返回"text"
    """
    _, ext = os.path.splitext(file_path.lower())
    return EXTENSION_TO_LANGUAGE.get(ext, "text")

def normalize_path(path: str) -> str:
    """
    规范化文件路径
    
    Args:
        path: 文件路径
        
    Returns:
        规范化后的路径字符串
    """
    return os.path.normpath(path).replace('\\', '/')

def get_relative_path(path: str, base_path: str) -> str:
    """
    获取相对于基础路径的相对路径
    
    Args:
        path: 文件路径
        base_path: 基础路径
        
    Returns:
        相对路径字符串
    """
    rel_path = os.path.relpath(path, base_path)
    return normalize_path(rel_path)

def read_file_content(file_path: str, max_size_kb: int = 1024) -> Tuple[Optional[str], Optional[str]]:
    """
    读取文件内容
    
    Args:
        file_path: 文件路径
        max_size_kb: 最大文件大小（KB）
        
    Returns:
        元组(文件内容, 错误消息)，如果读取失败则内容为None
    """
    try:
        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size > max_size_kb * 1024:
            return None, f"文件过大: {file_size / 1024:.1f} KB > {max_size_kb} KB"
        
        # 检查是否为二进制文件
        if is_binary_file(file_path):
            return None, "二进制文件不支持读取"
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        return content, None
    except Exception as e:
        logger.error(f"读取文件失败 {file_path}: {str(e)}")
        return None, str(e)

def get_file_hash(file_path: str) -> Optional[str]:
    """
    计算文件的MD5哈希值
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件的MD5哈希值，如果计算失败则返回None
    """
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5()
            # 分块读取大文件
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        return file_hash.hexdigest()
    except Exception as e:
        logger.error(f"计算文件哈希失败 {file_path}: {str(e)}")
        return None

def find_files(directory: str, file_pattern: str = None, 
              exclude_patterns: List[str] = None) -> List[str]:
    """
    在目录中查找符合模式的文件
    
    Args:
        directory: 目录路径
        file_pattern: 文件名模式（正则表达式）
        exclude_patterns: 排除模式列表
        
    Returns:
        符合条件的文件路径列表
    """
    if exclude_patterns is None:
        exclude_patterns = []
        
    file_regex = re.compile(file_pattern) if file_pattern else None
    result = []
    
    for root, dirs, files in os.walk(directory):
        # 排除忽略的目录
        dirs[:] = [d for d in dirs if not any(re.match(p, d) for p in exclude_patterns)]
        
        for file in files:
            if file_regex and not file_regex.match(file):
                continue
                
            file_path = os.path.join(root, file)
            if not any(re.match(p, file) for p in exclude_patterns):
                result.append(file_path)
    
    return result

def ensure_directory(directory: str) -> bool:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
        
    Returns:
        如果目录已存在或创建成功则返回True，否则返回False
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"创建目录失败 {directory}: {str(e)}")
        return False

def get_file_modification_time(file_path: str) -> Optional[float]:
    """
    获取文件的最后修改时间
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件的最后修改时间戳，如果获取失败则返回None
    """
    try:
        return os.path.getmtime(file_path)
    except Exception as e:
        logger.error(f"获取文件修改时间失败 {file_path}: {str(e)}")
        return None

def get_file_size(file_path: str) -> Optional[int]:
    """
    获取文件大小
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件大小（字节），如果获取失败则返回None
    """
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        logger.error(f"获取文件大小失败 {file_path}: {str(e)}")
        return None