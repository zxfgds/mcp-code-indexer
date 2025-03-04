"""
MCP代码检索工具包
提供基于MCP协议的代码检索功能，帮助AI大语言模型高效理解代码库
"""

__version__ = "0.1.0"
__author__ = "MCP Team"

# 导出主要类和函数，方便用户直接从包中导入
from .indexer import CodeIndexer
from .project_identity import ProjectIdentifier
from .search_engine import SearchEngine
from .mcp_formatter import McpFormatter
from .config import Config

__all__ = [
    "CodeIndexer",
    "ProjectIdentifier", 
    "SearchEngine",
    "McpFormatter",
    "Config"
]