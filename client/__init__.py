"""
客户端包
提供MCP代码检索服务的客户端实现
"""

# 导出主要类，方便用户直接从包中导入
from .plugin import McpPlugin
from .cli import main as cli_main

__all__ = [
    "McpPlugin",
    "cli_main"
]