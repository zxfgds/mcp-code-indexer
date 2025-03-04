"""
服务器应用模块
提供MCP代码检索服务实现
"""

import os
import sys
import logging
import argparse
import locale
import asyncio

# 设置控制台编码为UTF-8以解决编码问题
if sys.platform == 'win32':
    # Windows平台
    os.system('chcp 65001')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
else:
    # 其他平台
    if locale.getpreferredencoding().upper() != 'UTF-8':
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 从MCP库导入stdio_server函数
from mcp.server.stdio import stdio_server

from mcp_code_indexer.config import Config
from mcp_code_indexer.indexer import CodeIndexer
from mcp_code_indexer.search_engine import SearchEngine
from mcp_code_indexer.mcp_formatter import McpFormatter

# 使用绝对导入，避免作为脚本直接运行时的问题
from server.mcp_server import setup_mcp_server

# 创建日志过滤器
class MpcLogFilter(logging.Filter):
    """过滤MCP服务器的底层日志"""
    def filter(self, record):
        # 过滤掉MCP服务器的底层处理日志
        if record.name.startswith('mcp.server.lowlevel'):
            return False
        return True

# 禁用所有日志输出
class NullHandler(logging.Handler):
    def emit(self, record):
        pass

# 配置空日志处理器
logging.basicConfig(handlers=[NullHandler()])

# 设置所有已知logger的级别为CRITICAL以上
logging.getLogger('mcp').setLevel(logging.CRITICAL)
logging.getLogger('chromadb').setLevel(logging.CRITICAL)
logging.getLogger('sentence_transformers').setLevel(logging.CRITICAL)
logging.getLogger('tree_sitter').setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)

async def run_server(server):
    """
    运行带有stdio传输的MCP服务器
    
    Args:
        server: MCP服务器实例
        
    Returns:
        无返回值
    """
    async with stdio_server() as (read_stream, write_stream):
        # 使用流运行服务器
        from mcp.server.models import InitializationOptions
        from mcp.server.lowlevel import NotificationOptions
        
        init_options = InitializationOptions(
            server_name="mcp-code-indexer",
            server_version="0.1.0",
            capabilities=server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={}
            )
        )
        
        await server.run(read_stream, write_stream, init_options)

def main():
    """
    主函数，启动MCP服务器
    
    Returns:
        无返回值
    """
    parser = argparse.ArgumentParser(description='MCP Code Retrieval Service')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = Config(args.config)
        
        # 初始化组件
        indexer = CodeIndexer(config)
        search_engine = SearchEngine(config, indexer)
        formatter = McpFormatter()
        
        # 创建MCP服务器
        server = setup_mcp_server(config, indexer, search_engine, formatter)
        
        # 处理退出信号
        def handle_exit(signum, frame):
            sys.exit(0)
        
        import signal
        signal.signal(signal.SIGINT, handle_exit)
        signal.signal(signal.SIGTERM, handle_exit)
        
        # 运行服务器
        asyncio.run(run_server(server))
        
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        sys.exit(1)

if __name__ == '__main__':
    main()