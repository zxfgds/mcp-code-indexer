"""
命令行接口模块
提供MCP代码检索服务的命令行工具
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any, List, Optional
import time

from .plugin import McpPlugin

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_indexer_cli.log')
    ]
)

logger = logging.getLogger(__name__)

def setup_parser() -> argparse.ArgumentParser:
    """
    设置命令行参数解析器
    
    Returns:
        参数解析器对象
    """
    parser = argparse.ArgumentParser(description='MCP代码检索工具')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 服务器URL参数
    parser.add_argument('--server', type=str, default='http://127.0.0.1:5000',
                       help='MCP服务器URL')
    
    # 识别项目命令
    identify_parser = subparsers.add_parser('identify', help='识别项目')
    identify_parser.add_argument('--path', type=str, default=os.getcwd(),
                               help='项目路径')
    
    # 索引项目命令
    index_parser = subparsers.add_parser('index', help='索引项目')
    index_parser.add_argument('--path', type=str, default=os.getcwd(),
                            help='项目路径')
    index_parser.add_argument('--wait', action='store_true',
                            help='等待索引完成')
    index_parser.add_argument('--timeout', type=int, default=300,
                            help='等待超时时间（秒）')
    
    # 搜索代码命令
    search_parser = subparsers.add_parser('search', help='搜索代码')
    search_parser.add_argument('query', type=str, help='查询字符串')
    search_parser.add_argument('--project-id', type=str, help='项目ID')
    search_parser.add_argument('--language', type=str, help='编程语言')
    search_parser.add_argument('--file-path', type=str, help='文件路径')
    search_parser.add_argument('--limit', type=int, default=10,
                             help='返回结果数量限制')
    
    # 获取代码上下文命令
    context_parser = subparsers.add_parser('context', help='获取代码上下文')
    context_parser.add_argument('--file', type=str, required=True,
                              help='文件路径')
    context_parser.add_argument('--line', type=int, required=True,
                              help='行号')
    context_parser.add_argument('--context-lines', type=int, default=10,
                              help='上下文行数')
    
    # 获取项目列表命令
    subparsers.add_parser('projects', help='获取项目列表')
    
    # 删除项目索引命令
    delete_parser = subparsers.add_parser('delete', help='删除项目索引')
    delete_parser.add_argument('project_id', type=str, help='项目ID')
    
    # 健康检查命令
    subparsers.add_parser('health', help='检查服务器健康状态')
    
    return parser

def handle_identify(plugin: McpPlugin, args: argparse.Namespace) -> None:
    """
    处理识别项目命令
    
    Args:
        plugin: MCP插件对象
        args: 命令行参数
        
    Returns:
        无返回值
    """
    result = plugin.identify_project(args.path)
    print(json.dumps(result, indent=2))

def handle_index(plugin: McpPlugin, args: argparse.Namespace) -> None:
    """
    处理索引项目命令
    
    Args:
        plugin: MCP插件对象
        args: 命令行参数
        
    Returns:
        无返回值
    """
    result = plugin.index_project(args.path, args.wait, args.timeout)
    print(json.dumps(result, indent=2))

def handle_search(plugin: McpPlugin, args: argparse.Namespace) -> None:
    """
    处理搜索代码命令
    
    Args:
        plugin: MCP插件对象
        args: 命令行参数
        
    Returns:
        无返回值
    """
    # 构建过滤条件
    filters = {}
    if args.language:
        filters['language'] = args.language
    if args.file_path:
        filters['file_path'] = args.file_path
    
    # 构建项目ID列表
    project_ids = [args.project_id] if args.project_id else None
    
    # 执行搜索
    result = plugin.search(args.query, project_ids, filters, args.limit)
    
    # 输出结果
    if args.project_id:
        # 格式化输出
        print(plugin.format_for_ai(result))
    else:
        # JSON输出
        print(json.dumps(result, indent=2))

def handle_context(plugin: McpPlugin, args: argparse.Namespace) -> None:
    """
    处理获取代码上下文命令
    
    Args:
        plugin: MCP插件对象
        args: 命令行参数
        
    Returns:
        无返回值
    """
    result = plugin.get_code_context(args.file, args.line, args.context_lines)
    print(json.dumps(result, indent=2))

def handle_projects(plugin: McpPlugin, args: argparse.Namespace) -> None:
    """
    处理获取项目列表命令
    
    Args:
        plugin: MCP插件对象
        args: 命令行参数
        
    Returns:
        无返回值
    """
    result = plugin.get_projects()
    print(json.dumps(result, indent=2))

def handle_delete(plugin: McpPlugin, args: argparse.Namespace) -> None:
    """
    处理删除项目索引命令
    
    Args:
        plugin: MCP插件对象
        args: 命令行参数
        
    Returns:
        无返回值
    """
    result = plugin.delete_project(args.project_id)
    print(json.dumps(result, indent=2))

def handle_health(plugin: McpPlugin, args: argparse.Namespace) -> None:
    """
    处理健康检查命令
    
    Args:
        plugin: MCP插件对象
        args: 命令行参数
        
    Returns:
        无返回值
    """
    is_healthy = plugin.health_check()
    if is_healthy:
        print("服务器状态: 正常")
    else:
        print("服务器状态: 异常")
        sys.exit(1)

def main():
    """
    主函数，处理命令行参数并执行相应操作
    
    Returns:
        无返回值
    """
    parser = setup_parser()
    args = parser.parse_args()
    
    # 如果没有指定命令，显示帮助信息
    if not args.command:
        parser.print_help()
        return
    
    # 创建MCP插件
    plugin = McpPlugin(args.server)
    
    # 处理命令
    command_handlers = {
        'identify': handle_identify,
        'index': handle_index,
        'search': handle_search,
        'context': handle_context,
        'projects': handle_projects,
        'delete': handle_delete,
        'health': handle_health
    }
    
    if args.command in command_handlers:
        try:
            command_handlers[args.command](plugin, args)
        except Exception as e:
            logger.error(f"命令执行失败: {str(e)}")
            print(f"错误: {str(e)}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()