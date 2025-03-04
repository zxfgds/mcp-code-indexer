"""
MCP响应格式化模块
负责将检索结果转换为符合MCP协议的格式
"""

import os
import json
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class McpFormatter:
    """
    MCP响应格式化器类
    
    将检索结果转换为符合MCP协议的格式，包含代码内容、位置、关系等信息
    """
    
    def __init__(self):
        """
        初始化MCP响应格式化器
        
        Returns:
            无返回值
        """
        logger.info("MCP响应格式化器初始化完成")
    
    def format_search_results(self, results: List[Dict[str, Any]], 
                             query: str, confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        格式化搜索结果为MCP响应
        
        Args:
            results: 搜索结果列表
            query: 原始查询字符串
            confidence_threshold: 置信度阈值，低于此值的结果将被标记为低置信度
            
        Returns:
            MCP格式的响应字典
        """
        if not results:
            return self._create_empty_response(query)
        
        # 格式化代码块
        code_blocks = []
        for result in results:
            code_block = self._format_code_block(result, confidence_threshold)
            if code_block:
                code_blocks.append(code_block)
        
        # 创建MCP响应
        response = {
            "mcp_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result_count": len(code_blocks),
            "code_blocks": code_blocks
        }
        
        return response
    
    def _format_code_block(self, result: Dict[str, Any], 
                          confidence_threshold: float) -> Optional[Dict[str, Any]]:
        """
        格式化单个代码块
        
        Args:
            result: 代码块搜索结果
            confidence_threshold: 置信度阈值
            
        Returns:
            格式化后的代码块字典，如果格式化失败则返回None
        """
        try:
            # 提取基本信息
            file_path = result.get('file_path', '')
            content = result.get('content', '')
            start_line = result.get('start_line', 1)
            end_line = result.get('end_line', 1)
            language = result.get('language', 'text')
            similarity = result.get('similarity', 0.0)
            
            # 计算置信度
            confidence = similarity
            low_confidence = confidence < confidence_threshold
            
            # 创建代码块字典
            code_block = {
                "id": self._generate_block_id(file_path, start_line, end_line),
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "language": language,
                "start_line": start_line,
                "end_line": end_line,
                "content": content,
                "confidence": confidence,
                "low_confidence": low_confidence,
                "metadata": {
                    "similarity": similarity,
                    "type": result.get('type', 'code')
                }
            }
            
            return code_block
        except Exception as e:
            logger.error(f"格式化代码块失败: {str(e)}")
            return None
    
    def _generate_block_id(self, file_path: str, start_line: int, end_line: int) -> str:
        """
        生成代码块ID
        
        Args:
            file_path: 文件路径
            start_line: 起始行号
            end_line: 结束行号
            
        Returns:
            代码块ID字符串
        """
        file_name = os.path.basename(file_path)
        return f"{file_name}:{start_line}-{end_line}"
    
    def _create_empty_response(self, query: str) -> Dict[str, Any]:
        """
        创建空响应
        
        Args:
            query: 原始查询字符串
            
        Returns:
            空的MCP响应字典
        """
        return {
            "mcp_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result_count": 0,
            "code_blocks": [],
            "message": "未找到结果"
        }
    
    def format_project_info(self, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化项目信息为MCP响应
        
        Args:
            project_info: 项目信息字典
            
        Returns:
            MCP格式的项目信息响应字典
        """
        return {
            "mcp_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "project_id": project_info.get("project_id"),
            "project_path": project_info.get("project_path"),
            "indexed_at": project_info.get("indexed_at"),
            "file_count": project_info.get("file_count"),
            "status": project_info.get("status"),
            "progress": project_info.get("progress", 0.0),
            "metadata": project_info.get("metadata", {})
        }
    
    def format_code_context(self, code_context: Dict[str, Any], 
                           related_blocks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        格式化代码上下文为MCP响应
        
        Args:
            code_context: 代码上下文字典
            related_blocks: 相关代码块列表
            
        Returns:
            MCP格式的代码上下文响应字典
        """
        # 格式化主代码块
        main_block = {
            "id": self._generate_block_id(
                code_context.get('file_path', ''),
                code_context.get('start_line', 1),
                code_context.get('end_line', 1)
            ),
            "file_path": code_context.get('file_path', ''),
            "file_name": os.path.basename(code_context.get('file_path', '')),
            "start_line": code_context.get('start_line', 1),
            "end_line": code_context.get('end_line', 1),
            "target_line": code_context.get('target_line', 1),
            "content": code_context.get('content', ''),
            "language": self._guess_language(code_context.get('file_path', ''))
        }
        
        # 格式化相关代码块
        related_code_blocks = []
        if related_blocks:
            for block in related_blocks:
                formatted_block = self._format_code_block(block, 0.0)
                if formatted_block:
                    related_code_blocks.append(formatted_block)
        
        # 创建MCP响应
        response = {
            "mcp_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "code_context": main_block,
            "related_blocks": related_code_blocks,
            "related_count": len(related_code_blocks)
        }
        
        return response
    
    def _guess_language(self, file_path: str) -> str:
        """
        根据文件扩展名猜测编程语言
        
        Args:
            file_path: 文件路径
            
        Returns:
            编程语言字符串
        """
        ext_to_lang = {
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
        
        _, ext = os.path.splitext(file_path.lower())
        return ext_to_lang.get(ext, "text")
    
    def format_error(self, error_message: str, query: str = None) -> Dict[str, Any]:
        """
        格式化错误为MCP响应
        
        Args:
            error_message: 错误消息
            query: 原始查询字符串
            
        Returns:
            MCP格式的错误响应字典
        """
        return {
            "mcp_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "error": True,
            "error_message": error_message,
            "result_count": 0,
            "code_blocks": []
        }
    
    def format_indexing_status(self, project_id: str, status: str, 
                              progress: float, message: str = None) -> Dict[str, Any]:
        """
        格式化索引状态为MCP响应
        
        Args:
            project_id: 项目ID
            status: 索引状态
            progress: 索引进度（0.0-1.0）
            message: 状态消息
            
        Returns:
            MCP格式的索引状态响应字典
        """
        return {
            "mcp_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "project_id": project_id,
            "indexing_status": {
                "status": status,
                "progress": progress,
                "message": message or f"索引状态：{status} ({progress:.1%})"
            }
        }