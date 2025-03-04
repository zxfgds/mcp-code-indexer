"""
检索引擎模块
负责基于向量数据库的代码语义检索
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import json

from .config import Config
from .indexer import CodeIndexer

logger = logging.getLogger(__name__)

class SearchEngine:
    """
    检索引擎类
    
    基于向量数据库的代码语义检索，提供高级搜索功能
    """
    
    def __init__(self, config: Config, indexer: CodeIndexer):
        """
        初始化检索引擎
        
        Args:
            config: 配置对象
            indexer: 代码索引器对象
            
        Returns:
            无返回值
        """
        self.config = config
        self.indexer = indexer
        logger.info("检索引擎初始化完成")
    
    def search(self, query: str, project_ids: Optional[List[str]] = None, 
               filters: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        搜索代码
        
        Args:
            query: 查询字符串
            project_ids: 项目ID列表，如果为None则搜索所有项目
            filters: 过滤条件，如语言、文件类型等
            limit: 返回结果数量限制
            
        Returns:
            代码块字典列表
        """
        logger.info(f"搜索代码: {query}, 项目: {project_ids}, 过滤条件: {filters}")
        
        # 如果未指定项目，获取所有已索引项目
        if not project_ids:
            indexed_projects = self.indexer.get_indexed_projects()
            project_ids = [p["project_id"] for p in indexed_projects]
        
        all_results = []
        
        # 对每个项目执行搜索
        for project_id in project_ids:
            results = self.indexer.search(project_id, query, limit=limit)
            
            # 应用过滤条件
            if filters:
                results = self._apply_filters(results, filters)
            
            all_results.extend(results)
        
        # 按相似度排序
        all_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # 限制结果数量
        return all_results[:limit]
    
    def _apply_filters(self, results: List[Dict[str, Any]], 
                      filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        应用过滤条件
        
        Args:
            results: 搜索结果列表
            filters: 过滤条件字典
            
        Returns:
            过滤后的结果列表
        """
        filtered_results = []
        
        for result in results:
            match = True
            
            # 语言过滤
            if 'language' in filters and result.get('language') != filters['language']:
                match = False
            
            # 文件路径过滤
            if 'file_path' in filters:
                file_path = result.get('file_path', '')
                if filters['file_path'] not in file_path:
                    match = False
            
            # 代码类型过滤
            if 'type' in filters and result.get('type') != filters['type']:
                match = False
            
            if match:
                filtered_results.append(result)
        
        return filtered_results
    
    def search_by_file(self, file_path: str, project_id: Optional[str] = None, 
                      limit: int = 10) -> List[Dict[str, Any]]:
        """
        按文件路径搜索代码
        
        Args:
            file_path: 文件路径
            project_id: 项目ID，如果为None则搜索所有项目
            limit: 返回结果数量限制
            
        Returns:
            代码块字典列表
        """
        # 构建过滤条件
        filters = {'file_path': file_path}
        
        # 如果未指定项目，获取所有已索引项目
        project_ids = [project_id] if project_id else None
        
        # 使用文件路径作为查询，并应用过滤条件
        return self.search(os.path.basename(file_path), project_ids, filters, limit)
    
    def search_by_function(self, function_name: str, project_ids: Optional[List[str]] = None, 
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        按函数名搜索代码
        
        Args:
            function_name: 函数名
            project_ids: 项目ID列表，如果为None则搜索所有项目
            limit: 返回结果数量限制
            
        Returns:
            代码块字典列表
        """
        # 构建函数定义的查询
        query = f"function {function_name}" if function_name else ""
        
        # 执行搜索
        return self.search(query, project_ids, None, limit)
    
    def search_by_class(self, class_name: str, project_ids: Optional[List[str]] = None, 
                       limit: int = 10) -> List[Dict[str, Any]]:
        """
        按类名搜索代码
        
        Args:
            class_name: 类名
            project_ids: 项目ID列表，如果为None则搜索所有项目
            limit: 返回结果数量限制
            
        Returns:
            代码块字典列表
        """
        # 构建类定义的查询
        query = f"class {class_name}" if class_name else ""
        
        # 执行搜索
        return self.search(query, project_ids, None, limit)
    
    def get_related_code(self, code_chunk: Dict[str, Any], 
                        limit: int = 5) -> List[Dict[str, Any]]:
        """
        获取与给定代码块相关的代码
        
        Args:
            code_chunk: 代码块字典
            limit: 返回结果数量限制
            
        Returns:
            相关代码块字典列表
        """
        # 使用代码块内容作为查询
        query = code_chunk.get('content', '')
        
        # 获取项目ID
        file_path = code_chunk.get('file_path', '')
        project_id = self._get_project_id_by_file_path(file_path)
        
        if not project_id:
            return []
        
        # 执行搜索，排除自身
        results = self.indexer.search(project_id, query, limit=limit+1)
        
        # 过滤掉自身
        filtered_results = []
        for result in results:
            if (result.get('file_path') != file_path or 
                result.get('start_line') != code_chunk.get('start_line')):
                filtered_results.append(result)
        
        return filtered_results[:limit]
    
    def _get_project_id_by_file_path(self, file_path: str) -> Optional[str]:
        """
        根据文件路径获取项目ID
        
        Args:
            file_path: 文件路径
            
        Returns:
            项目ID，如果未找到则返回None
        """
        # 获取所有已索引项目
        indexed_projects = self.indexer.get_indexed_projects()
        
        for project in indexed_projects:
            project_path = project.get("project_path", "")
            if project_path and file_path.startswith(project_path):
                return project.get("project_id")
        
        return None
    
    def get_code_context(self, file_path: str, line_number: int, 
                        context_lines: int = 10) -> Dict[str, Any]:
        """
        获取代码上下文
        
        Args:
            file_path: 文件路径
            line_number: 行号
            context_lines: 上下文行数
            
        Returns:
            代码上下文字典
        """
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            # 计算上下文范围
            start_line = max(1, line_number - context_lines)
            end_line = min(len(lines), line_number + context_lines)
            
            # 提取上下文
            context_content = ''.join(lines[start_line-1:end_line])
            
            return {
                'file_path': file_path,
                'start_line': start_line,
                'end_line': end_line,
                'target_line': line_number,
                'content': context_content
            }
        except Exception as e:
            logger.error(f"获取代码上下文失败 {file_path}:{line_number}: {str(e)}")
            return {
                'file_path': file_path,
                'start_line': line_number,
                'end_line': line_number,
                'target_line': line_number,
                'content': '',
                'error': str(e)
            }
    
    def get_file_overview(self, file_path: str) -> Dict[str, Any]:
        """
        获取文件概览
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件概览字典
        """
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # 获取文件大小
            file_size = os.path.getsize(file_path)
            
            # 获取行数
            lines = content.split('\n')
            line_count = len(lines)
            
            # 获取文件扩展名
            _, ext = os.path.splitext(file_path)
            
            # 提取文件头部（最多100行）
            header = '\n'.join(lines[:min(100, line_count)])
            
            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': file_size,
                'line_count': line_count,
                'extension': ext,
                'header': header
            }
        except Exception as e:
            logger.error(f"获取文件概览失败 {file_path}: {str(e)}")
            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'error': str(e)
            }