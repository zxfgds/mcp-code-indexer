"""
AI插件接口模块
提供与AI大语言模型集成的插件接口
"""

import os
import sys
import json
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple, Callable
import time

logger = logging.getLogger(__name__)

class McpPlugin:
    """
    MCP插件类
    
    提供与AI大语言模型集成的插件接口，实现代码检索功能
    """
    
    def __init__(self, server_url: str = "http://127.0.0.1:5000"):
        """
        初始化MCP插件
        
        Args:
            server_url: MCP服务器URL
            
        Returns:
            无返回值
        """
        self.server_url = server_url.rstrip('/')
        logger.info(f"MCP插件初始化完成，服务器URL: {server_url}")
    
    def identify_project(self, project_path: str) -> Dict[str, Any]:
        """
        识别项目
        
        Args:
            project_path: 项目路径
            
        Returns:
            项目识别结果字典
        """
        url = f"{self.server_url}/api/project/identify"
        data = {"project_path": os.path.abspath(project_path)}
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"项目识别失败: {str(e)}")
            return {
                "error": True,
                "message": f"项目识别失败: {str(e)}"
            }
    
    def index_project(self, project_path: str, 
                     wait_complete: bool = False, 
                     timeout: int = 300) -> Dict[str, Any]:
        """
        索引项目
        
        Args:
            project_path: 项目路径
            wait_complete: 是否等待索引完成
            timeout: 等待超时时间（秒）
            
        Returns:
            索引结果字典
        """
        url = f"{self.server_url}/api/project/index"
        data = {"project_path": os.path.abspath(project_path)}
        
        try:
            # 启动索引
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            
            # 如果不等待完成，直接返回
            if not wait_complete:
                return result
            
            # 等待索引完成
            project_id = result.get("project_id")
            if not project_id:
                return result
            
            return self._wait_indexing_complete(project_id, timeout)
        except Exception as e:
            logger.error(f"索引项目失败: {str(e)}")
            return {
                "error": True,
                "message": f"索引项目失败: {str(e)}"
            }
    
    def _wait_indexing_complete(self, project_id: str, timeout: int) -> Dict[str, Any]:
        """
        等待索引完成
        
        Args:
            project_id: 项目ID
            timeout: 超时时间（秒）
            
        Returns:
            索引状态字典
        """
        url = f"{self.server_url}/api/project/status/{project_id}"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url)
                response.raise_for_status()
                result = response.json()
                
                status = result.get("indexing_status", {}).get("status")
                
                # 如果索引完成或失败，返回结果
                if status in ["completed", "failed"]:
                    return result
                
                # 等待一段时间再检查
                time.sleep(2)
            except Exception as e:
                logger.error(f"检查索引状态失败: {str(e)}")
                return {
                    "error": True,
                    "message": f"检查索引状态失败: {str(e)}"
                }
        
        # 超时
        return {
            "error": True,
            "message": f"索引超时: {timeout}秒"
        }
    
    def search(self, query: str, project_ids: Optional[List[str]] = None, 
              filters: Optional[Dict[str, Any]] = None, limit: int = 10) -> Dict[str, Any]:
        """
        搜索代码
        
        Args:
            query: 查询字符串
            project_ids: 项目ID列表，如果为None则搜索所有项目
            filters: 过滤条件，如语言、文件类型等
            limit: 返回结果数量限制
            
        Returns:
            搜索结果字典
        """
        url = f"{self.server_url}/api/search"
        data = {
            "query": query,
            "limit": limit
        }
        
        if project_ids:
            data["project_ids"] = project_ids
        
        if filters:
            data["filters"] = filters
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"搜索代码失败: {str(e)}")
            return {
                "error": True,
                "message": f"搜索代码失败: {str(e)}"
            }
    
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
        url = f"{self.server_url}/api/context"
        data = {
            "file_path": os.path.abspath(file_path),
            "line_number": line_number,
            "context_lines": context_lines
        }
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取代码上下文失败: {str(e)}")
            return {
                "error": True,
                "message": f"获取代码上下文失败: {str(e)}"
            }
    
    def get_projects(self) -> Dict[str, Any]:
        """
        获取项目列表
        
        Returns:
            项目列表字典
        """
        url = f"{self.server_url}/api/projects"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取项目列表失败: {str(e)}")
            return {
                "error": True,
                "message": f"获取项目列表失败: {str(e)}"
            }
    
    def delete_project(self, project_id: str) -> Dict[str, Any]:
        """
        删除项目索引
        
        Args:
            project_id: 项目ID
            
        Returns:
            删除结果字典
        """
        url = f"{self.server_url}/api/project/{project_id}"
        
        try:
            response = requests.delete(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"删除项目索引失败: {str(e)}")
            return {
                "error": True,
                "message": f"删除项目索引失败: {str(e)}"
            }
    
    def health_check(self) -> bool:
        """
        检查服务器健康状态
        
        Returns:
            如果服务器正常则返回True，否则返回False
        """
        url = f"{self.server_url}/health"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return True
        except:
            return False
    
    def format_for_ai(self, search_results: Dict[str, Any]) -> str:
        """
        将搜索结果格式化为适合AI的文本
        
        Args:
            search_results: 搜索结果字典
            
        Returns:
            格式化后的文本
        """
        if "error" in search_results:
            return f"Error: {search_results.get('message', 'Unknown error')}"
        
        code_blocks = search_results.get("code_blocks", [])
        if not code_blocks:
            return "No code found matching your query."
        
        # 构建格式化文本
        result = []
        result.append(f"Found {len(code_blocks)} code blocks matching your query:")
        result.append("")
        
        for i, block in enumerate(code_blocks, 1):
            file_path = block.get("file_path", "Unknown file")
            language = block.get("language", "text")
            start_line = block.get("start_line", 1)
            end_line = block.get("end_line", 1)
            content = block.get("content", "")
            confidence = block.get("confidence", 0.0)
            
            result.append(f"### {i}. {os.path.basename(file_path)} (lines {start_line}-{end_line}, confidence: {confidence:.2f})")
            result.append(f"File: {file_path}")
            result.append("```" + language)
            result.append(content)
            result.append("```")
            result.append("")
        
        return "\n".join(result)
    
    def process_query(self, query: str, current_dir: Optional[str] = None) -> str:
        """
        处理查询，自动识别项目并搜索
        
        Args:
            query: 查询字符串
            current_dir: 当前目录，如果为None则使用当前工作目录
            
        Returns:
            格式化后的搜索结果文本
        """
        # 确定项目目录
        project_dir = current_dir or os.getcwd()
        
        # 识别项目
        project_info = self.identify_project(project_dir)
        if "error" in project_info:
            return f"Error identifying project: {project_info.get('message', 'Unknown error')}"
        
        project_id = project_info.get("project_id")
        if not project_id:
            return "Failed to identify project."
        
        # 检查索引状态
        status = project_info.get("status", "")
        
        # 如果项目未索引，启动索引
        if status == "new":
            index_result = self.index_project(project_dir, wait_complete=True)
            if "error" in index_result:
                return f"Error indexing project: {index_result.get('message', 'Unknown error')}"
        
        # 搜索代码
        search_results = self.search(query, [project_id])
        
        # 格式化结果
        return self.format_for_ai(search_results)