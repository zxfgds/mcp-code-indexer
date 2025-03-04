"""
MCP服务器模块
实现符合MCP协议的服务器
"""

import os
import logging
from typing import Dict, Any, List, Optional
import json

from mcp.server.lowlevel import Server
from mcp.types import (
    ListToolsRequest,
    CallToolRequest,
    ListResourcesRequest,
    ListResourceTemplatesRequest,
    Tool,
    TextContent,
    Resource,
    ResourceTemplate
)

logger = logging.getLogger(__name__)

def setup_mcp_server(config, indexer, search_engine, formatter):
    """
    设置MCP服务器
    
    Args:
        config: 配置对象
        indexer: 代码索引器
        search_engine: 搜索引擎
        formatter: MCP响应格式化器
        
    Returns:
        MCP服务器实例
    """
    # 创建服务器实例
    server = Server(
        "mcp-code-indexer",
        version="0.1.0"
    )
    
    # 设置资源列表请求处理程序
    @server.list_resources()
    async def list_resources():
        # 返回空资源列表，因为我们的服务器不提供任何资源
        return []
    
    # 设置资源模板列表请求处理程序
    @server.list_resource_templates()
    async def list_resource_templates():
        # 返回空资源模板列表，因为我们的服务器不提供任何资源模板
        return []
    
    # 设置工具列表请求处理程序
    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="identify_project",
                description="识别代码项目，返回项目ID和状态",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "项目路径"
                        }
                    },
                    "required": ["project_path"]
                }
            ),
            Tool(
                name="index_project",
                description="索引代码项目，生成向量索引",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Project path"
                        },
                        "wait": {
                            "type": "boolean",
                            "description": "是否等待索引完成",
                            "default": False
                        }
                    },
                    "required": ["project_path"]
                }
            ),
            Tool(
                name="search_code",
                description="搜索代码，返回相关代码片段",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询字符串"
                        },
                        "project_id": {
                            "type": "string",
                            "description": "项目ID，如果不提供则搜索所有项目"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "返回结果数量的限制",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_code_structure",
                description="获取代码结构信息（类、函数、依赖关系）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "文件路径"
                        },
                        "language": {
                            "type": "string",
                            "description": "编程语言"
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            Tool(
                name="analyze_code_quality",
                description="分析代码质量指标",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "文件路径"
                        },
                        "language": {
                            "type": "string",
                            "description": "编程语言"
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            Tool(
                name="find_similar_code",
                description="查找相似代码片段",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "代码片段"
                        },
                        "language": {
                            "type": "string",
                            "description": "编程语言"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "返回结果数量限制",
                            "default": 5
                        }
                    },
                    "required": ["code"]
                }
            ),
            Tool(
                name="get_code_metrics",
                description="获取代码度量数据",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "文件路径"
                        },
                        "language": {
                            "type": "string",
                            "description": "编程语言"
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            Tool(
                name="analyze_dependencies",
                description="分析项目依赖关系",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "项目路径"
                        }
                    },
                    "required": ["project_path"]
                }
            )
        ]
    
    # 设置工具调用请求处理程序
    @server.call_tool()
    async def call_tool(name, args):
        
        if name == "identify_project":
            if "project_path" not in args:
                return [
                    TextContent(
                        type="text",
                        text="错误：缺少项目路径参数"
                    )
                ]
            
            project_id, is_new, metadata = indexer.project_identifier.identify_project(args["project_path"])
            status, progress = indexer.get_indexing_status(project_id)
            
            return [
                TextContent(
                    type="text",
                    text=f"Project identification successful. Project ID: {project_id}, Status: {status}, Progress: {progress:.1%}"
                )
            ]
        
        elif name == "index_project":
            if "project_path" not in args:
                return [
                    TextContent(
                        type="text",
                        text="Error: Missing project path parameter"
                    )
                ]
            
            wait = args.get("wait", False)
            project_id = indexer.index_project(args["project_path"])
            
            if wait:
                import time
                max_wait = 300  # Wait up to 5 minutes
                start_time = time.time()
                while time.time() - start_time < max_wait:
                    status, progress = indexer.get_indexing_status(project_id)
                    if status == "completed" or status == "failed":
                        break
                    time.sleep(2)
                
                return [
                    TextContent(
                        type="text",
                        text=f"Project indexing {status}. Project ID: {project_id}, Progress: {progress:.1%}"
                    )
                ]
            else:
                return [
                    TextContent(
                        type="text",
                        text=f"Project indexing started. Project ID: {project_id}"
                    )
                ]
        
        elif name == "search_code":
            if "query" not in args:
                return [
                    TextContent(
                        type="text",
                        text="错误：缺少查询参数"
                    )
                ]
            
            project_ids = [args["project_id"]] if "project_id" in args else None
            limit = args.get("limit", 10)
            
            results = search_engine.search(args["query"], project_ids, None, limit)
            
            if not results:
                return [
                    TextContent(
                        type="text",
                        text="未找到匹配的代码。"
                    )
                ]
            
            formatted_results = []
            for i, result in enumerate(results):
                file_path = result.get("file_path", "")
                language = result.get("language", "text")
                start_line = result.get("start_line", 1)
                end_line = result.get("end_line", 1)
                content = result.get("content", "")
                
                formatted_results.append(f"### {i+1}. {os.path.basename(file_path)} (lines {start_line}-{end_line})")
                formatted_results.append(f"File: {file_path}")
                formatted_results.append(f"```{language}")
                formatted_results.append(content)
                formatted_results.append("```")
                formatted_results.append("")
            
            return [
                TextContent(
                    type="text",
                    text="\n".join(formatted_results)
                )
            ]

        elif name == "get_code_structure":
            if "file_path" not in args:
                return [
                    TextContent(
                        type="text",
                        text="错误：缺少文件路径参数"
                    )
                ]
            
            try:
                with open(args["file_path"], 'r', encoding='utf-8') as f:
                    content = f.read()
                
                language = args.get("language", os.path.splitext(args["file_path"])[1][1:])
                analyzer = indexer.optimizer.analyzer
                
                analysis = analyzer.analyze_code(content, language)
                
                result = {
                    "functions": analysis["functions"],
                    "classes": analysis["classes"],
                    "imports": analysis["imports"],
                    "dependencies": analysis["dependencies"]
                }
                
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, ensure_ascii=False)
                    )
                ]
            except Exception as e:
                return [
                    TextContent(
                        type="text",
                        text=f"分析代码结构失败: {str(e)}"
                    )
                ]

        elif name == "analyze_code_quality":
            if "file_path" not in args:
                return [
                    TextContent(
                        type="text",
                        text="错误：缺少文件路径参数"
                    )
                ]
            
            try:
                with open(args["file_path"], 'r', encoding='utf-8') as f:
                    content = f.read()
                
                language = args.get("language", os.path.splitext(args["file_path"])[1][1:])
                optimizer = indexer.optimizer
                
                quality_metrics = optimizer.analyze_code_quality(content, args["file_path"], language)
                
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(quality_metrics, indent=2, ensure_ascii=False)
                    )
                ]
            except Exception as e:
                return [
                    TextContent(
                        type="text",
                        text=f"分析代码质量失败: {str(e)}"
                    )
                ]

        elif name == "find_similar_code":
            if "code" not in args:
                return [
                    TextContent(
                        type="text",
                        text="错误：缺少代码片段参数"
                    )
                ]
            
            try:
                code = args["code"]
                language = args.get("language", "text")
                limit = args.get("limit", 5)
                
                similar_code = search_engine.find_similar_code(code, language, limit)
                
                if not similar_code:
                    return [
                        TextContent(
                            type="text",
                            text="未找到相似代码。"
                        )
                    ]
                
                formatted_results = []
                for i, result in enumerate(similar_code):
                    file_path = result.get("file_path", "")
                    similarity = result.get("similarity", 0)
                    content = result.get("content", "")
                    
                    formatted_results.append(f"### {i+1}. 相似度: {similarity:.2%}")
                    formatted_results.append(f"File: {file_path}")
                    formatted_results.append(f"```{language}")
                    formatted_results.append(content)
                    formatted_results.append("```")
                    formatted_results.append("")
                
                return [
                    TextContent(
                        type="text",
                        text="\n".join(formatted_results)
                    )
                ]
            except Exception as e:
                return [
                    TextContent(
                        type="text",
                        text=f"查找相似代码失败: {str(e)}"
                    )
                ]

        elif name == "get_code_metrics":
            if "file_path" not in args:
                return [
                    TextContent(
                        type="text",
                        text="错误：缺少文件路径参数"
                    )
                ]
            
            try:
                with open(args["file_path"], 'r', encoding='utf-8') as f:
                    content = f.read()
                
                language = args.get("language", os.path.splitext(args["file_path"])[1][1:])
                optimizer = indexer.optimizer
                
                metrics = optimizer.get_code_metrics(content, args["file_path"], language)
                
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(metrics, indent=2, ensure_ascii=False)
                    )
                ]
            except Exception as e:
                return [
                    TextContent(
                        type="text",
                        text=f"获取代码度量数据失败: {str(e)}"
                    )
                ]

        elif name == "analyze_dependencies":
            if "project_path" not in args:
                return [
                    TextContent(
                        type="text",
                        text="错误：缺少项目路径参数"
                    )
                ]
            
            try:
                dependencies = indexer.optimizer.analyze_project_dependencies(args["project_path"])
                
                # 将set类型转换为list，以便JSON序列化
                def convert_sets_to_lists(obj):
                    if isinstance(obj, dict):
                        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
                    elif isinstance(obj, set):
                        return list(obj)
                    elif isinstance(obj, list):
                        return [convert_sets_to_lists(item) for item in obj]
                    else:
                        return obj
                
                serializable_dependencies = convert_sets_to_lists(dependencies)
                
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(serializable_dependencies, indent=2, ensure_ascii=False)
                    )
                ]
            except Exception as e:
                return [
                    TextContent(
                        type="text",
                        text=f"分析项目依赖关系失败: {str(e)}"
                    )
                ]
        
        else:
            return [
                TextContent(
                    type="text",
                    text=f"未知工具：{name}"
                )
            ]
    
    return server
