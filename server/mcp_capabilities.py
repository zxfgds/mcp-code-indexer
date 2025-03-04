"""
MCP协议能力接口模块
提供符合MCP协议的能力描述和接口
"""

import os
import logging
from typing import Dict, Any, List, Optional
from flask import Flask, jsonify, request, current_app

logger = logging.getLogger(__name__)

def setup_mcp_routes(app: Flask) -> None:
    """
    设置MCP协议路由
    
    Args:
        app: Flask应用实例
        
    Returns:
        无返回值
    """
    # MCP协议版本
    MCP_VERSION = "0.1"
    
    # MCP协议根路径
    @app.route('/mcp', methods=['GET'])
    def mcp_root():
        """
        MCP协议根路径
        
        Returns:
            MCP协议信息
        """
        return jsonify({
            "name": "MCP Code Indexer",
            "version": MCP_VERSION,
            "description": "基于MCP协议的代码检索工具，为AI大语言模型提供高效、准确的代码库检索能力",
            "capabilities": [
                "tools",
                "resources"
            ]
        })
    
    # MCP协议能力接口
    @app.route('/mcp/capabilities', methods=['GET'])
    def mcp_capabilities():
        """
        MCP协议能力接口
        
        Returns:
            MCP协议能力描述
        """
        return jsonify({
            "version": MCP_VERSION,
            "capabilities": {
                "tools": {
                    "list": "/mcp/tools",
                    "call": "/mcp/tools/call"
                },
                "resources": {
                    "list": "/mcp/resources",
                    "templates": "/mcp/resource-templates",
                    "read": "/mcp/resources/read"
                }
            }
        })
    
    # MCP工具列表接口
    @app.route('/mcp/tools', methods=['GET'])
    def mcp_tools():
        """
        MCP工具列表接口
        
        Returns:
            MCP工具列表
        """
        return jsonify({
            "tools": [
                {
                    "name": "identify_project",
                    "description": "识别代码项目，返回项目ID和状态",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "project_path": {
                                "type": "string",
                                "description": "项目路径"
                            }
                        },
                        "required": ["project_path"]
                    }
                },
                {
                    "name": "index_project",
                    "description": "索引代码项目，生成向量索引",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "project_path": {
                                "type": "string",
                                "description": "项目路径"
                            },
                            "wait": {
                                "type": "boolean",
                                "description": "是否等待索引完成",
                                "default": false
                            }
                        },
                        "required": ["project_path"]
                    }
                },
                {
                    "name": "search_code",
                    "description": "搜索代码，返回相关代码片段",
                    "inputSchema": {
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
                            "language": {
                                "type": "string",
                                "description": "编程语言过滤"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "返回结果数量限制",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_project_status",
                    "description": "获取项目索引状态",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "project_id": {
                                "type": "string",
                                "description": "项目ID"
                            }
                        },
                        "required": ["project_id"]
                    }
                },
                {
                    "name": "get_projects",
                    "description": "获取所有已索引的项目列表",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "get_code_context",
                    "description": "获取代码上下文",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "文件路径"
                            },
                            "line_number": {
                                "type": "integer",
                                "description": "行号"
                            },
                            "context_lines": {
                                "type": "integer",
                                "description": "上下文行数",
                                "default": 10
                            }
                        },
                        "required": ["file_path", "line_number"]
                    }
                }
            ]
        })
    
    # MCP工具调用接口
    @app.route('/mcp/tools/call', methods=['POST'])
    def mcp_call_tool():
        """
        MCP工具调用接口
        
        Returns:
            工具调用结果
        """
        data = request.json
        if not data or 'name' not in data or 'arguments' not in data:
            return jsonify({
                "error": "Invalid request",
                "message": "Missing name or arguments"
            }), 400
        
        tool_name = data['name']
        arguments = data['arguments']
        
        # 获取组件
        indexer = current_app.config['mcp_indexer']
        search_engine = current_app.config['mcp_search_engine']
        formatter = current_app.config['mcp_formatter']
        
        try:
            # 根据工具名称调用相应的功能
            if tool_name == "identify_project":
                if 'project_path' not in arguments:
                    return jsonify({
                        "error": "Invalid arguments",
                        "message": "Missing project_path"
                    }), 400
                
                project_id, is_new, metadata = indexer.project_identifier.identify_project(arguments['project_path'])
                status, progress = indexer.get_indexing_status(project_id)
                
                return jsonify({
                    "content": [
                        {
                            "type": "text",
                            "text": f"项目识别成功。项目ID: {project_id}, 状态: {status}, 进度: {progress:.1%}"
                        }
                    ]
                })
            
            elif tool_name == "index_project":
                if 'project_path' not in arguments:
                    return jsonify({
                        "error": "Invalid arguments",
                        "message": "Missing project_path"
                    }), 400
                
                wait = arguments.get('wait', False)
                
                if wait:
                    # 识别项目
                    project_id, _, _ = indexer.project_identifier.identify_project(arguments['project_path'])
                    
                    # 启动索引
                    indexer.index_project(arguments['project_path'])
                    
                    # 等待索引完成
                    import time
                    max_wait = 300  # 最多等待5分钟
                    start_time = time.time()
                    while time.time() - start_time < max_wait:
                        status, progress = indexer.get_indexing_status(project_id)
                        if status == "completed" or status == "failed":
                            break
                        time.sleep(2)
                    
                    return jsonify({
                        "content": [
                            {
                                "type": "text",
                                "text": f"项目索引{status}。项目ID: {project_id}, 进度: {progress:.1%}"
                            }
                        ]
                    })
                else:
                    # 启动索引但不等待
                    project_id = indexer.index_project(arguments['project_path'])
                    
                    return jsonify({
                        "content": [
                            {
                                "type": "text",
                                "text": f"项目索引已启动。项目ID: {project_id}"
                            }
                        ]
                    })
            
            elif tool_name == "search_code":
                if 'query' not in arguments:
                    return jsonify({
                        "error": "Invalid arguments",
                        "message": "Missing query"
                    }), 400
                
                # 构建过滤条件
                filters = {}
                if 'language' in arguments:
                    filters['language'] = arguments['language']
                
                # 构建项目ID列表
                project_ids = [arguments['project_id']] if 'project_id' in arguments else None
                
                # 执行搜索
                limit = arguments.get('limit', 10)
                results = search_engine.search(arguments['query'], project_ids, filters, limit)
                
                # 格式化结果
                formatted_results = []
                for i, result in enumerate(results):
                    file_path = result.get('file_path', '')
                    language = result.get('language', 'text')
                    start_line = result.get('start_line', 1)
                    end_line = result.get('end_line', 1)
                    content = result.get('content', '')
                    
                    formatted_results.append(f"### {i+1}. {os.path.basename(file_path)} (行 {start_line}-{end_line})")
                    formatted_results.append(f"文件: {file_path}")
                    formatted_results.append(f"```{language}")
                    formatted_results.append(content)
                    formatted_results.append("```")
                    formatted_results.append("")
                
                if not formatted_results:
                    formatted_results = ["未找到匹配的代码。"]
                
                return jsonify({
                    "content": [
                        {
                            "type": "text",
                            "text": "\n".join(formatted_results)
                        }
                    ]
                })
            
            elif tool_name == "get_project_status":
                if 'project_id' not in arguments:
                    return jsonify({
                        "error": "Invalid arguments",
                        "message": "Missing project_id"
                    }), 400
                
                status, progress = indexer.get_indexing_status(arguments['project_id'])
                
                return jsonify({
                    "content": [
                        {
                            "type": "text",
                            "text": f"项目状态: {status}, 进度: {progress:.1%}"
                        }
                    ]
                })
            
            elif tool_name == "get_projects":
                projects = indexer.get_indexed_projects()
                
                if not projects:
                    return jsonify({
                        "content": [
                            {
                                "type": "text",
                                "text": "没有已索引的项目。"
                            }
                        ]
                    })
                
                formatted_projects = ["已索引的项目:"]
                for i, project in enumerate(projects):
                    project_id = project.get('project_id', '')
                    project_path = project.get('project_path', '')
                    status = project.get('status', '')
                    
                    formatted_projects.append(f"{i+1}. ID: {project_id}")
                    formatted_projects.append(f"   路径: {project_path}")
                    formatted_projects.append(f"   状态: {status}")
                    formatted_projects.append("")
                
                return jsonify({
                    "content": [
                        {
                            "type": "text",
                            "text": "\n".join(formatted_projects)
                        }
                    ]
                })
            
            elif tool_name == "get_code_context":
                if 'file_path' not in arguments or 'line_number' not in arguments:
                    return jsonify({
                        "error": "Invalid arguments",
                        "message": "Missing file_path or line_number"
                    }), 400
                
                context_lines = arguments.get('context_lines', 10)
                context = search_engine.get_code_context(
                    arguments['file_path'],
                    arguments['line_number'],
                    context_lines
                )
                
                file_path = context.get('file_path', '')
                start_line = context.get('start_line', 1)
                end_line = context.get('end_line', 1)
                target_line = context.get('target_line', 1)
                content = context.get('content', '')
                
                language = search_engine._guess_language(file_path)
                
                return jsonify({
                    "content": [
                        {
                            "type": "text",
                            "text": f"文件: {file_path} (行 {start_line}-{end_line}, 目标行: {target_line})\n\n```{language}\n{content}\n```"
                        }
                    ]
                })
            
            else:
                return jsonify({
                    "error": "Unknown tool",
                    "message": f"Tool not found: {tool_name}"
                }), 404
                
        except Exception as e:
            logger.error(f"工具调用失败: {str(e)}")
            return jsonify({
                "content": [
                    {
                        "type": "text",
                        "text": f"工具调用失败: {str(e)}"
                    }
                ],
                "isError": True
            })
    
    # MCP资源列表接口
    @app.route('/mcp/resources', methods=['GET'])
    def mcp_resources():
        """
        MCP资源列表接口
        
        Returns:
            MCP资源列表
        """
        # 获取组件
        indexer = current_app.config['mcp_indexer']
        
        # 获取所有项目
        projects = indexer.get_indexed_projects()
        
        resources = []
        for project in projects:
            project_id = project.get('project_id', '')
            project_path = project.get('project_path', '')
            
            resources.append({
                "uri": f"code://{project_id}/info",
                "name": f"项目信息: {os.path.basename(project_path)}",
                "description": f"项目 {project_path} 的基本信息",
                "mimeType": "application/json"
            })
        
        return jsonify({
            "resources": resources
        })
    
    # MCP资源模板接口
    @app.route('/mcp/resource-templates', methods=['GET'])
    def mcp_resource_templates():
        """
        MCP资源模板接口
        
        Returns:
            MCP资源模板列表
        """
        return jsonify({
            "resourceTemplates": [
                {
                    "uriTemplate": "code://{project_id}/info",
                    "name": "项目信息",
                    "description": "获取项目的基本信息",
                    "mimeType": "application/json"
                },
                {
                    "uriTemplate": "code://{project_id}/file/{file_path}",
                    "name": "文件内容",
                    "description": "获取项目中指定文件的内容",
                    "mimeType": "text/plain"
                }
            ]
        })
    
    # MCP资源读取接口
    @app.route('/mcp/resources/read', methods=['POST'])
    def mcp_read_resource():
        """
        MCP资源读取接口
        
        Returns:
            资源内容
        """
        data = request.json
        if not data or 'uri' not in data:
            return jsonify({
                "error": "Invalid request",
                "message": "Missing uri"
            }), 400
        
        uri = data['uri']
        
        # 获取组件
        indexer = current_app.config['mcp_indexer']
        search_engine = current_app.config['mcp_search_engine']
        
        try:
            # 解析URI
            if uri.startswith("code://"):
                # 项目信息: code://{project_id}/info
                if uri.endswith("/info"):
                    project_id = uri[7:-5]  # 去掉"code://"和"/info"
                    
                    # 获取项目信息
                    project = indexer.project_identifier.get_project_by_id(project_id)
                    if not project:
                        return jsonify({
                            "error": "Resource not found",
                            "message": f"Project not found: {project_id}"
                        }), 404
                    
                    return jsonify({
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps(project, indent=2)
                            }
                        ]
                    })
                
                # 文件内容: code://{project_id}/file/{file_path}
                elif "/file/" in uri:
                    parts = uri[7:].split("/file/", 1)
                    if len(parts) != 2:
                        return jsonify({
                            "error": "Invalid URI",
                            "message": f"Invalid file URI: {uri}"
                        }), 400
                    
                    project_id, file_path = parts
                    
                    # 获取项目信息
                    project = indexer.project_identifier.get_project_by_id(project_id)
                    if not project:
                        return jsonify({
                            "error": "Resource not found",
                            "message": f"Project not found: {project_id}"
                        }), 404
                    
                    # 构建完整文件路径
                    project_path = project.get("project_path", "")
                    full_path = os.path.join(project_path, file_path)
                    
                    # 读取文件内容
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                        
                        return jsonify({
                            "contents": [
                                {
                                    "uri": uri,
                                    "mimeType": "text/plain",
                                    "text": content
                                }
                            ]
                        })
                    except Exception as e:
                        return jsonify({
                            "error": "File read error",
                            "message": f"Failed to read file: {str(e)}"
                        }), 500
            
            return jsonify({
                "error": "Invalid URI",
                "message": f"Unsupported URI: {uri}"
            }), 400
                
        except Exception as e:
            logger.error(f"资源读取失败: {str(e)}")
            return jsonify({
                "error": "Resource read error",
                "message": str(e)
            }), 500
    
    logger.info("MCP协议路由设置完成")