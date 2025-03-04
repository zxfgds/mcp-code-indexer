"""
API路由模块
定义MCP代码检索服务的API路由
"""

import os
import logging
from typing import Dict, Any, List, Optional
from flask import Flask, jsonify, request, current_app

logger = logging.getLogger(__name__)

def setup_routes(app: Flask) -> None:
    """
    设置API路由
    
    Args:
        app: Flask应用实例
        
    Returns:
        无返回值
    """
    # 健康检查
    @app.route('/health', methods=['GET'])
    def health_check():
        """
        健康检查接口
        
        Returns:
            健康状态响应
        """
        return jsonify({
            'status': 'ok',
            'version': '0.1.0'
        })
    
    # 项目识别
    @app.route('/api/project/identify', methods=['POST'])
    def identify_project():
        """
        项目识别接口
        
        请求体:
        {
            "project_path": "项目路径"
        }
        
        Returns:
            项目识别结果
        """
        data = request.json
        if not data or 'project_path' not in data:
            return jsonify({
                'error': '无效请求',
                'message': '缺少项目路径'
            }), 400
        
        project_path = data['project_path']
        if not os.path.exists(project_path):
            return jsonify({
                'error': '无效的项目路径',
                'message': f'路径不存在：{project_path}'
            }), 400
        
        try:
            # 获取组件
            indexer = current_app.config['mcp_indexer']
            formatter = current_app.config['mcp_formatter']
            
            # 识别项目
            project_id, is_new, metadata = indexer.project_identifier.identify_project(project_path)
            
            # 获取索引状态
            status, progress = indexer.get_indexing_status(project_id)
            
            # 格式化响应
            response = formatter.format_project_info({
                'project_id': project_id,
                'project_path': project_path,
                'is_new': is_new,
                'status': status,
                'progress': progress,
                'metadata': metadata
            })
            
            return jsonify(response)
        except Exception as e:
            logger.error(f"项目识别失败: {str(e)}")
            return jsonify({
                'error': '项目识别失败',
                'message': str(e)
            }), 500
    
    # 索引项目
    @app.route('/api/project/index', methods=['POST'])
    def index_project():
        """
        索引项目接口
        
        请求体:
        {
            "project_path": "项目路径"
        }
        
        Returns:
            索引启动结果
        """
        data = request.json
        if not data or 'project_path' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Missing project_path'
            }), 400
        
        project_path = data['project_path']
        if not os.path.exists(project_path):
            return jsonify({
                'error': 'Invalid project path',
                'message': f'Path does not exist: {project_path}'
            }), 400
        
        try:
            # 获取组件
            indexer = current_app.config['mcp_indexer']
            formatter = current_app.config['mcp_formatter']
            
            # 启动索引
            project_id = indexer.index_project(project_path)
            
            # 获取索引状态
            status, progress = indexer.get_indexing_status(project_id)
            
            # 格式化响应
            response = formatter.format_indexing_status(
                project_id, status, progress, 
                "索引启动成功"
            )
            
            return jsonify(response)
        except Exception as e:
            logger.error(f"启动索引失败: {str(e)}")
            return jsonify({
                'error': '索引失败',
                'message': str(e)
            }), 500
    
    # 获取索引状态
    @app.route('/api/project/status/<project_id>', methods=['GET'])
    def get_project_status(project_id):
        """
        获取项目索引状态接口
        
        Args:
            project_id: 项目ID
        
        Returns:
            项目索引状态
        """
        try:
            # 获取组件
            indexer = current_app.config['mcp_indexer']
            formatter = current_app.config['mcp_formatter']
            
            # 获取索引状态
            status, progress = indexer.get_indexing_status(project_id)
            
            # 格式化响应
            response = formatter.format_indexing_status(
                project_id, status, progress
            )
            
            return jsonify(response)
        except Exception as e:
            logger.error(f"获取索引状态失败: {str(e)}")
            return jsonify({
                'error': '获取状态失败',
                'message': str(e)
            }), 500
    
    # 搜索代码
    @app.route('/api/search', methods=['POST'])
    def search_code():
        """
        搜索代码接口
        
        请求体:
        {
            "query": "查询字符串",
            "project_ids": ["项目ID1", "项目ID2"], // 可选
            "filters": {                          // 可选
                "language": "python",
                "file_path": "src"
            },
            "limit": 10                           // 可选
        }
        
        Returns:
            搜索结果
        """
        data = request.json
        if not data or 'query' not in data:
            return jsonify({
                'error': '无效请求',
                'message': '缺少查询参数'
            }), 400
        
        query = data['query']
        project_ids = data.get('project_ids')
        filters = data.get('filters')
        limit = data.get('limit', 10)
        
        try:
            # 获取组件
            search_engine = current_app.config['mcp_search_engine']
            formatter = current_app.config['mcp_formatter']
            
            # 执行搜索
            results = search_engine.search(query, project_ids, filters, limit)
            
            # 格式化响应
            response = formatter.format_search_results(results, query)
            
            return jsonify(response)
        except Exception as e:
            logger.error(f"搜索代码失败: {str(e)}")
            return jsonify({
                'error': '搜索失败',
                'message': str(e)
            }), 500
    
    # 获取代码上下文
    @app.route('/api/context', methods=['POST'])
    def get_code_context():
        """
        获取代码上下文接口
        
        请求体:
        {
            "file_path": "文件路径",
            "line_number": 10,
            "context_lines": 5  // 可选
        }
        
        Returns:
            代码上下文
        """
        data = request.json
        if not data or 'file_path' not in data or 'line_number' not in data:
            return jsonify({
                'error': '无效请求',
                'message': '缺少文件路径或行号'
            }), 400
        
        file_path = data['file_path']
        line_number = data['line_number']
        context_lines = data.get('context_lines', 10)
        
        if not os.path.exists(file_path):
            return jsonify({
                'error': '无效的文件路径',
                'message': f'文件不存在：{file_path}'
            }), 400
        
        try:
            # 获取组件
            search_engine = current_app.config['mcp_search_engine']
            formatter = current_app.config['mcp_formatter']
            
            # 获取代码上下文
            context = search_engine.get_code_context(file_path, line_number, context_lines)
            
            # 获取相关代码
            related_blocks = search_engine.get_related_code(context, 5)
            
            # 格式化响应
            response = formatter.format_code_context(context, related_blocks)
            
            return jsonify(response)
        except Exception as e:
            logger.error(f"获取代码上下文失败: {str(e)}")
            return jsonify({
                'error': '获取代码上下文失败',
                'message': str(e)
            }), 500
    
    # 获取项目列表
    @app.route('/api/projects', methods=['GET'])
    def get_projects():
        """
        获取项目列表接口
        
        Returns:
            项目列表
        """
        try:
            # 获取组件
            indexer = current_app.config['mcp_indexer']
            
            # 获取项目列表
            projects = indexer.get_indexed_projects()
            
            return jsonify({
                'projects': projects,
                'count': len(projects)
            })
        except Exception as e:
            logger.error(f"获取项目列表失败: {str(e)}")
            return jsonify({
                'error': '获取项目列表失败',
                'message': str(e)
            }), 500
    
    # 删除项目索引
    @app.route('/api/project/<project_id>', methods=['DELETE'])
    def delete_project(project_id):
        """
        删除项目索引接口
        
        Args:
            project_id: 项目ID
        
        Returns:
            删除结果
        """
        try:
            # 获取组件
            indexer = current_app.config['mcp_indexer']
            
            # 删除项目索引
            success = indexer.delete_project_index(project_id)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'项目 {project_id} 删除成功'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': f'删除项目 {project_id} 失败'
                }), 400
        except Exception as e:
            logger.error(f"删除项目索引失败: {str(e)}")
            return jsonify({
                'error': '删除项目失败',
                'message': str(e)
            }), 500
    
    logger.info("API路由设置完成")