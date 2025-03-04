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
            
    def find_similar_code(self, code: str, language: str = None,
                         threshold: float = 0.7, limit: int = 5) -> List[Dict[str, Any]]:
        """
        查找相似代码片段，使用改进的相似度算法
        
        Args:
            code: 代码片段
            language: 编程语言（可选）
            threshold: 相似度阈值（0-1之间）
            limit: 返回结果数量限制
            
        Returns:
            相似代码片段列表，每个元素包含:
            - content: 代码内容
            - file_path: 文件路径
            - start_line: 起始行
            - end_line: 结束行
            - similarity: 相似度分数
            - matched_lines: 匹配的行
            - similarity_details: 相似度详细信息
        """
        try:
            # 标准化代码，传入语言参数
            normalized_code = self._normalize_code(code, language)
            
            # 使用代码内容作为查询
            results = []
            indexed_projects = self.indexer.get_indexed_projects()
            
            # 记录开始时间，用于性能监控
            import time
            start_time = time.time()
            
            # 首先使用向量搜索找到候选结果
            candidate_results = []
            for project in indexed_projects:
                project_id = project.get("project_id")
                if not project_id:
                    continue
                    
                # 搜索相似代码，增加搜索范围以提高召回率
                search_results = self.indexer.search(project_id, normalized_code, limit=limit*3)
                candidate_results.extend(search_results)
            
            # 对每个候选结果计算详细相似度
            for result in candidate_results:
                # 获取结果的语言
                result_language = result.get('language', language)
                
                # 标准化结果代码，传入语言参数
                result_code = self._normalize_code(result.get('content', ''), result_language)
                
                # 计算详细相似度，传入语言参数
                similarity_info = self._calculate_detailed_similarity(
                    normalized_code,
                    result_code,
                    result_language
                )
                
                # 如果相似度超过阈值，添加到结果中
                if similarity_info['overall_similarity'] >= threshold:
                    # 添加相似度信息
                    result.update({
                        'similarity': similarity_info['overall_similarity'],
                        'matched_lines': similarity_info['matched_lines'],
                        'similarity_details': {
                            'line_similarity': similarity_info['line_similarity'],
                            'structure_similarity': similarity_info['structure_similarity'],
                            'semantic_similarity': similarity_info['semantic_similarity'],
                            'control_flow_similarity': similarity_info['control_flow_similarity']
                        }
                    })
                    results.append(result)
            
            # 按相似度排序
            results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            # 如果指定了语言，过滤结果
            if language:
                results = [r for r in results if r.get('language') == language]
            
            # 记录搜索时间
            search_time = time.time() - start_time
            logger.info(f"相似代码搜索完成，耗时: {search_time:.2f}秒，找到 {len(results)} 个结果")
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"查找相似代码失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
            
    def _normalize_code(self, code: str, language: str = None) -> str:
        """
        标准化代码，移除空白字符和注释，并进行语言特定的标准化
        
        Args:
            code: 原始代码
            language: 编程语言
            
        Returns:
            标准化后的代码
        """
        if not code:
            return ""
            
        # 移除注释
        lines = []
        in_multiline_comment = False
        in_python_docstring = False
        
        # 语言特定的注释标记
        single_line_comment_markers = {
            'python': ['#'],
            'javascript': ['//'],
            'typescript': ['//'],
            'java': ['//'],
            'c': ['//'],
            'cpp': ['//'],
            'csharp': ['//'],
            'php': ['//'],
            'ruby': ['#'],
            'go': ['//'],
            'rust': ['//'],
            'swift': ['//'],
            'kotlin': ['//']
        }
        
        # 获取当前语言的单行注释标记
        comment_markers = single_line_comment_markers.get(language, ['#', '//'])
        
        for line in code.split('\n'):
            processed_line = line.strip()
            
            # 处理Python文档字符串
            if language == 'python':
                if '"""' in processed_line or "'''" in processed_line:
                    # 计算三引号的数量
                    triple_double_count = processed_line.count('"""')
                    triple_single_count = processed_line.count("'''")
                    
                    # 如果数量为奇数，切换docstring状态
                    if triple_double_count % 2 == 1 or triple_single_count % 2 == 1:
                        in_python_docstring = not in_python_docstring
                        
                    # 移除docstring部分
                    if '"""' in processed_line:
                        parts = processed_line.split('"""')
                        processed_line = parts[0] if not in_python_docstring else parts[-1]
                    if "'''" in processed_line:
                        parts = processed_line.split("'''")
                        processed_line = parts[0] if not in_python_docstring else parts[-1]
                
                if in_python_docstring:
                    continue
            
            # 处理多行注释
            if '/*' in processed_line and not any(processed_line.startswith(m) for m in comment_markers):
                in_multiline_comment = True
                processed_line = processed_line[:processed_line.index('/*')].strip()
            if '*/' in processed_line:
                in_multiline_comment = False
                if processed_line.index('*/') + 2 < len(processed_line):
                    processed_line = processed_line[processed_line.index('*/')+2:].strip()
                else:
                    processed_line = ""
            if in_multiline_comment:
                continue
                
            # 处理单行注释
            for marker in comment_markers:
                if marker in processed_line:
                    # 确保不是字符串中的注释标记
                    parts = []
                    in_string = False
                    string_char = None
                    i = 0
                    
                    while i < len(processed_line):
                        if processed_line[i] in ['"', "'"]:
                            if not in_string:
                                in_string = True
                                string_char = processed_line[i]
                            elif processed_line[i] == string_char:
                                in_string = False
                        elif processed_line[i:i+len(marker)] == marker and not in_string:
                            processed_line = processed_line[:i].strip()
                            break
                        i += 1
            
            # 语言特定的标准化
            if language:
                # 移除变量类型声明（适用于静态类型语言）
                if language in ['java', 'c', 'cpp', 'csharp', 'typescript']:
                    # 简化类型声明，如 "int x = 5;" -> "x = 5;"
                    import re
                    processed_line = re.sub(r'\b(int|float|double|char|boolean|string|var|let|const|auto)\b\s+([a-zA-Z_][a-zA-Z0-9_]*)', r'\2', processed_line)
            
            # 移除多余空白字符
            processed_line = ' '.join(processed_line.split())
            
            if processed_line:
                lines.append(processed_line)
                
        return '\n'.join(lines)
        
    def _calculate_detailed_similarity(self, code1: str, code2: str, language: str = None) -> Dict[str, Any]:
        """
        计算详细的代码相似度，使用多种算法综合评估
        
        Args:
            code1: 第一段代码
            code2: 第二段代码
            language: 编程语言
            
        Returns:
            相似度详细信息
        """
        # 分割为行
        lines1 = code1.split('\n')
        lines2 = code2.split('\n')
        
        # 1. 使用最长公共子序列算法找到匹配行
        lcs_matrix = self._compute_lcs_matrix(lines1, lines2)
        matched_pairs = self._extract_lcs_matches(lcs_matrix, lines1, lines2)
        
        # 2. 计算行级别的匹配
        matched_lines = []
        total_line_similarity = 0
        
        for i, j in matched_pairs:
            # 计算行相似度
            similarity = self._calculate_line_similarity(lines1[i], lines2[j])
            
            if similarity >= 0.7:  # 行匹配阈值
                matched_lines.append({
                    'line1': i + 1,
                    'line2': j + 1,
                    'similarity': similarity
                })
                total_line_similarity += similarity
        
        # 3. 计算结构相似度
        structure_similarity = len(matched_pairs) / max(len(lines1), len(lines2)) if max(len(lines1), len(lines2)) > 0 else 0
        
        # 4. 计算语义相似度 (使用词袋模型)
        semantic_similarity = self._calculate_semantic_similarity(code1, code2)
        
        # 5. 计算控制流相似度
        control_flow_similarity = self._calculate_control_flow_similarity(code1, code2, language)
        
        # 6. 综合计算整体相似度 (加权平均)
        weights = {
            'line': 0.4,
            'structure': 0.3,
            'semantic': 0.2,
            'control_flow': 0.1
        }
        
        line_similarity_normalized = total_line_similarity / max(len(lines1), len(lines2)) if max(len(lines1), len(lines2)) > 0 else 0
        
        overall_similarity = (
            weights['line'] * line_similarity_normalized +
            weights['structure'] * structure_similarity +
            weights['semantic'] * semantic_similarity +
            weights['control_flow'] * control_flow_similarity
        )
        
        return {
            'overall_similarity': overall_similarity,
            'matched_lines': matched_lines,
            'line_similarity': line_similarity_normalized,
            'structure_similarity': structure_similarity,
            'semantic_similarity': semantic_similarity,
            'control_flow_similarity': control_flow_similarity
        }
    
    def _compute_lcs_matrix(self, lines1: List[str], lines2: List[str]) -> List[List[int]]:
        """计算最长公共子序列矩阵"""
        m, n = len(lines1), len(lines2)
        lcs_matrix = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if self._calculate_line_similarity(lines1[i-1], lines2[j-1]) >= 0.7:
                    lcs_matrix[i][j] = lcs_matrix[i-1][j-1] + 1
                else:
                    lcs_matrix[i][j] = max(lcs_matrix[i-1][j], lcs_matrix[i][j-1])
                    
        return lcs_matrix
    
    def _extract_lcs_matches(self, lcs_matrix: List[List[int]], lines1: List[str], lines2: List[str]) -> List[Tuple[int, int]]:
        """从LCS矩阵中提取匹配的行对"""
        matches = []
        i, j = len(lines1), len(lines2)
        
        while i > 0 and j > 0:
            if self._calculate_line_similarity(lines1[i-1], lines2[j-1]) >= 0.7 and lcs_matrix[i][j] == lcs_matrix[i-1][j-1] + 1:
                matches.append((i-1, j-1))
                i -= 1
                j -= 1
            elif lcs_matrix[i-1][j] >= lcs_matrix[i][j-1]:
                i -= 1
            else:
                j -= 1
                
        return matches[::-1]  # 反转以获得正序
        
    def _calculate_line_similarity(self, line1: str, line2: str) -> float:
        """
        计算两行代码的相似度，使用改进的算法
        
        Args:
            line1: 第一行代码
            line2: 第二行代码
            
        Returns:
            相似度分数 (0-1)
        """
        if not line1 and not line2:
            return 1.0
        if not line1 or not line2:
            return 0.0
            
        # 1. 标记化
        tokens1 = self._tokenize_code_line(line1)
        tokens2 = self._tokenize_code_line(line2)
        
        # 2. 计算Jaccard相似度
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard = intersection / union if union > 0 else 0
        
        # 3. 计算序列相似度 (考虑顺序)
        lcs_length = self._longest_common_subsequence(tokens1, tokens2)
        sequence_sim = lcs_length / max(len(tokens1), len(tokens2)) if max(len(tokens1), len(tokens2)) > 0 else 0
        
        # 4. 计算编辑距离相似度
        edit_distance = self._levenshtein_distance(tokens1, tokens2)
        max_length = max(len(tokens1), len(tokens2))
        edit_sim = 1 - (edit_distance / max_length) if max_length > 0 else 0
        
        # 5. 综合相似度 (加权平均)
        return 0.3 * jaccard + 0.4 * sequence_sim + 0.3 * edit_sim
    
    def _tokenize_code_line(self, line: str) -> List[str]:
        """将代码行分解为标记"""
        import re
        # 分割标识符、运算符、括号等
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[+\-*/=<>!&|^~%]+|[{}()\[\],.;:]', line)
        return tokens
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    
        return dp[m][n]
    
    def _levenshtein_distance(self, seq1: List[str], seq2: List[str]) -> int:
        """计算编辑距离"""
        m, n = len(seq1), len(seq2)
        
        # 创建距离矩阵
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化第一行和第一列
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        # 填充矩阵
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,      # 删除
                        dp[i][j-1] + 1,      # 插入
                        dp[i-1][j-1] + 1     # 替换
                    )
                    
        return dp[m][n]
    
    def _calculate_semantic_similarity(self, code1: str, code2: str) -> float:
        """计算代码的语义相似度"""
        # 提取所有标识符
        import re
        identifiers1 = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', code1))
        identifiers2 = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', code2))
        
        # 过滤掉关键字
        keywords = {'if', 'else', 'for', 'while', 'return', 'function', 'class', 'var', 'let', 'const', 'import', 'export'}
        identifiers1 = identifiers1 - keywords
        identifiers2 = identifiers2 - keywords
        
        # 计算标识符相似度
        if not identifiers1 and not identifiers2:
            return 1.0
        if not identifiers1 or not identifiers2:
            return 0.0
            
        intersection = len(identifiers1.intersection(identifiers2))
        union = len(identifiers1.union(identifiers2))
        
        return intersection / union
    
    def _calculate_control_flow_similarity(self, code1: str, code2: str, language: str = None) -> float:
        """计算控制流相似度"""
        # 提取控制流结构
        control_patterns = {
            'if': r'\bif\b',
            'else': r'\belse\b',
            'for': r'\bfor\b',
            'while': r'\bwhile\b',
            'switch': r'\bswitch\b',
            'case': r'\bcase\b',
            'try': r'\btry\b',
            'catch': r'\bcatch\b',
            'return': r'\breturn\b'
        }
        
        # 计算每种控制结构的出现次数
        import re
        control_counts1 = {pattern: len(re.findall(regex, code1)) for pattern, regex in control_patterns.items()}
        control_counts2 = {pattern: len(re.findall(regex, code2)) for pattern, regex in control_patterns.items()}
        
        # 计算控制结构分布的相似度
        total_structures1 = sum(control_counts1.values())
        total_structures2 = sum(control_counts2.values())
        
        if total_structures1 == 0 and total_structures2 == 0:
            return 1.0
        if total_structures1 == 0 or total_structures2 == 0:
            return 0.0
        
        # 计算每种控制结构的比例差异
        similarity = 0.0
        for pattern in control_patterns:
            ratio1 = control_counts1[pattern] / total_structures1 if total_structures1 > 0 else 0
            ratio2 = control_counts2[pattern] / total_structures2 if total_structures2 > 0 else 0
            similarity += 1.0 - abs(ratio1 - ratio2)
            
        return similarity / len(control_patterns)