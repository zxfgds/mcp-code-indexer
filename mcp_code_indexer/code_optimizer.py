"""
代码优化器模块
提供代码分析、优化和压缩功能
"""

import os
import math
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import re

from tree_sitter import Node, Tree
from .code_analyzer import CodeAnalyzer

logger = logging.getLogger(__name__)

class CodeBlockType(Enum):
    """代码块类型"""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    IMPORT = "import"
    COMMENT = "comment"
    CODE = "code"

@dataclass
class CodeBlock:
    """代码块数据类"""
    content: str
    block_type: CodeBlockType
    start_line: int
    end_line: int
    importance_score: float
    complexity_score: float
    dependencies: Set[str]
    symbols: Set[str]

class ASTCache:
    """AST缓存管理器"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化AST缓存
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".mcp_cache" / "ast"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, Tuple[float, Tree]] = {}
        
    def get_cache_key(self, file_path: str, content: str) -> str:
        """
        生成缓存键
        
        Args:
            file_path: 文件路径
            content: 文件内容
            
        Returns:
            缓存键
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{file_path}:{content_hash}"
    
    def get(self, file_path: str, content: str) -> Optional[Tree]:
        """
        获取AST缓存
        
        Args:
            file_path: 文件路径
            content: 文件内容
            
        Returns:
            缓存的AST树
        """
        cache_key = self.get_cache_key(file_path, content)
        
        # 检查内存缓存
        if cache_key in self.cache:
            mtime, tree = self.cache[cache_key]
            if os.path.getmtime(file_path) <= mtime:
                return tree
        
        # 检查文件缓存
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if os.path.getmtime(file_path) <= data['mtime']:
                        return self._deserialize_tree(data['tree'])
            except:
                pass
                
        return None
    
    def put(self, file_path: str, content: str, tree: Tree) -> None:
        """
        存储AST缓存
        
        Args:
            file_path: 文件路径
            content: 文件内容
            tree: AST树
        """
        cache_key = self.get_cache_key(file_path, content)
        mtime = os.path.getmtime(file_path)
        
        # 更新内存缓存
        self.cache[cache_key] = (mtime, tree)
        
        # 更新文件缓存
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'mtime': mtime,
                    'tree': self._serialize_tree(tree)
                }, f)
        except:
            logger.warning(f"Failed to save AST cache: {cache_file}")
    
    def _serialize_tree(self, tree: Tree) -> Dict:
        """序列化AST树"""
        # TODO: 实现AST树序列化
        pass
    
    def _deserialize_tree(self, data: Dict) -> Tree:
        """反序列化AST树"""
        # TODO: 实现AST树反序列化
        pass

class CodeOptimizer:
    """代码优化器类"""
    
    def __init__(self):
        """初始化代码优化器"""
        self.analyzer = CodeAnalyzer()
        self.ast_cache = ASTCache()
        
    def analyze_code(self, content: str, file_path: str, language: str) -> Dict[str, Any]:
        """
        分析代码
        
        Args:
            content: 代码内容
            file_path: 文件路径
            language: 编程语言
            
        Returns:
            分析结果
        """
        # 获取或解析AST
        tree = self.ast_cache.get(file_path, content)
        if not tree and language in self.analyzer.parsers:
            parser = self.analyzer.parsers[language]
            tree = parser.parse(content.encode())
            self.ast_cache.put(file_path, content, tree)
        
        if not tree:
            return self._simple_analysis(content)
        
        # 分析代码结构
        blocks = self._extract_code_blocks(tree.root_node, content)
        
        # 计算复杂度和重要性
        self._analyze_metrics(blocks)
        
        # 构建依赖图
        dependencies = self._analyze_dependencies(blocks)
        
        # 生成代码摘要
        summary = self._generate_summary(blocks)
        
        return {
            'blocks': blocks,
            'dependencies': dependencies,
            'summary': summary,
            'metrics': {
                'complexity': sum(b.complexity_score for b in blocks),
                'importance': sum(b.importance_score for b in blocks)
            }
        }
    
    def _extract_code_blocks(self, node: Node, content: str) -> List[CodeBlock]:
        """提取代码块"""
        blocks = []
        
        def visit(node: Node):
            if node.type == "function_definition":
                blocks.append(self._create_function_block(node, content))
            elif node.type == "class_definition":
                blocks.append(self._create_class_block(node, content))
            elif node.type == "method_definition":
                blocks.append(self._create_method_block(node, content))
            elif node.type == "comment":
                blocks.append(self._create_comment_block(node, content))
                
            for child in node.children:
                visit(child)
        
        visit(node)
        return blocks
    
    def _create_function_block(self, node: Node, content: str) -> CodeBlock:
        """创建函数代码块"""
        return CodeBlock(
            content=self._get_node_text(node, content),
            block_type=CodeBlockType.FUNCTION,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            importance_score=0.0,
            complexity_score=0.0,
            dependencies=set(),
            symbols=self._extract_symbols(node)
        )
    
    def _create_class_block(self, node: Node, content: str) -> CodeBlock:
        """创建类代码块"""
        return CodeBlock(
            content=self._get_node_text(node, content),
            block_type=CodeBlockType.CLASS,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            importance_score=0.0,
            complexity_score=0.0,
            dependencies=set(),
            symbols=self._extract_symbols(node)
        )
    
    def _create_method_block(self, node: Node, content: str) -> CodeBlock:
        """创建方法代码块"""
        return CodeBlock(
            content=self._get_node_text(node, content),
            block_type=CodeBlockType.METHOD,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            importance_score=0.0,
            complexity_score=0.0,
            dependencies=set(),
            symbols=self._extract_symbols(node)
        )
    
    def _create_comment_block(self, node: Node, content: str) -> CodeBlock:
        """创建注释代码块"""
        return CodeBlock(
            content=self._get_node_text(node, content),
            block_type=CodeBlockType.COMMENT,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            importance_score=0.0,
            complexity_score=0.0,
            dependencies=set(),
            symbols=set()
        )
    
    def _get_node_text(self, node: Node, content: str) -> str:
        """获取节点文本"""
        start_byte = node.start_byte
        end_byte = node.end_byte
        return content[start_byte:end_byte].decode('utf-8')
    
    def _extract_symbols(self, node: Node) -> Set[str]:
        """提取符号"""
        symbols = set()
        
        def visit(node: Node):
            if node.type in ["identifier", "variable", "property_identifier"]:
                symbols.add(node.text.decode('utf-8'))
            for child in node.children:
                visit(child)
                
        visit(node)
        return symbols
    
    def _analyze_metrics(self, blocks: List[CodeBlock]) -> None:
        """分析代码度量指标"""
        for block in blocks:
            # 计算圈复杂度
            block.complexity_score = self._calculate_complexity(block)
            # 计算重要性分数
            block.importance_score = self._calculate_importance(block)
    
    def _calculate_complexity(self, block: CodeBlock) -> float:
        """计算代码复杂度"""
        if block.block_type == CodeBlockType.COMMENT:
            return 0.0
            
        complexity = 1.0
        
        # 控制流复杂度
        control_patterns = [
            r'\bif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b',
            r'\btry\b', r'\bcatch\b', r'\bswitch\b', r'\bcase\b'
        ]
        for pattern in control_patterns:
            complexity += len(re.findall(pattern, block.content)) * 0.1
            
        # 符号复杂度
        complexity += len(block.symbols) * 0.05
        
        # 依赖复杂度
        complexity += len(block.dependencies) * 0.1
        
        return complexity
    
    def _calculate_importance(self, block: CodeBlock) -> float:
        """计算代码重要性"""
        if block.block_type == CodeBlockType.COMMENT:
            return 0.5 if "TODO" in block.content or "FIXME" in block.content else 0.1
            
        importance = 1.0
        
        # 基于类型的重要性
        type_weights = {
            CodeBlockType.CLASS: 2.0,
            CodeBlockType.FUNCTION: 1.5,
            CodeBlockType.METHOD: 1.5,
            CodeBlockType.IMPORT: 1.0,
            CodeBlockType.CODE: 1.0
        }
        importance *= type_weights.get(block.block_type, 1.0)
        
        # 基于依赖的重要性
        importance += len(block.dependencies) * 0.2
        
        # 基于符号的重要性
        importance += len(block.symbols) * 0.1
        
        return importance
    
    def _analyze_dependencies(self, blocks: List[CodeBlock]) -> Dict[str, Set[str]]:
        """分析代码依赖关系"""
        dependencies = {}
        symbol_to_block = {}
        
        # 构建符号到代码块的映射
        for block in blocks:
            for symbol in block.symbols:
                symbol_to_block[symbol] = block
        
        # 分析依赖关系
        for block in blocks:
            block_deps = set()
            for symbol in block.symbols:
                if symbol in symbol_to_block and symbol_to_block[symbol] != block:
                    dep_block = symbol_to_block[symbol]
                    block_deps.add(f"{dep_block.block_type.value}:{dep_block.start_line}")
            dependencies[f"{block.block_type.value}:{block.start_line}"] = block_deps
            
        return dependencies
    
    def _generate_summary(self, blocks: List[CodeBlock]) -> Dict[str, Any]:
        """生成代码摘要"""
        return {
            'block_count': len(blocks),
            'block_types': {
                block_type.value: len([b for b in blocks if b.block_type == block_type])
                for block_type in CodeBlockType
            },
            'important_blocks': [
                {
                    'type': block.block_type.value,
                    'start_line': block.start_line,
                    'importance': block.importance_score
                }
                for block in sorted(blocks, key=lambda b: b.importance_score, reverse=True)[:5]
            ],
            'complex_blocks': [
                {
                    'type': block.block_type.value,
                    'start_line': block.start_line,
                    'complexity': block.complexity_score
                }
                for block in sorted(blocks, key=lambda b: b.complexity_score, reverse=True)[:5]
            ]
        }
    
    def _simple_analysis(self, content: str) -> Dict[str, Any]:
        """简单的文本分析"""
        lines = content.split('\n')
        return {
            'blocks': [
                CodeBlock(
                    content=content,
                    block_type=CodeBlockType.CODE,
                    start_line=1,
                    end_line=len(lines),
                    importance_score=1.0,
                    complexity_score=1.0,
                    dependencies=set(),
                    symbols=set()
                )
            ],
            'dependencies': {},
            'summary': {
                'block_count': 1,
                'block_types': {CodeBlockType.CODE.value: 1},
                'important_blocks': [],
                'complex_blocks': []
            },
            'metrics': {
                'complexity': 1.0,
                'importance': 1.0
            }
        }
        
    def analyze_code_quality(self, content: str, file_path: str, language: str) -> Dict[str, Any]:
        """
        分析代码质量
        
        Args:
            content: 代码内容
            file_path: 文件路径
            language: 编程语言
            
        Returns:
            质量分析结果，包含:
            - complexity: 复杂度指标
            - maintainability: 可维护性指标
            - duplication: 代码重复信息
            - issues: 潜在问题
            - metrics: 代码度量数据
        """
        # 获取或解析AST
        tree = self.ast_cache.get(file_path, content)
        if not tree and language in self.analyzer.parsers:
            parser = self.analyzer.parsers[language]
            tree = parser.parse(content.encode())
            self.ast_cache.put(file_path, content, tree)
            
        if not tree:
            return self._simple_quality_analysis(content)
            
        try:
            # 提取代码块
            blocks = self._extract_code_blocks(tree.root_node, content)
            
            # 分析复杂度
            complexity = self._analyze_complexity(blocks)
            
            # 分析可维护性
            maintainability = self._analyze_maintainability(blocks, content)
            
            # 分析代码重复
            duplication = self._analyze_duplication(blocks)
            
            # 分析潜在问题
            issues = self._analyze_issues(blocks, content, language)
            
            # 计算代码度量
            metrics = self._calculate_metrics(blocks, content)
            
            return {
                'complexity': complexity,
                'maintainability': maintainability,
                'duplication': duplication,
                'issues': issues,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"代码质量分析失败: {str(e)}")
            return self._simple_quality_analysis(content)
            
    def _simple_quality_analysis(self, content: str) -> Dict[str, Any]:
        """简单的质量分析"""
        lines = content.split('\n')
        return {
            'complexity': {
                'cyclomatic': 1.0,
                'cognitive': 1.0
            },
            'maintainability': {
                'index': 100.0,
                'rating': 'A'
            },
            'duplication': {
                'blocks': [],
                'percentage': 0.0
            },
            'issues': [],
            'metrics': {
                'loc': len(lines),
                'sloc': len([l for l in lines if l.strip()]),
                'comments': len([l for l in lines if l.strip().startswith(('#', '//', '/*'))]),
                'functions': 0,
                'classes': 0
            }
        }
        
    def _analyze_complexity(self, blocks: List[CodeBlock]) -> Dict[str, Any]:
        """分析代码复杂度"""
        total_cyclomatic = 0
        total_cognitive = 0
        complex_blocks = []
        
        for block in blocks:
            if block.block_type in [CodeBlockType.FUNCTION, CodeBlockType.METHOD]:
                # 计算圈复杂度
                cyclomatic = self._calculate_cyclomatic_complexity(block)
                
                # 计算认知复杂度
                cognitive = self._calculate_cognitive_complexity(block)
                
                if cyclomatic > 10 or cognitive > 15:
                    complex_blocks.append({
                        'type': block.block_type.value,
                        'start_line': block.start_line,
                        'cyclomatic': cyclomatic,
                        'cognitive': cognitive
                    })
                    
                total_cyclomatic += cyclomatic
                total_cognitive += cognitive
                
        return {
            'cyclomatic': {
                'total': total_cyclomatic,
                'average': total_cyclomatic / len(blocks) if blocks else 0,
                'complex_blocks': complex_blocks
            },
            'cognitive': {
                'total': total_cognitive,
                'average': total_cognitive / len(blocks) if blocks else 0
            }
        }
        
    def _calculate_cyclomatic_complexity(self, block: CodeBlock) -> float:
        """计算圈复杂度"""
        complexity = 1.0  # 基础复杂度
        
        # 控制流语句
        control_patterns = [
            r'\bif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b',
            r'\btry\b', r'\bcatch\b', r'\bswitch\b', r'\bcase\b'
        ]
        
        for pattern in control_patterns:
            complexity += len(re.findall(pattern, block.content))
            
        # 逻辑运算符
        logic_patterns = [r'&&', r'\|\|']
        for pattern in logic_patterns:
            complexity += len(re.findall(pattern, block.content)) * 0.5
            
        return complexity
        
    def _calculate_cognitive_complexity(self, block: CodeBlock) -> float:
        """计算认知复杂度"""
        complexity = 0.0
        
        # 嵌套结构
        nesting_patterns = [
            (r'\bif\b', 1),
            (r'\bfor\b', 2),
            (r'\bwhile\b', 2),
            (r'\btry\b', 1),
            (r'\bcatch\b', 1),
            (r'\bswitch\b', 2)
        ]
        
        # 计算嵌套深度
        lines = block.content.split('\n')
        current_depth = 0
        for line in lines:
            # 增加深度
            if re.search(r'{', line):
                current_depth += 1
            # 减少深度
            if re.search(r'}', line):
                current_depth -= 1
            
            # 在当前深度下的复杂度
            for pattern, weight in nesting_patterns:
                if re.search(pattern, line):
                    complexity += weight * (1 + current_depth)
                    
        return complexity
        
    def _analyze_maintainability(self, blocks: List[CodeBlock], content: str) -> Dict[str, Any]:
        """分析可维护性"""
        # 计算Halstead度量
        halstead = self._calculate_halstead_metrics(content)
        
        # 计算代码行度量
        loc_metrics = self._calculate_loc_metrics(content)
        
        # 计算维护指数
        mi = 171 - 5.2 * math.log(halstead['volume']) - 0.23 * halstead['difficulty'] - 16.2 * math.log(loc_metrics['sloc'])
        mi = max(0, min(100, mi))  # 归一化到0-100
        
        return {
            'index': mi,
            'rating': self._get_maintainability_rating(mi),
            'factors': {
                'halstead': halstead,
                'loc': loc_metrics
            }
        }
        
    def _calculate_halstead_metrics(self, content: str) -> Dict[str, float]:
        """计算Halstead度量"""
        # 操作符和操作数
        operators = set()
        operands = set()
        
        # 简单的操作符识别
        operator_pattern = r'[+\-*/=<>!&|^~%]+'
        for match in re.finditer(operator_pattern, content):
            operators.add(match.group())
            
        # 简单的操作数识别（变量和常量）
        operand_pattern = r'\b[a-zA-Z_]\w*\b|\b\d+\b'
        for match in re.finditer(operand_pattern, content):
            operands.add(match.group())
            
        n1 = len(operators)  # 不同操作符数量
        n2 = len(operands)   # 不同操作数数量
        N1 = len(re.findall(operator_pattern, content))  # 操作符总数
        N2 = len(re.findall(operand_pattern, content))   # 操作数总数
        
        # 计算基本度量
        vocabulary = n1 + n2
        length = N1 + N2
        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (n1 * N2) / (2 * n2) if n2 > 0 else 0
        effort = volume * difficulty
        
        return {
            'vocabulary': vocabulary,
            'length': length,
            'volume': volume,
            'difficulty': difficulty,
            'effort': effort
        }
        
    def _calculate_loc_metrics(self, content: str) -> Dict[str, int]:
        """计算代码行度量"""
        lines = content.split('\n')
        
        sloc = 0    # 源代码行
        blank = 0   # 空行
        comments = 0  # 注释行
        
        in_block_comment = False
        
        for line in lines:
            line = line.strip()
            
            if not line:
                blank += 1
                continue
                
            if line.startswith('/*'):
                in_block_comment = True
                comments += 1
            elif line.endswith('*/'):
                in_block_comment = False
                comments += 1
            elif in_block_comment:
                comments += 1
            elif line.startswith('//') or line.startswith('#'):
                comments += 1
            else:
                sloc += 1
                
        return {
            'total': len(lines),
            'sloc': sloc,
            'blank': blank,
            'comments': comments
        }
        
    def _get_maintainability_rating(self, index: float) -> str:
        """获取可维护性等级"""
        if index >= 85:
            return 'A'
        elif index >= 70:
            return 'B'
        elif index >= 55:
            return 'C'
        elif index >= 40:
            return 'D'
        else:
            return 'F'
            
    def _analyze_duplication(self, blocks: List[CodeBlock]) -> Dict[str, Any]:
        """分析代码重复"""
        duplicates = []
        total_lines = 0
        duplicate_lines = 0
        
        # 对每个代码块进行比较
        for i, block1 in enumerate(blocks):
            total_lines += block1.end_line - block1.start_line + 1
            
            for j, block2 in enumerate(blocks[i+1:], i+1):
                # 计算相似度
                similarity = self._calculate_similarity(block1.content, block2.content)
                
                if similarity > 0.8:  # 80%相似度阈值
                    duplicate_lines += min(
                        block1.end_line - block1.start_line + 1,
                        block2.end_line - block2.start_line + 1
                    )
                    duplicates.append({
                        'block1': {
                            'start': block1.start_line,
                            'end': block1.end_line
                        },
                        'block2': {
                            'start': block2.start_line,
                            'end': block2.end_line
                        },
                        'similarity': similarity
                    })
                    
        return {
            'blocks': duplicates,
            'percentage': (duplicate_lines / total_lines * 100) if total_lines > 0 else 0
        }
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（使用简单的行集合相似度）"""
        lines1 = set(text1.split('\n'))
        lines2 = set(text2.split('\n'))
        
        intersection = len(lines1.intersection(lines2))
        union = len(lines1.union(lines2))
        
        return intersection / union if union > 0 else 0
        
    def _analyze_issues(self, blocks: List[CodeBlock], content: str, language: str) -> List[Dict[str, Any]]:
        """分析潜在问题"""
        issues = []
        
        # 检查过长的函数
        for block in blocks:
            if block.block_type in [CodeBlockType.FUNCTION, CodeBlockType.METHOD]:
                lines = block.content.split('\n')
                if len(lines) > 50:  # 函数过长阈值
                    issues.append({
                        'type': 'long_function',
                        'severity': 'warning',
                        'message': f'Function is too long ({len(lines)} lines)',
                        'location': {
                            'start_line': block.start_line,
                            'end_line': block.end_line
                        }
                    })
                    
        # 检查过深的嵌套
        for block in blocks:
            if block.block_type in [CodeBlockType.FUNCTION, CodeBlockType.METHOD]:
                max_depth = self._get_max_nesting_depth(block.content)
                if max_depth > 4:  # 嵌套深度阈值
                    issues.append({
                        'type': 'deep_nesting',
                        'severity': 'warning',
                        'message': f'Deep nesting detected (depth: {max_depth})',
                        'location': {
                            'start_line': block.start_line,
                            'end_line': block.end_line
                        }
                    })
                    
        # 检查命名规范
        for block in blocks:
            if not self._check_naming_convention(block, language):
                issues.append({
                    'type': 'naming_convention',
                    'severity': 'info',
                    'message': 'Name does not follow convention',
                    'location': {
                        'start_line': block.start_line,
                        'end_line': block.end_line
                    }
                })
                
        return issues
        
    def _get_max_nesting_depth(self, content: str) -> int:
        """获取最大嵌套深度"""
        max_depth = 0
        current_depth = 0
        
        for line in content.split('\n'):
            # 增加深度
            if re.search(r'{', line):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            # 减少深度
            if re.search(r'}', line):
                current_depth -= 1
                
        return max_depth
        
    def _check_naming_convention(self, block: CodeBlock, language: str) -> bool:
        """检查命名规范"""
        if language == "python":
            # Python命名规范
            if block.block_type == CodeBlockType.FUNCTION:
                return re.match(r'^[a-z_][a-z0-9_]*$', block.content) is not None
            elif block.block_type == CodeBlockType.CLASS:
                return re.match(r'^[A-Z][a-zA-Z0-9]*$', block.content) is not None
        elif language in ["javascript", "typescript"]:
            # JS/TS命名规范
            if block.block_type == CodeBlockType.FUNCTION:
                return re.match(r'^[a-z][a-zA-Z0-9]*$', block.content) is not None
            elif block.block_type == CodeBlockType.CLASS:
                return re.match(r'^[A-Z][a-zA-Z0-9]*$', block.content) is not None
                
        return True
        
    def analyze_project_dependencies(self, project_path: str) -> Dict[str, Any]:
        """
        分析项目依赖关系
        
        Args:
            project_path: 项目根目录路径
            
        Returns:
            依赖分析结果，包含:
            - file_dependencies: 文件间的依赖关系
            - module_dependencies: 模块间的依赖关系
            - external_dependencies: 外部依赖
            - dependency_graph: 依赖关系图
        """
        project_path = Path(project_path)
        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")
            
        # 收集所有代码文件
        code_files = []
        for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']:
            code_files.extend(project_path.rglob(f"*{ext}"))
            
        # 分析结果
        file_dependencies = {}  # 文件间依赖
        module_dependencies = {}  # 模块间依赖
        external_dependencies = set()  # 外部依赖
        
        # 并行分析文件
        with ThreadPoolExecutor() as executor:
            futures = []
            for file_path in code_files:
                if file_path.is_file():
                    futures.append(executor.submit(self._analyze_file_dependencies, file_path))
                    
            # 收集分析结果
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        file_path, deps = result
                        file_dependencies[str(file_path)] = deps
                        
                        # 提取模块依赖
                        module = str(file_path.parent.relative_to(project_path))
                        if module not in module_dependencies:
                            module_dependencies[module] = set()
                        for dep in deps:
                            if dep.startswith(str(project_path)):
                                dep_module = str(Path(dep).parent.relative_to(project_path))
                                if dep_module != module:
                                    module_dependencies[module].add(dep_module)
                            else:
                                external_dependencies.add(dep)
                except Exception as e:
                    logger.error(f"Failed to analyze dependencies for {file_path}: {e}")
                    
        # 构建依赖图
        dependency_graph = {
            'nodes': [],
            'edges': []
        }
        
        # 添加文件节点
        for file_path in file_dependencies:
            dependency_graph['nodes'].append({
                'id': file_path,
                'type': 'file',
                'label': Path(file_path).name
            })
            
        # 添加模块节点
        for module in module_dependencies:
            dependency_graph['nodes'].append({
                'id': f"module:{module}",
                'type': 'module',
                'label': module
            })
            
        # 添加依赖边
        for source, targets in file_dependencies.items():
            for target in targets:
                dependency_graph['edges'].append({
                    'source': source,
                    'target': target,
                    'type': 'file_dependency'
                })
                
        for source, targets in module_dependencies.items():
            for target in targets:
                dependency_graph['edges'].append({
                    'source': f"module:{source}",
                    'target': f"module:{target}",
                    'type': 'module_dependency'
                })
                
        return {
            'file_dependencies': file_dependencies,
            'module_dependencies': module_dependencies,
            'external_dependencies': list(external_dependencies),
            'dependency_graph': dependency_graph
        }
        
    def _analyze_file_dependencies(self, file_path: Path) -> Optional[Tuple[Path, Set[str]]]:
        """分析单个文件的依赖关系"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 获取文件语言
            language = self._get_language_by_extension(file_path.suffix)
            if not language:
                return None
                
            # 分析代码
            result = self.analyze_code(content, str(file_path), language)
            
            # 提取文件级依赖
            dependencies = set()
            
            # 从导入语句中提取依赖
            import_patterns = {
                'python': [
                    r'^\s*import\s+(\w+)',
                    r'^\s*from\s+(\w+(?:\.\w+)*)\s+import'
                ],
                'javascript': [
                    r'^\s*import.*from\s+[\'"]([^\'"]+)[\'"]',
                    r'^\s*require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
                ],
                'typescript': [
                    r'^\s*import.*from\s+[\'"]([^\'"]+)[\'"]'
                ]
            }
            
            if language in import_patterns:
                for pattern in import_patterns[language]:
                    for match in re.finditer(pattern, content, re.MULTILINE):
                        dependencies.add(match.group(1))
                        
            return file_path, dependencies
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return None
            
    def _get_language_by_extension(self, ext: str) -> Optional[str]:
        """根据文件扩展名获取语言"""
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust'
        }
        return ext_to_lang.get(ext.lower())
        
    def get_code_metrics(self, content: str, file_path: str, language: str) -> Dict[str, Any]:
        """
        获取代码度量数据
        
        Args:
            content: 代码内容
            file_path: 文件路径
            language: 编程语言
            
        Returns:
            代码度量数据，包含:
            - size_metrics: 大小度量（行数、字符数等）
            - complexity_metrics: 复杂度度量
            - maintainability_metrics: 可维护性度量
            - structural_metrics: 结构度量
        """
        try:
            # 获取或解析AST
            tree = self.ast_cache.get(file_path, content)
            if not tree and language in self.analyzer.parsers:
                parser = self.analyzer.parsers[language]
                tree = parser.parse(content.encode())
                self.ast_cache.put(file_path, content, tree)
                
            if not tree:
                return self._simple_metrics_analysis(content)
                
            # 提取代码块
            blocks = self._extract_code_blocks(tree.root_node, content)
            
            # 计算大小度量
            size_metrics = self._calculate_size_metrics(content)
            
            # 计算复杂度度量
            complexity_metrics = self._calculate_complexity_metrics(blocks)
            
            # 计算可维护性度量
            maintainability_metrics = self._calculate_maintainability_metrics(blocks, content)
            
            # 计算结构度量
            structural_metrics = self._calculate_structural_metrics(blocks)
            
            return {
                'size_metrics': size_metrics,
                'complexity_metrics': complexity_metrics,
                'maintainability_metrics': maintainability_metrics,
                'structural_metrics': structural_metrics
            }
            
        except Exception as e:
            logger.error(f"获取代码度量数据失败: {str(e)}")
            return self._simple_metrics_analysis(content)
            
    def _simple_metrics_analysis(self, content: str) -> Dict[str, Any]:
        """简单的度量分析"""
        lines = content.split('\n')
        return {
            'size_metrics': {
                'total_lines': len(lines),
                'source_lines': len([l for l in lines if l.strip()]),
                'comment_lines': len([l for l in lines if l.strip().startswith(('#', '//', '/*'))]),
                'blank_lines': len([l for l in lines if not l.strip()]),
                'character_count': len(content)
            },
            'complexity_metrics': {
                'cyclomatic': 1.0,
                'cognitive': 1.0
            },
            'maintainability_metrics': {
                'maintainability_index': 100.0,
                'halstead_volume': 0.0,
                'halstead_difficulty': 0.0
            },
            'structural_metrics': {
                'function_count': 0,
                'class_count': 0,
                'average_function_size': 0.0,
                'max_nesting_depth': 0
            }
        }