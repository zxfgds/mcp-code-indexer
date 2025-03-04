"""
代码优化器模块
提供代码分析、优化和压缩功能
"""

import os
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
    DOCSTRING = "docstring"
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
            elif node.type in ["comment", "docstring"]:
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
        block_type = CodeBlockType.DOCSTRING if node.type == "docstring" else CodeBlockType.COMMENT
        return CodeBlock(
            content=self._get_node_text(node, content),
            block_type=block_type,
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
        if block.block_type in [CodeBlockType.COMMENT, CodeBlockType.DOCSTRING]:
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
        if block.block_type in [CodeBlockType.COMMENT, CodeBlockType.DOCSTRING]:
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