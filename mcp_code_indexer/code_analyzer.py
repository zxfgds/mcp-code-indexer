"""
代码分析器模块
提供代码语法分析和依赖关系分析功能
"""

import os
from typing import List, Dict, Any, Optional, Set
from tree_sitter import Language, Parser
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CodeAnalyzer:
    """代码分析器类"""
    
    def __init__(self):
        """初始化代码分析器"""
        # 初始化tree-sitter
        self.parsers = {}
        self._init_parsers()
        
    def _init_parsers(self):
        """初始化各语言的解析器"""
        try:
            # 获取语言文件路径
            languages_dir = Path(__file__).parent / "parsers"
            languages_dir.mkdir(exist_ok=True)
            
            # 构建语言支持
            Language.build_library(
                str(languages_dir / "languages.so"),
                [
                    str(languages_dir / "tree-sitter-python"),
                    str(languages_dir / "tree-sitter-javascript"),
                    str(languages_dir / "tree-sitter-typescript"),
                    str(languages_dir / "tree-sitter-php"),
                    str(languages_dir / "tree-sitter-rust"),
                    str(languages_dir / "tree-sitter-vue")
                ]
            )
            
            # 加载语言
            PYTHON = Language(str(languages_dir / "languages.so"), "python")
            JAVASCRIPT = Language(str(languages_dir / "languages.so"), "javascript")
            TYPESCRIPT = Language(str(languages_dir / "languages.so"), "typescript")
            PHP = Language(str(languages_dir / "languages.so"), "php")
            RUST = Language(str(languages_dir / "languages.so"), "rust")
            VUE = Language(str(languages_dir / "languages.so"), "vue")
            
            # 创建解析器
            self.parsers = {
                "python": Parser(),
                "javascript": Parser(),
                "typescript": Parser(),
                "php": Parser(),
                "rust": Parser(),
                "vue": Parser()
            }
            
            # 设置语言
            self.parsers["python"].set_language(PYTHON)
            self.parsers["javascript"].set_language(JAVASCRIPT)
            self.parsers["typescript"].set_language(TYPESCRIPT)
            self.parsers["php"].set_language(PHP)
            self.parsers["rust"].set_language(RUST)
            self.parsers["vue"].set_language(VUE)
            
        except Exception as e:
            logger.error(f"初始化解析器失败: {str(e)}")
    
    def analyze_code(self, content: str, language: str) -> Dict[str, Any]:
        """
        分析代码内容
        
        Args:
            content: 代码内容
            language: 编程语言
            
        Returns:
            分析结果字典
        """
        if language not in self.parsers:
            return self._simple_analysis(content)
            
        try:
            parser = self.parsers[language]
            tree = parser.parse(content.encode())
            
            # 分析结果
            result = {
                "imports": self._analyze_imports(tree.root_node, language),
                "functions": self._analyze_functions(tree.root_node, language),
                "classes": self._analyze_classes(tree.root_node, language),
                "dependencies": self._analyze_dependencies(tree.root_node, language)
            }
            
            return result
        except Exception as e:
            logger.error(f"代码分析失败: {str(e)}")
            return self._simple_analysis(content)
    
    def _simple_analysis(self, content: str) -> Dict[str, Any]:
        """
        简单的文本分析（用于不支持的语言）
        
        Args:
            content: 代码内容
            
        Returns:
            分析结果字典
        """
        lines = content.split('\n')
        return {
            "imports": [],
            "functions": [],
            "classes": [],
            "dependencies": [],
            "line_count": len(lines)
        }
    
    def _analyze_imports(self, node: Any, language: str) -> List[str]:
        """
        分析导入语句
        
        Args:
            node: 语法树节点
            language: 编程语言
            
        Returns:
            导入模块列表
        """
        imports = []
        
        if language == "python":
            # 分析Python的import语句
            import_nodes = node.children_by_field_name("import")
            for import_node in import_nodes:
                imports.append(import_node.text.decode())
                
        elif language in ["javascript", "typescript"]:
            # 分析JS/TS的import语句
            import_nodes = node.children_by_field_name("import")
            for import_node in import_nodes:
                imports.append(import_node.text.decode())
                
        return imports
    
    def _analyze_functions(self, node: Any, language: str) -> List[Dict[str, Any]]:
        """
        分析函数定义
        
        Args:
            node: 语法树节点
            language: 编程语言
            
        Returns:
            函数信息列表
        """
        functions = []
        
        # 遍历AST查找函数定义
        cursor = node.walk()
        
        def visit(node):
            if node.type == "function_definition":  # Python
                functions.append({
                    "name": node.child_by_field_name("name").text.decode(),
                    "start_line": node.start_point[0],
                    "end_line": node.end_point[0]
                })
            elif node.type == "function_declaration":  # JS/TS
                functions.append({
                    "name": node.child_by_field_name("name").text.decode(),
                    "start_line": node.start_point[0],
                    "end_line": node.end_point[0]
                })
                
            for child in node.children:
                visit(child)
                
        visit(node)
        return functions
    
    def _analyze_classes(self, node: Any, language: str) -> List[Dict[str, Any]]:
        """
        分析类定义
        
        Args:
            node: 语法树节点
            language: 编程语言
            
        Returns:
            类信息列表
        """
        classes = []
        
        # 遍历AST查找类定义
        cursor = node.walk()
        
        def visit(node):
            if node.type == "class_definition":  # Python
                classes.append({
                    "name": node.child_by_field_name("name").text.decode(),
                    "start_line": node.start_point[0],
                    "end_line": node.end_point[0]
                })
            elif node.type == "class_declaration":  # JS/TS
                classes.append({
                    "name": node.child_by_field_name("name").text.decode(),
                    "start_line": node.start_point[0],
                    "end_line": node.end_point[0]
                })
                
            for child in node.children:
                visit(child)
                
        visit(node)
        return classes
    
    def _analyze_dependencies(self, node: Any, language: str) -> List[str]:
        """
        分析代码依赖
        
        Args:
            node: 语法树节点
            language: 编程语言
            
        Returns:
            依赖列表
        """
        dependencies = set()
        
        # 遍历AST查找依赖关系
        cursor = node.walk()
        
        def visit(node):
            # 记录函数调用
            if node.type == "call":
                func_name = node.child_by_field_name("function")
                if func_name:
                    dependencies.add(func_name.text.decode())
            
            # 记录类实例化
            elif node.type == "class_instantiation":
                class_name = node.child_by_field_name("class")
                if class_name:
                    dependencies.add(class_name.text.decode())
                    
            for child in node.children:
                visit(child)
                
        visit(node)
        return list(dependencies)