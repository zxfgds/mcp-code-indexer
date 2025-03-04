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
            
            # 加载语言
            language_names = ["python", "javascript", "typescript", "php", "rust"]
            
            # 创建解析器
            self.parsers = {}
            
            # 直接从parsers目录加载DLL文件
            for lang_name in language_names:
                try:
                    # 创建解析器
                    parser = Parser()
                    
                    # 直接检查parsers目录下的DLL文件
                    dll_path = languages_dir / f"{lang_name}.dll"
                    
                    if dll_path.exists():
                        # 尝试使用ctypes加载DLL
                        import ctypes
                        lib = ctypes.CDLL(str(dll_path))
                        # 获取语言函数指针
                        tree_sitter_lang_fn = getattr(lib, f"tree_sitter_{lang_name}")
                        # 创建语言对象
                        lang = Language(tree_sitter_lang_fn())
                        # 设置解析器语言
                        parser.set_language(lang)
                        logger.info(f"成功加载 {lang_name} 语言库")
                    else:
                        logger.warning(f"未找到 {lang_name} 语言库，使用默认解析器")
                    
                    # 保存解析器
                    self.parsers[lang_name] = parser
                    logger.info(f"初始化 {lang_name} 解析器")
                    logger.info(f"成功初始化 {lang_name} 解析器")
                except Exception as e:
                    logger.error(f"初始化 {lang_name} 解析器失败: {str(e)}")
            
            # 特别关注PHP解析器
            if "php" not in self.parsers:
                logger.error("PHP解析器初始化失败，这将导致PHP代码分析功能不可用")
                
        except Exception as e:
            logger.error(f"初始化解析器失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # 已删除不再需要的方法
    
    def _has_language(self, parser: Parser) -> bool:
        """检查解析器是否设置了语言"""
        try:
            # 尝试解析一个简单的字符串，如果成功，说明设置了语言
            test_tree = parser.parse(b"test")
            return test_tree is not None
        except ValueError:
            return False
            
    # 已删除正则表达式分析方法
    
    def analyze_code(self, content: str, language: str) -> Dict[str, Any]:
        """
        分析代码内容
        
        Args:
            content: 代码内容
            language: 编程语言
            
        Returns:
            分析结果字典
        """
        # 检查是否有对应语言的解析器
        if language not in self.parsers:
            logger.warning(f"未找到 {language} 语言的解析器，返回空结果")
            # 返回空结果
            return {
                "imports": [],
                "functions": [],
                "classes": [],
                "dependencies": [],
                "line_count": len(content.split('\n'))
            }
            
        try:
            # 使用DLL解析器分析代码
            parser = self.parsers[language]
            tree = parser.parse(content.encode())
            
            # 分析结果
            result = {
                "imports": self._analyze_imports(tree.root_node, language),
                "functions": self._analyze_functions(tree.root_node, language),
                "classes": self._analyze_classes(tree.root_node, language),
                "dependencies": self._analyze_dependencies(tree.root_node, language),
                "line_count": len(content.split('\n'))
            }
            
            return result
        except Exception as e:
            logger.error(f"使用DLL解析器分析代码失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 如果解析失败，返回空结果
            return {
                "imports": [],
                "functions": [],
                "classes": [],
                "dependencies": [],
                "line_count": len(content.split('\n'))
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
        def visit(node):
            # 根据不同语言处理不同的函数定义节点类型
            if language == "python" and node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    functions.append({
                        "name": name_node.text.decode('utf-8'),
                        "start_line": node.start_point[0],
                        "end_line": node.end_point[0]
                    })
            
            elif language in ["javascript", "typescript"]:
                # JS/TS函数声明
                if node.type == "function_declaration":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        functions.append({
                            "name": name_node.text.decode('utf-8'),
                            "start_line": node.start_point[0],
                            "end_line": node.end_point[0]
                        })
                # 箭头函数和方法定义
                elif node.type in ["arrow_function", "method_definition"]:
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        functions.append({
                            "name": name_node.text.decode('utf-8'),
                            "start_line": node.start_point[0],
                            "end_line": node.end_point[0]
                        })
                # 变量声明中的函数表达式
                elif node.type == "variable_declarator":
                    name_node = node.child_by_field_name("name")
                    value_node = node.child_by_field_name("value")
                    if name_node and value_node and value_node.type in ["function", "arrow_function"]:
                        functions.append({
                            "name": name_node.text.decode('utf-8'),
                            "start_line": node.start_point[0],
                            "end_line": node.end_point[0]
                        })
            
            elif language == "php":
                # PHP函数定义
                if node.type == "function_definition":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        # 提取参数信息
                        params = []
                        parameters = node.child_by_field_name("parameters")
                        if parameters:
                            for param in parameters.children:
                                if param.type == "formal_parameter":
                                    param_name = param.child_by_field_name("name")
                                    if param_name:
                                        params.append(param_name.text.decode('utf-8'))
                        
                        functions.append({
                            "name": name_node.text.decode('utf-8'),
                            "parameters": params,
                            "start_line": node.start_point[0],
                            "end_line": node.end_point[0]
                        })
                
                # PHP方法定义
                elif node.type == "method_declaration":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        # 提取参数信息
                        params = []
                        parameters = node.child_by_field_name("parameters")
                        if parameters:
                            for param in parameters.children:
                                if param.type == "formal_parameter":
                                    param_name = param.child_by_field_name("name")
                                    if param_name:
                                        params.append(param_name.text.decode('utf-8'))
                        
                        # 提取可见性
                        visibility = "public"  # 默认可见性
                        modifiers = node.children_by_field_name("modifiers")
                        for modifier in modifiers:
                            mod_text = modifier.text.decode('utf-8')
                            if mod_text in ["public", "protected", "private"]:
                                visibility = mod_text
                                break
                        
                        functions.append({
                            "name": name_node.text.decode('utf-8'),
                            "type": "method",
                            "visibility": visibility,
                            "parameters": params,
                            "start_line": node.start_point[0],
                            "end_line": node.end_point[0]
                        })
                
            # 递归处理子节点
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
        def visit(node):
            # 根据不同语言处理不同的类定义节点类型
            if language == "python" and node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    classes.append({
                        "name": name_node.text.decode('utf-8'),
                        "start_line": node.start_point[0],
                        "end_line": node.end_point[0]
                    })
            
            elif language in ["javascript", "typescript"]:
                # JS/TS类声明
                if node.type == "class_declaration":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        classes.append({
                            "name": name_node.text.decode('utf-8'),
                            "start_line": node.start_point[0],
                            "end_line": node.end_point[0]
                        })
                # 类表达式
                elif node.type == "class":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        classes.append({
                            "name": name_node.text.decode('utf-8'),
                            "start_line": node.start_point[0],
                            "end_line": node.end_point[0]
                        })
                # 变量声明中的类表达式
                elif node.type == "variable_declarator":
                    name_node = node.child_by_field_name("name")
                    value_node = node.child_by_field_name("value")
                    if name_node and value_node and value_node.type == "class":
                        classes.append({
                            "name": name_node.text.decode('utf-8'),
                            "start_line": node.start_point[0],
                            "end_line": node.end_point[0]
                        })
            
            elif language == "php":
                # PHP类定义
                if node.type == "class_declaration":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        classes.append({
                            "name": name_node.text.decode('utf-8'),
                            "start_line": node.start_point[0],
                            "end_line": node.end_point[0]
                        })
                # PHP接口定义
                elif node.type == "interface_declaration":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        classes.append({
                            "name": name_node.text.decode('utf-8') + " (interface)",
                            "start_line": node.start_point[0],
                            "end_line": node.end_point[0]
                        })
                
            # 递归处理子节点
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
        def visit(node):
            if language == "python":
                # Python函数调用
                if node.type == "call":
                    func_name = node.child_by_field_name("function")
                    if func_name:
                        dependencies.add(func_name.text.decode('utf-8'))
                
                # Python导入
                elif node.type in ["import_statement", "import_from_statement"]:
                    dependencies.add(node.text.decode('utf-8'))
            
            elif language in ["javascript", "typescript"]:
                # JS/TS函数调用
                if node.type == "call_expression":
                    func_name = node.child_by_field_name("function")
                    if func_name:
                        dependencies.add(func_name.text.decode('utf-8'))
                
                # JS/TS类实例化 (new 操作符)
                elif node.type == "new_expression":
                    constructor = node.child_by_field_name("constructor")
                    if constructor:
                        dependencies.add(constructor.text.decode('utf-8'))
                
                # JS/TS导入
                elif node.type in ["import_statement", "import_declaration"]:
                    dependencies.add(node.text.decode('utf-8'))
                
                # JS/TS require调用
                elif node.type == "call_expression" and node.child_by_field_name("function") and node.child_by_field_name("function").text.decode('utf-8') == "require":
                    args = node.child_by_field_name("arguments")
                    if args and args.children and len(args.children) > 0:
                        dependencies.add("require(" + args.children[0].text.decode('utf-8') + ")")
            
            elif language == "php":
                # PHP函数调用
                if node.type == "function_call_expression":
                    func_name = node.child_by_field_name("name")
                    if func_name:
                        dependencies.add(func_name.text.decode('utf-8'))
                
                # PHP类实例化
                elif node.type == "object_creation_expression":
                    class_name = node.child_by_field_name("class_name")
                    if class_name:
                        dependencies.add(class_name.text.decode('utf-8'))
                
                # PHP导入
                elif node.type in ["include_expression", "require_expression", "include_once_expression", "require_once_expression"]:
                    dependencies.add(node.text.decode('utf-8'))
            
            # 递归处理子节点
            for child in node.children:
                visit(child)
                
        visit(node)
        return list(dependencies)
        
        
    def get_code_structure(self, content: str, language: str) -> Dict[str, Any]:
        """
        获取代码结构信息
        
        Args:
            content: 代码内容
            language: 编程语言
            
        Returns:
            代码结构信息，包含:
            - functions: 函数列表
            - classes: 类列表
            - imports: 导入语句列表
            - dependencies: 依赖关系
            - variables: 全局变量
            - structure: 整体代码结构
        """
        # 检查是否有对应语言的解析器
        if language not in self.parsers:
            logger.warning(f"未找到 {language} 语言的解析器，返回空结果")
            # 返回空结果
            return {
                "functions": [],
                "classes": [],
                "imports": [],
                "dependencies": [],
                "variables": [],
                "structure": {
                    "type": "file",
                    "children": []
                }
            }
            
        try:
            # 使用DLL解析器分析代码结构
            parser = self.parsers[language]
            tree = parser.parse(content.encode())
            
            # 分析基本结构
            functions = self._analyze_functions(tree.root_node, language)
            classes = self._analyze_classes(tree.root_node, language)
            imports = self._analyze_imports(tree.root_node, language)
            dependencies = self._analyze_dependencies(tree.root_node, language)
            
            # 提取全局变量
            variables = self._extract_global_variables(tree.root_node, language)
            
            # 构建代码结构树
            structure = self._build_structure_tree(tree.root_node, language)
            
            return {
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "dependencies": dependencies,
                "variables": variables,
                "structure": structure
            }
            
        except Exception as e:
            logger.error(f"使用DLL解析器分析代码结构失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 如果解析失败，返回空结果
            return {
                "functions": [],
                "classes": [],
                "imports": [],
                "dependencies": [],
                "variables": [],
                "structure": {
                    "type": "file",
                    "children": []
                }
            }
        
    def _extract_global_variables(self, node: Any, language: str) -> List[Dict[str, Any]]:
        """提取全局变量"""
        variables = []
        
        def visit(node):
            # Python 全局变量赋值
            if language == "python" and node.type == "assignment" and node.parent.type == "module":
                left_node = node.child_by_field_name("left")
                right_node = node.child_by_field_name("right")
                if left_node and right_node:
                    variables.append({
                        "name": left_node.text.decode('utf-8'),
                        "value": right_node.text.decode('utf-8'),
                        "line": node.start_point[0]
                    })
            
            # JavaScript/TypeScript 全局变量
            elif language in ["javascript", "typescript"] and node.type in ["variable_declaration", "const_declaration"]:
                if node.parent.type == "program":
                    for child in node.children:
                        if child.type == "variable_declarator":
                            name_node = child.child_by_field_name("name")
                            value_node = child.child_by_field_name("value")
                            if name_node:
                                variables.append({
                                    "name": name_node.text.decode('utf-8'),
                                    "value": value_node.text.decode('utf-8') if value_node else None,
                                    "line": node.start_point[0]
                                })
            
            # PHP 全局变量
            elif language == "php":
                # PHP 全局变量声明 ($var = value)
                if node.type == "expression_statement":
                    expr = node.child_by_field_name("expression")
                    if expr and expr.type == "assignment_expression":
                        left = expr.child_by_field_name("left")
                        right = expr.child_by_field_name("right")
                        # 确保是全局作用域
                        if left and node.parent and node.parent.type in ["program", "namespace_definition_body"]:
                            # 确保是变量而不是属性访问
                            if left.type == "variable_name" and left.text.decode('utf-8').startswith('$'):
                                variables.append({
                                    "name": left.text.decode('utf-8'),
                                    "value": right.text.decode('utf-8') if right else None,
                                    "line": node.start_point[0]
                                })
                
                # PHP define 常量定义
                elif node.type == "function_call_expression":
                    func_name = node.child_by_field_name("name")
                    if func_name and func_name.text.decode('utf-8') == "define":
                        args = node.child_by_field_name("arguments")
                        if args and len(args.children) >= 2:
                            # define的第一个参数是常量名，第二个参数是值
                            name_arg = args.children[0]
                            value_arg = args.children[1]
                            if name_arg and name_arg.type == "string":
                                variables.append({
                                    "name": name_arg.text.decode('utf-8').strip('"\''),
                                    "value": value_arg.text.decode('utf-8') if value_arg else None,
                                    "type": "constant",
                                    "line": node.start_point[0]
                                })
                
                # PHP const 常量定义
                elif node.type == "const_declaration":
                    for child in node.children:
                        if child.type == "const_element":
                            name_node = child.child_by_field_name("name")
                            value_node = child.child_by_field_name("value")
                            if name_node:
                                variables.append({
                                    "name": name_node.text.decode('utf-8'),
                                    "value": value_node.text.decode('utf-8') if value_node else None,
                                    "type": "constant",
                                    "line": node.start_point[0]
                                })
            
            for child in node.children:
                visit(child)
        
        visit(node)
        return variables
        
    def _build_structure_tree(self, node: Any, language: str) -> Dict[str, Any]:
        """
        构建代码结构树，支持多种编程语言
        
        Args:
            node: 语法树节点
            language: 编程语言
            
        Returns:
            代码结构树
        """
        def create_node(type_name: str, name: str = None, children: List[Dict] = None,
                       attributes: Dict[str, Any] = None, location: Dict[str, int] = None) -> Dict[str, Any]:
            """创建结构树节点"""
            node_data = {"type": type_name}
            if name:
                node_data["name"] = name
            if children:
                node_data["children"] = children
            if attributes:
                node_data["attributes"] = attributes
            if location:
                node_data["location"] = location
            return node_data
        
        def get_location(node) -> Dict[str, int]:
            """获取节点位置信息"""
            return {
                "start_line": node.start_point[0] + 1,
                "start_column": node.start_point[1] + 1,
                "end_line": node.end_point[0] + 1,
                "end_column": node.end_point[1] + 1
            }
        
        def get_node_text(node) -> str:
            """获取节点文本"""
            try:
                return node.text.decode('utf-8')
            except:
                return ""
        
        def visit(node) -> List[Dict[str, Any]]:
            """递归访问节点构建结构树"""
            children = []
            
            # 根据不同语言处理不同的节点类型
            if language == "python":
                # Python导入语句
                if node.type in ["import_statement", "import_from_statement"]:
                    children.append(create_node(
                        "import",
                        get_node_text(node),
                        location=get_location(node)
                    ))
                
                # Python类定义
                elif node.type == "class_definition":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        # 提取基类信息
                        bases = []
                        arguments = node.child_by_field_name("superclasses")
                        if arguments:
                            for arg in arguments.children:
                                if arg.type not in [",", "("]:
                                    bases.append(get_node_text(arg))
                        
                        # 创建类节点
                        class_node = create_node(
                            "class",
                            get_node_text(name_node),
                            attributes={"bases": bases} if bases else None,
                            location=get_location(node)
                        )
                        
                        # 处理类体
                        body_node = node.child_by_field_name("body")
                        if body_node:
                            class_children = []
                            for child in body_node.children:
                                class_children.extend(visit(child))
                            if class_children:
                                class_node["children"] = class_children
                                
                        children.append(class_node)
                
                # Python函数定义
                elif node.type == "function_definition":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        # 提取参数信息
                        params = []
                        parameters = node.child_by_field_name("parameters")
                        if parameters:
                            for param in parameters.children:
                                if param.type not in [",", "(", ")"]:
                                    params.append(get_node_text(param))
                        
                        # 创建函数节点
                        func_node = create_node(
                            "function",
                            get_node_text(name_node),
                            attributes={"parameters": params} if params else None,
                            location=get_location(node)
                        )
                        
                        # 处理函数体
                        body_node = node.child_by_field_name("body")
                        if body_node:
                            func_children = []
                            for child in body_node.children:
                                func_children.extend(visit(child))
                            if func_children:
                                func_node["children"] = func_children
                                
                        children.append(func_node)
                
                # Python变量赋值
                elif node.type == "assignment" and node.parent and node.parent.type == "module":
                    left = node.child_by_field_name("left")
                    right = node.child_by_field_name("right")
                    if left and right:
                        children.append(create_node(
                            "variable",
                            get_node_text(left),
                            attributes={"value": get_node_text(right)},
                            location=get_location(node)
                        ))
            
            elif language in ["javascript", "typescript"]:
                # JS/TS导入语句
                if node.type in ["import_statement", "import_declaration"]:
                    children.append(create_node(
                        "import",
                        get_node_text(node),
                        location=get_location(node)
                    ))
                
                # JS/TS导出语句
                elif node.type in ["export_statement", "export_declaration"]:
                    children.append(create_node(
                        "export",
                        get_node_text(node),
                        location=get_location(node)
                    ))
                
                # JS/TS类定义
                elif node.type == "class_declaration":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        # 提取继承信息
                        extends_clause = node.child_by_field_name("extends")
                        extends_name = None
                        if extends_clause:
                            extends_name = get_node_text(extends_clause)
                        
                        # 创建类节点
                        class_node = create_node(
                            "class",
                            get_node_text(name_node),
                            attributes={"extends": extends_name} if extends_name else None,
                            location=get_location(node)
                        )
                        
                        # 处理类体
                        body_node = node.child_by_field_name("body")
                        if body_node:
                            class_children = []
                            for child in body_node.children:
                                class_children.extend(visit(child))
                            if class_children:
                                class_node["children"] = class_children
                                
                        children.append(class_node)
                
                # JS/TS函数定义
                elif node.type in ["function_declaration", "method_definition"]:
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        # 提取参数信息
                        params = []
                        parameters = node.child_by_field_name("parameters")
                        if parameters:
                            for param in parameters.children:
                                if param.type not in [",", "(", ")"]:
                                    params.append(get_node_text(param))
                        
                        # 创建函数节点
                        func_node = create_node(
                            "function" if node.type == "function_declaration" else "method",
                            get_node_text(name_node),
                            attributes={"parameters": params} if params else None,
                            location=get_location(node)
                        )
                        
                        # 处理函数体
                        body_node = node.child_by_field_name("body")
                        if body_node:
                            func_children = []
                            for child in body_node.children:
                                func_children.extend(visit(child))
                            if func_children:
                                func_node["children"] = func_children
                                
                        children.append(func_node)
                
                # JS/TS变量声明
                elif node.type in ["variable_declaration", "const_declaration", "let_declaration"]:
                    for child in node.children:
                        if child.type == "variable_declarator":
                            name_node = child.child_by_field_name("name")
                            value_node = child.child_by_field_name("value")
                            if name_node:
                                children.append(create_node(
                                    "variable",
                                    get_node_text(name_node),
                                    attributes={"value": get_node_text(value_node) if value_node else None},
                                    location=get_location(child)
                                ))
                
                # 箭头函数和函数表达式
                elif node.type in ["arrow_function", "function"]:
                    # 如果是变量声明的一部分，已经在上面处理过了
                    if node.parent and node.parent.type == "variable_declarator":
                        pass
                    else:
                        # 匿名函数
                        func_node = create_node(
                            "anonymous_function",
                            "anonymous",
                            location=get_location(node)
                        )
                        
                        # 处理函数体
                        body_node = node.child_by_field_name("body")
                        if body_node:
                            func_children = []
                            for child in body_node.children:
                                func_children.extend(visit(child))
                            if func_children:
                                func_node["children"] = func_children
                                
                        children.append(func_node)
            
            elif language == "php":
                # PHP命名空间
                if node.type == "namespace_definition":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        namespace_node = create_node(
                            "namespace",
                            get_node_text(name_node),
                            location=get_location(node)
                        )
                        
                        # 处理命名空间体
                        body_node = node.child_by_field_name("body")
                        if body_node:
                            namespace_children = []
                            for child in body_node.children:
                                namespace_children.extend(visit(child))
                            if namespace_children:
                                namespace_node["children"] = namespace_children
                                
                        children.append(namespace_node)
                
                # PHP类定义
                elif node.type == "class_declaration":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        # 提取继承和实现信息
                        extends_clause = node.child_by_field_name("extends")
                        implements_clause = node.child_by_field_name("implements")
                        
                        attributes = {}
                        if extends_clause:
                            attributes["extends"] = get_node_text(extends_clause)
                        if implements_clause:
                            attributes["implements"] = get_node_text(implements_clause)
                        
                        # 创建类节点
                        class_node = create_node(
                            "class",
                            get_node_text(name_node),
                            attributes=attributes if attributes else None,
                            location=get_location(node)
                        )
                        
                        # 处理类体
                        body_node = node.child_by_field_name("body")
                        if body_node:
                            class_children = []
                            for child in body_node.children:
                                class_children.extend(visit(child))
                            if class_children:
                                class_node["children"] = class_children
                                
                        children.append(class_node)
                
                # PHP接口定义
                elif node.type == "interface_declaration":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        interface_node = create_node(
                            "interface",
                            get_node_text(name_node),
                            location=get_location(node)
                        )
                        
                        # 处理接口体
                        body_node = node.child_by_field_name("body")
                        if body_node:
                            interface_children = []
                            for child in body_node.children:
                                interface_children.extend(visit(child))
                            if interface_children:
                                interface_node["children"] = interface_children
                                
                        children.append(interface_node)
                
                # PHP函数定义
                elif node.type == "function_definition":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        # 提取参数信息
                        params = []
                        parameters = node.child_by_field_name("parameters")
                        if parameters:
                            for param in parameters.children:
                                if param.type not in [",", "(", ")"]:
                                    params.append(get_node_text(param))
                        
                        # 创建函数节点
                        func_node = create_node(
                            "function",
                            get_node_text(name_node),
                            attributes={"parameters": params} if params else None,
                            location=get_location(node)
                        )
                        
                        # 处理函数体
                        body_node = node.child_by_field_name("body")
                        if body_node:
                            func_children = []
                            for child in body_node.children:
                                func_children.extend(visit(child))
                            if func_children:
                                func_node["children"] = func_children
                                
                        children.append(func_node)
            
            # 递归处理子节点
            for child in node.children:
                children.extend(visit(child))
            
            return children
        
        # 构建文件级结构树
        file_name = ""
        if hasattr(node, 'filename'):
            file_name = os.path.basename(node.filename)
            
        root = create_node(
            "file",
            name=file_name,
            children=visit(node),
            attributes={"language": language}
        )
        return root