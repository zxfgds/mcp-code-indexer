"""
代码压缩器模块
提供代码压缩、规范化和简化功能
"""

import re
from typing import Dict, List, Set, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class NormalizationLevel(Enum):
    """规范化级别"""
    MINIMAL = "minimal"  # 最小化改动
    NORMAL = "normal"   # 标准规范化
    AGGRESSIVE = "aggressive"  # 激进规范化

@dataclass
class CompressionOptions:
    """压缩选项"""
    remove_comments: bool = True
    remove_empty_lines: bool = True
    normalize_whitespace: bool = True
    normalize_names: bool = False
    combine_imports: bool = True
    remove_unused: bool = True
    minify_strings: bool = False
    normalize_level: NormalizationLevel = NormalizationLevel.NORMAL

class CodeCompressor:
    """代码压缩器类"""
    
    def __init__(self):
        """初始化代码压缩器"""
        self.name_mapping: Dict[str, str] = {}
        self.used_names: Set[str] = set()
        self.preserved_names: Set[str] = {
            'self', 'cls', 'super', 'None', 'True', 'False',
            '__init__', '__main__', '__name__', '__file__'
        }
    
    def compress(self, content: str, language: str, 
                options: Optional[CompressionOptions] = None) -> str:
        """
        压缩代码
        
        Args:
            content: 代码内容
            language: 编程语言
            options: 压缩选项
            
        Returns:
            压缩后的代码
        """
        if options is None:
            options = CompressionOptions()
            
        # 保存重要注释
        preserved_comments = self._extract_important_comments(content)
        
        # 移除注释和空行
        if options.remove_comments:
            content = self._remove_comments(content, language)
        if options.remove_empty_lines:
            content = self._remove_empty_lines(content)
            
        # 规范化代码
        if options.normalize_whitespace:
            content = self._normalize_whitespace(content)
        if options.normalize_names:
            content = self._normalize_names(content, language)
            
        # 合并导入语句
        if options.combine_imports:
            content = self._combine_imports(content, language)
            
        # 移除未使用的代码
        if options.remove_unused:
            content = self._remove_unused_code(content, language)
            
        # 最小化字符串
        if options.minify_strings:
            content = self._minify_strings(content, language)
            
        # 恢复重要注释
        content = self._restore_important_comments(content, preserved_comments)
        
        return content
    
    def normalize(self, content: str, language: str, 
                 level: NormalizationLevel = NormalizationLevel.NORMAL) -> str:
        """
        规范化代码
        
        Args:
            content: 代码内容
            language: 编程语言
            level: 规范化级别
            
        Returns:
            规范化后的代码
        """
        # 基于级别选择规范化选项
        options = CompressionOptions(
            remove_comments=level == NormalizationLevel.AGGRESSIVE,
            remove_empty_lines=level != NormalizationLevel.MINIMAL,
            normalize_whitespace=True,
            normalize_names=level == NormalizationLevel.AGGRESSIVE,
            combine_imports=level != NormalizationLevel.MINIMAL,
            remove_unused=level == NormalizationLevel.AGGRESSIVE,
            minify_strings=False,
            normalize_level=level
        )
        
        return self.compress(content, language, options)
    
    def _extract_important_comments(self, content: str) -> List[Tuple[int, str]]:
        """提取重要注释"""
        important_patterns = [
            r'#\s*TODO',
            r'#\s*FIXME',
            r'#\s*NOTE',
            r'#\s*IMPORTANT',
            r'""".*?TODO.*?"""',
            r"'''.*?TODO.*?'''",
        ]
        
        important_comments = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            for pattern in important_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    important_comments.append((i + 1, line))
                    break
                    
        return important_comments
    
    def _remove_comments(self, content: str, language: str) -> str:
        """移除注释"""
        if language in ['python']:
            # 移除单行注释
            content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
            # 移除多行注释
            content = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
            content = re.sub(r"'''.*?'''", '', content, flags=re.DOTALL)
            
        elif language in ['javascript', 'typescript']:
            # 移除单行注释
            content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
            # 移除多行注释
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            
        return content
    
    def _remove_empty_lines(self, content: str) -> str:
        """移除空行"""
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        return '\n'.join(non_empty_lines)
    
    def _normalize_whitespace(self, content: str) -> str:
        """规范化空白字符"""
        # 规范化缩进
        lines = content.split('\n')
        normalized_lines = []
        
        for line in lines:
            # 将制表符转换为空格
            line = line.replace('\t', '    ')
            # 移除行尾空白
            line = line.rstrip()
            # 确保操作符周围有空格
            line = re.sub(r'([=+\-*/<>!]+)', r' \1 ', line)
            # 移除多余空格
            line = re.sub(r'\s+', ' ', line)
            # 保持缩进
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                line = ' ' * indent + line.lstrip()
            normalized_lines.append(line)
            
        return '\n'.join(normalized_lines)
    
    def _normalize_names(self, content: str, language: str) -> str:
        """规范化变量名"""
        # 识别变量名
        if language in ['python']:
            pattern = r'\b[a-zA-Z_]\w*\b'
        else:
            pattern = r'\b[a-zA-Z_$]\w*\b'
            
        def replace_name(match):
            name = match.group(0)
            if name in self.preserved_names:
                return name
                
            if name not in self.name_mapping:
                new_name = self._generate_name(len(self.name_mapping))
                self.name_mapping[name] = new_name
                
            return self.name_mapping[name]
            
        return re.sub(pattern, replace_name, content)
    
    def _generate_name(self, index: int) -> str:
        """生成简短的变量名"""
        chars = 'abcdefghijklmnopqrstuvwxyz'
        base = len(chars)
        
        if index < base:
            return chars[index]
            
        name = ''
        while index >= 0:
            name = chars[index % base] + name
            index = index // base - 1
            
        return name
    
    def _combine_imports(self, content: str, language: str) -> str:
        """合并导入语句"""
        if language == 'python':
            # 提取所有导入语句
            import_pattern = r'^(?:from\s+[\w.]+\s+)?import\s+(?:[\w.]+(?:\s+as\s+\w+)?(?:\s*,\s*[\w.]+(?:\s+as\s+\w+)?)*)'
            imports = re.finditer(import_pattern, content, re.MULTILINE)
            
            # 按模块分组
            grouped_imports = {}
            for match in imports:
                import_stmt = match.group(0)
                if import_stmt.startswith('from'):
                    module = re.match(r'from\s+([\w.]+)', import_stmt).group(1)
                    if module not in grouped_imports:
                        grouped_imports[module] = []
                    grouped_imports[module].append(import_stmt)
                else:
                    if 'direct' not in grouped_imports:
                        grouped_imports['direct'] = []
                    grouped_imports['direct'].append(import_stmt)
                    
            # 合并导入语句
            new_imports = []
            for module, stmts in grouped_imports.items():
                if module == 'direct':
                    new_imports.extend(stmts)
                else:
                    imports = []
                    for stmt in stmts:
                        imports.extend(re.findall(r'import\s+((?:[\w.]+(?:\s+as\s+\w+)?(?:\s*,\s*)?)+)', stmt))
                    new_imports.append(f"from {module} import {', '.join(imports)}")
                    
            # 替换原有导入语句
            content = re.sub(import_pattern + r'\n?', '', content, flags=re.MULTILINE)
            return '\n'.join(new_imports) + '\n\n' + content.lstrip()
            
        return content
    
    def _remove_unused_code(self, content: str, language: str) -> str:
        """移除未使用的代码"""
        # TODO: 实现未使用代码检测和移除
        return content
    
    def _minify_strings(self, content: str, language: str) -> str:
        """最小化字符串"""
        def shorten_string(match):
            string = match.group(0)
            # 保持字符串引号
            quote = string[0]
            # 压缩空白字符
            content = string[1:-1]
            content = re.sub(r'\s+', ' ', content)
            return quote + content + quote
            
        # 处理单引号和双引号字符串
        content = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', shorten_string, content)
        content = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", shorten_string, content)
        
        return content
    
    def _restore_important_comments(self, content: str, 
                                  comments: List[Tuple[int, str]]) -> str:
        """恢复重要注释"""
        if not comments:
            return content
            
        lines = content.split('\n')
        
        # 按行号倒序插入注释
        for line_num, comment in sorted(comments, reverse=True):
            if line_num <= len(lines):
                lines.insert(line_num - 1, comment)
            else:
                lines.append(comment)
                
        return '\n'.join(lines)