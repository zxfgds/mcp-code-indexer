"""
上下文管理器模块
提供代码上下文的缓存、压缩和优先级管理功能
"""

import os
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
import time
import heapq
from concurrent.futures import ThreadPoolExecutor

from .code_optimizer import CodeOptimizer
from .code_compressor import CodeCompressor, NormalizationLevel

logger = logging.getLogger(__name__)

class ContextType(Enum):
    """上下文类型"""
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    FILE = "file"
    DIRECTORY = "directory"

class ContextPriority(Enum):
    """上下文优先级"""
    CRITICAL = 0    # 关键上下文
    HIGH = 1       # 高优先级
    NORMAL = 2     # 普通优先级
    LOW = 3        # 低优先级
    BACKGROUND = 4 # 背景上下文

@dataclass
class ContextItem:
    """上下文项"""
    content: str
    context_type: ContextType
    priority: ContextPriority
    file_path: str
    start_line: int
    end_line: int
    last_used: float
    access_count: int
    dependencies: Set[str]
    compressed: bool = False
    
    def __lt__(self, other):
        """优先级比较"""
        if not isinstance(other, ContextItem):
            return NotImplemented
        return (self.priority.value, -self.access_count, -self.last_used) < \
               (other.priority.value, -other.access_count, -other.last_used)

class ContextCache:
    """上下文缓存"""
    
    def __init__(self, cache_dir: Optional[str] = None, max_size: int = 1000):
        """
        初始化上下文缓存
        
        Args:
            cache_dir: 缓存目录
            max_size: 最大缓存项数
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".mcp_cache" / "context"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.items: Dict[str, ContextItem] = {}
        self.compressed_items: Dict[str, ContextItem] = {}
        self._load_cache()
        
    def _load_cache(self):
        """加载缓存"""
        try:
            cache_file = self.cache_dir / "context_cache.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item_data in data['items']:
                        item = ContextItem(
                            content=item_data['content'],
                            context_type=ContextType(item_data['type']),
                            priority=ContextPriority(item_data['priority']),
                            file_path=item_data['file_path'],
                            start_line=item_data['start_line'],
                            end_line=item_data['end_line'],
                            last_used=item_data['last_used'],
                            access_count=item_data['access_count'],
                            dependencies=set(item_data['dependencies']),
                            compressed=item_data['compressed']
                        )
                        if item.compressed:
                            self.compressed_items[self._get_key(item)] = item
                        else:
                            self.items[self._get_key(item)] = item
        except Exception as e:
            logger.error(f"加载上下文缓存失败: {str(e)}")
    
    def _save_cache(self):
        """保存缓存"""
        try:
            cache_file = self.cache_dir / "context_cache.json"
            items_data = []
            
            for item in list(self.items.values()) + list(self.compressed_items.values()):
                items_data.append({
                    'content': item.content,
                    'type': item.context_type.value,
                    'priority': item.priority.value,
                    'file_path': item.file_path,
                    'start_line': item.start_line,
                    'end_line': item.end_line,
                    'last_used': item.last_used,
                    'access_count': item.access_count,
                    'dependencies': list(item.dependencies),
                    'compressed': item.compressed
                })
                
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({'items': items_data}, f, indent=2)
        except Exception as e:
            logger.error(f"保存上下文缓存失败: {str(e)}")
    
    def _get_key(self, item: ContextItem) -> str:
        """生成缓存键"""
        return f"{item.file_path}:{item.start_line}-{item.end_line}"
    
    def get(self, file_path: str, start_line: int, end_line: int) -> Optional[ContextItem]:
        """获取上下文项"""
        key = f"{file_path}:{start_line}-{end_line}"
        
        # 先查找未压缩的缓存
        if key in self.items:
            item = self.items[key]
            item.last_used = time.time()
            item.access_count += 1
            return item
            
        # 再查找压缩的缓存
        if key in self.compressed_items:
            item = self.compressed_items[key]
            item.last_used = time.time()
            item.access_count += 1
            return item
            
        return None
    
    def put(self, item: ContextItem):
        """存储上下文项"""
        key = self._get_key(item)
        
        # 检查缓存大小
        if len(self.items) + len(self.compressed_items) >= self.max_size:
            self._evict_items()
            
        # 存储项
        if item.compressed:
            self.compressed_items[key] = item
        else:
            self.items[key] = item
            
        # 定期保存缓存
        if (len(self.items) + len(self.compressed_items)) % 100 == 0:
            self._save_cache()
    
    def _evict_items(self):
        """驱逐缓存项"""
        # 合并所有项
        all_items = list(self.items.values()) + list(self.compressed_items.values())
        
        # 按优先级、访问次数和最后访问时间排序
        sorted_items = sorted(all_items)
        
        # 移除低优先级项直到缓存大小合适
        while len(sorted_items) > self.max_size * 0.8:  # 保留80%的空间
            item = sorted_items.pop()
            key = self._get_key(item)
            if item.compressed:
                self.compressed_items.pop(key, None)
            else:
                self.items.pop(key, None)

class ContextManager:
    """上下文管理器"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化上下文管理器
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache = ContextCache(cache_dir)
        self.optimizer = CodeOptimizer()
        self.compressor = CodeCompressor()
        
    def get_context(self, file_path: str, line_number: int, 
                   context_type: ContextType = ContextType.FUNCTION,
                   priority: ContextPriority = ContextPriority.NORMAL) -> Optional[str]:
        """
        获取代码上下文
        
        Args:
            file_path: 文件路径
            line_number: 行号
            context_type: 上下文类型
            priority: 上下文优先级
            
        Returns:
            上下文内容
        """
        try:
            # 分析代码结构
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 获取文件语言
            ext = os.path.splitext(file_path)[1].lower()
            language = self._get_language(ext)
            
            # 分析代码
            analysis = self.optimizer.analyze_code(content, file_path, language)
            
            # 查找包含目标行的代码块
            target_block = None
            for block in analysis['blocks']:
                if block.start_line <= line_number <= block.end_line:
                    target_block = block
                    break
                    
            if not target_block:
                return None
                
            # 查找缓存
            context_item = self.cache.get(
                file_path, 
                target_block.start_line,
                target_block.end_line
            )
            
            if context_item:
                return context_item.content
                
            # 创建新的上下文项
            context_item = ContextItem(
                content=target_block.content,
                context_type=context_type,
                priority=priority,
                file_path=file_path,
                start_line=target_block.start_line,
                end_line=target_block.end_line,
                last_used=time.time(),
                access_count=1,
                dependencies=target_block.dependencies
            )
            
            # 根据优先级决定是否压缩
            if priority in [ContextPriority.LOW, ContextPriority.BACKGROUND]:
                context_item.content = self.compressor.compress(
                    context_item.content,
                    language
                )
                context_item.compressed = True
                
            # 缓存上下文
            self.cache.put(context_item)
            
            return context_item.content
            
        except Exception as e:
            logger.error(f"获取上下文失败 {file_path}:{line_number}: {str(e)}")
            return None
    
    def get_module_context(self, file_path: str,
                          priority: ContextPriority = ContextPriority.NORMAL) -> Optional[str]:
        """
        获取模块级上下文
        
        Args:
            file_path: 文件路径
            priority: 上下文优先级
            
        Returns:
            上下文内容
        """
        try:
            # 查找缓存
            context_item = self.cache.get(file_path, 1, -1)
            if context_item:
                return context_item.content
                
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 获取文件语言
            ext = os.path.splitext(file_path)[1].lower()
            language = self._get_language(ext)
            
            # 分析代码
            analysis = self.optimizer.analyze_code(content, file_path, language)
            
            # 创建新的上下文项
            context_item = ContextItem(
                content=content,
                context_type=ContextType.MODULE,
                priority=priority,
                file_path=file_path,
                start_line=1,
                end_line=len(content.splitlines()),
                last_used=time.time(),
                access_count=1,
                dependencies=set()
            )
            
            # 根据优先级决定是否压缩
            if priority in [ContextPriority.LOW, ContextPriority.BACKGROUND]:
                context_item.content = self.compressor.compress(
                    context_item.content,
                    language
                )
                context_item.compressed = True
                
            # 缓存上下文
            self.cache.put(context_item)
            
            return context_item.content
            
        except Exception as e:
            logger.error(f"获取模块上下文失败 {file_path}: {str(e)}")
            return None
    
    def _get_language(self, ext: str) -> str:
        """获取文件语言"""
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.vue': 'vue',
            '.php': 'php',
            '.rs': 'rust'
        }
        return language_map.get(ext, 'text')