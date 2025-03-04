"""
索引核心模块
负责扫描和索引代码库，生成向量索引
"""

import os
import time
import threading
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from .config import Config
from .project_identity import ProjectIdentifier
from .code_optimizer import CodeOptimizer
from .code_compressor import CodeCompressor, NormalizationLevel
from .context_manager import ContextManager, ContextType, ContextPriority

# 配置默认的优化选项
DEFAULT_OPTIMIZATION_OPTIONS = {
    "use_ast_cache": True,
    "normalize_level": NormalizationLevel.NORMAL,
    "compress_threshold": 1000,  # 超过1000行的代码块会被压缩
    "cache_context": True,
    "parallel_processing": True
}

class IndexingStatus:
    """索引状态常量"""
    NEW = "new"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    UPDATING = "updating"

class CodeChunk:
    """代码块类，表示一个代码片段"""
    
    def __init__(self, content: str, file_path: str, start_line: int, end_line: int, 
                 language: str, type: str = "code"):
        """
        初始化代码块
        
        Args:
            content: 代码内容
            file_path: 文件路径
            start_line: 起始行号
            end_line: 结束行号
            language: 编程语言
            type: 代码块类型（代码、注释等）
            
        Returns:
            无返回值
        """
        self.content = content
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.language = language
        self.type = type
        
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            代码块字典表示
        """
        return {
            "content": self.content,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "type": self.type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeChunk':
        """
        从字典创建代码块
        
        Args:
            data: 代码块字典数据
            
        Returns:
            CodeChunk对象
        """
        return cls(
            content=data["content"],
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            language=data["language"],
            type=data.get("type", "code")
        )
    
    def get_id(self) -> str:
        """
        获取代码块唯一ID
        
        Returns:
            代码块ID字符串
        """
        id_str = f"{self.file_path}:{self.start_line}-{self.end_line}"
        return hashlib.md5(id_str.encode()).hexdigest()

class CodeIndexer:
    """
    代码索引器类
    
    负责扫描和索引代码库，生成向量索引
    """
    
    # 文件扩展名到语言的映射
    EXTENSION_TO_LANGUAGE = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".rs": "rust",
        ".sh": "bash",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".sql": "sql",
        ".md": "markdown",
        ".json": "json",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml"
    }
    
    def __init__(self, config: Config):
        """
        初始化代码索引器
        
        Args:
            config: 配置对象
            
        Returns:
            无返回值
        """
        self.config = config
        self.project_identifier = ProjectIdentifier(config)
        
        # 初始化向量数据库
        self.vector_db_path = Path(config.get("storage.vector_db_path"))
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化状态存储路径
        self.status_path = self.vector_db_path / "indexing_status"
        self.status_path.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vector_db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 初始化嵌入模型
        self.embedding_model_name = config.get("indexer.embedding_model")
        self.embedding_model = None  # 延迟加载
        
        # 初始化优化器和管理器
        self.optimizer = CodeOptimizer()
        self.compressor = CodeCompressor()
        self.context_manager = ContextManager()
        
        # 获取优化选项
        self.optimization_options = config.get("indexer.optimization_options", DEFAULT_OPTIMIZATION_OPTIONS)
        
        # 索引状态
        self.indexing_status = {}
        self.indexing_lock = threading.Lock()
        
        # 从磁盘加载索引状态
        self._load_indexing_status()
        
        # 并行处理配置
        self.max_workers = os.cpu_count() if self.optimization_options["parallel_processing"] else 1
    
    def _load_embedding_model(self) -> None:
        """
        加载嵌入模型
        
        Returns:
            无返回值
        """
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
    
    def index_project(self, project_path: str,
                     progress_callback: Optional[Callable[[str, float], None]] = None,
                     force_reindex: bool = False) -> str:
        """
        索引项目，支持增量索引
        
        Args:
            project_path: 项目路径
            progress_callback: 进度回调函数，接收状态和进度百分比
            force_reindex: 是否强制重新索引
            
        Returns:
            项目ID
        """
        # 识别项目
        project_id, is_new, metadata = self.project_identifier.identify_project(project_path)
        
        # 验证现有索引
        if not force_reindex and not is_new:
            if self._verify_index(project_id, project_path):
                return project_id
        
        # 检查是否已在索引中
        with self.indexing_lock:
            if project_id in self.indexing_status:
                status = self.indexing_status[project_id]
                if status == IndexingStatus.INDEXING or status == IndexingStatus.UPDATING:
                    return project_id
            
            # 设置索引状态并保存
            status = IndexingStatus.NEW if is_new else IndexingStatus.UPDATING
            self.indexing_status[project_id] = status
            self._save_indexing_status()
        
        # 启动索引线程
        threading.Thread(
            target=self._index_project_thread,
            args=(project_id, project_path, progress_callback),
            daemon=True
        ).start()
        
        return project_id

    def _verify_index(self, project_id: str, project_path: str) -> bool:
        """
        验证项目索引是否有效
        
        Args:
            project_id: 项目ID
            project_path: 项目路径
            
        Returns:
            索引是否有效
        """
        try:
            # 检查元数据文件
            metadata_file = self.vector_db_path / f"metadata_{project_id}.json"
            if not metadata_file.exists():
                return False
            
            # 读取元数据
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 验证项目路径
            if metadata.get("project_path") != project_path:
                return False
            
            # 验证向量数据库集合
            collection_name = f"project_{project_id}"
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
                if collection.count() == 0:
                    return False
            except:
                return False
            
            # 获取已索引文件的修改时间
            indexed_files = metadata.get("indexed_files", {})
            
            # 扫描当前项目文件
            current_files = self._scan_project_files(project_path)
            
            # 检查文件变化
            for file_path in current_files:
                mtime = os.path.getmtime(file_path)
                if file_path not in indexed_files or mtime > indexed_files[file_path]:
                    return False
            
            return True
            
        except:
            return False
    
    def _index_project_thread(self, project_id: str, project_path: str,
                              progress_callback: Optional[Callable[[str, float], None]]) -> None:
        """
        索引项目线程
        
        Args:
            project_id: 项目ID
            project_path: 项目路径
            progress_callback: 进度回调函数
            
        Returns:
            无返回值
        """
        try:
            # 更新状态并保存
            with self.indexing_lock:
                self.indexing_status[project_id] = IndexingStatus.INDEXING
                self._save_indexing_status()
            
            if progress_callback:
                progress_callback(IndexingStatus.INDEXING, 0.0)
            
            # 加载嵌入模型
            self._load_embedding_model()
            
            # 扫描项目文件
            files = self._scan_project_files(project_path)
            
            if progress_callback:
                progress_callback(IndexingStatus.INDEXING, 0.1)
            
            # 处理文件
            total_files = len(files)
            processed_files = 0
            
            # 获取或创建集合 - 删除原来的清除数据操作，移到 _get_or_create_collection 中
            collection = self._get_or_create_collection(project_id)

            # 批量处理
            batch_size = self.config.get("indexer.batch_size", 32)
            chunks_batch = []
            ids_batch = []
            embeddings_batch = []
            metadatas_batch = []
            
            # 优化线程池配置，根据CPU核心数和文件数量动态调整
            optimal_workers = min(os.cpu_count() or 4, max(4, len(files) // 10))
            logging.info(f"使用 {optimal_workers} 个线程处理 {len(files)} 个文件")
            
            # 增加批处理大小以减少数据库操作次数
            batch_size = self.config.get("indexer.batch_size", 32)
            # 对大型项目使用更大的批处理大小
            if len(files) > 1000:
                batch_size = max(batch_size, 64)
            
            # 使用线程池并行处理文件
            with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                # 分批提交任务，避免一次性创建过多线程
                batch_files = [files[i:i+100] for i in range(0, len(files), 100)]
                completed_files = 0
                
                for file_batch in batch_files:
                    future_to_file = {executor.submit(self._process_file, file_path): file_path for file_path in file_batch}
                    
                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            file_chunks = future.result()
                            
                            if file_chunks:
                                for chunk in file_chunks:
                                    chunks_batch.append(chunk.content)
                                    ids_batch.append(chunk.get_id())
                                    metadatas_batch.append({
                                        "file_path": chunk.file_path,
                                        "start_line": chunk.start_line,
                                        "end_line": chunk.end_line,
                                        "language": chunk.language,
                                        "type": chunk.type,
                                        "project_id": project_id,
                                        "chunk_data": json.dumps(chunk.to_dict())
                                    })
                                
                                # 当达到批处理大小时，生成嵌入并添加到数据库
                                if len(chunks_batch) >= batch_size:
                                    try:
                                        # 使用更高效的批量编码方法
                                        embeddings = self.embedding_model.encode(
                                            chunks_batch,
                                            batch_size=batch_size,
                                            show_progress_bar=False,
                                            convert_to_tensor=False,  # 直接返回numpy数组，避免额外转换
                                            normalize_embeddings=True  # 预先归一化，提高后续搜索效率
                                        ).tolist()
                                        
                                        embeddings_batch.extend(embeddings)
                                        
                                        # 使用批量添加，减少数据库操作次数
                                        collection.add(
                                            documents=chunks_batch,
                                            embeddings=embeddings_batch,
                                            metadatas=metadatas_batch,
                                            ids=ids_batch
                                        )
                                        
                                        chunks_batch = []
                                        ids_batch = []
                                        embeddings_batch = []
                                        metadatas_batch = []
                                    except Exception as e:
                                        logging.error(f"添加向量数据失败: {str(e)}")
                                        # 继续处理，不中断索引过程
                            
                            completed_files += 1
                            if progress_callback:
                                progress = 0.1 + (completed_files / total_files) * 0.8
                                progress_callback(IndexingStatus.INDEXING, progress)
                                
                        except Exception as e:
                            logging.error(f"处理文件失败 {file_path}: {str(e)}")
                        
                        # 这里已经在上面的try-except块中处理了进度更新，不需要重复
            
            # 处理剩余的批次
            if chunks_batch:
                embeddings = self.embedding_model.encode(chunks_batch).tolist()
                embeddings_batch.extend(embeddings)
                
                collection.add(
                    documents=chunks_batch,
                    embeddings=embeddings_batch,
                    metadatas=metadatas_batch,
                    ids=ids_batch
                )
            
            # 保存索引元数据
            self._save_index_metadata(project_id, project_path, total_files)
            
            # 更新状态并保存
            with self.indexing_lock:
                self.indexing_status[project_id] = IndexingStatus.COMPLETED
                self._save_indexing_status()
            
            if progress_callback:
                progress_callback(IndexingStatus.COMPLETED, 1.0)
            
        except:
            # 更新状态并保存
            with self.indexing_lock:
                self.indexing_status[project_id] = IndexingStatus.FAILED
                self._save_indexing_status()
            
            if progress_callback:
                progress_callback(IndexingStatus.FAILED, 0.0)
    
    def _scan_project_files(self, project_path: str) -> List[str]:
        """
        扫描项目文件
        
        Args:
            project_path: 项目路径
            
        Returns:
            文件路径列表
        """
        files = []
        exclude_patterns = self.config.get_exclude_patterns()
        file_extensions = self.config.get_file_extensions()
        max_file_size_kb = self.config.get("max_file_size_kb", 1024)
        
        # 检查.mcp_ignore文件
        ignore_patterns = []
        ignore_file = os.path.join(project_path, ".mcp_ignore")
        if os.path.exists(ignore_file):
            try:
                with open(ignore_file, 'r', encoding='utf-8') as f:
                    ignore_patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            except:
                pass
        
        # 合并排除模式
        all_exclude_patterns = exclude_patterns + ignore_patterns
        
        for root, dirs, file_names in os.walk(project_path):
            # 排除忽略的目录
            dirs[:] = [d for d in dirs if not self._should_ignore(os.path.join(root, d), all_exclude_patterns)]
            
            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                
                # 检查是否应该忽略
                if self._should_ignore(file_path, all_exclude_patterns):
                    continue
                
                # 检查文件扩展名
                ext = os.path.splitext(file_name)[1].lower()
                if file_extensions and ext not in file_extensions:
                    continue
                
                # 检查文件大小
                try:
                    if os.path.getsize(file_path) > max_file_size_kb * 1024:
                        continue
                except:
                    continue
                
                files.append(file_path)
        
        return files
    
    def _should_ignore(self, path: str, exclude_patterns: List[str]) -> bool:
        """
        检查路径是否应该被忽略
        
        Args:
            path: 文件或目录路径
            exclude_patterns: 排除模式列表
            
        Returns:
            如果应该忽略则返回True，否则返回False
        """
        rel_path = os.path.basename(path)
        
        # 检查是否匹配排除模式
        for pattern in exclude_patterns:
            if pattern.endswith('/'):
                pattern = pattern[:-1]
            if rel_path == pattern or rel_path.startswith(pattern + os.path.sep):
                return True
        
        return False
    
    def _process_file(self, file_path: str) -> List[CodeChunk]:
        """
        处理单个文件，提取代码块
        
        Args:
            file_path: 文件路径
            
        Returns:
            代码块列表
        """
        try:
            # 检查上下文缓存
            if self.optimization_options["cache_context"]:
                cached_context = self.context_manager.get_module_context(
                    file_path,
                    priority=ContextPriority.NORMAL
                )
                if cached_context:
                    content = cached_context
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
            else:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            
            # 获取文件语言
            ext = os.path.splitext(file_path)[1].lower()
            language = self.EXTENSION_TO_LANGUAGE.get(ext, "text")
            
            # 优化代码
            analysis = self.optimizer.analyze_code(content, file_path, language)
            
            chunks = []
            for block in analysis['blocks']:
                # 根据代码块大小决定是否压缩
                block_content = block.content
                if len(block_content.splitlines()) > self.optimization_options["compress_threshold"]:
                    block_content = self.compressor.compress(
                        block_content,
                        language,
                        level=self.optimization_options["normalize_level"]
                    )
                
                chunks.append(CodeChunk(
                    content=block_content,
                    file_path=file_path,
                    start_line=block.start_line,
                    end_line=block.end_line,
                    language=language,
                    type=block.block_type.value
                ))
            
            return chunks
            
        except:
            return []

    def _split_into_chunks(self, content: str, file_path: str, language: str) -> List[CodeChunk]:
        """
        将文件内容分割为代码块，使用代码分析器进行智能分块
        
        Args:
            content: 文件内容
            file_path: 文件路径
            language: 编程语言
            
        Returns:
            代码块列表
        """
        from .code_analyzer import CodeAnalyzer
        
        chunks = []
        analyzer = CodeAnalyzer()
        
        # 使用代码分析器分析代码结构
        analysis = analyzer.analyze_code(content, language)
        
        # 基于函数定义创建代码块
        for func in analysis["functions"]:
            start_line = func["start_line"]
            end_line = func["end_line"]
            chunk_content = '\n'.join(content.split('\n')[start_line:end_line+1])
            
            if chunk_content.strip():
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                    language=language,
                    type="function"
                ))
        
        # 基于类定义创建代码块
        for cls in analysis["classes"]:
            start_line = cls["start_line"]
            end_line = cls["end_line"]
            chunk_content = '\n'.join(content.split('\n')[start_line:end_line+1])
            
            if chunk_content.strip():
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                    language=language,
                    type="class"
                ))
        
        # 如果没有找到函数或类，回退到基于行的分块
        if not chunks:
            chunk_size = self.config.get("indexer.chunk_size", 1000)
            chunk_overlap = self.config.get("indexer.chunk_overlap", 200)
            lines = content.split('\n')
            i = 0
            
            while i < len(lines):
                end = min(i + chunk_size, len(lines))
                chunk_content = '\n'.join(lines[i:end])
                
                if chunk_content.strip():
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        start_line=i + 1,
                        end_line=end,
                        language=language,
                        type="code"
                    ))
                
                i += chunk_size - chunk_overlap
        
        return chunks
    
    def _get_or_create_collection(self, project_id: str) -> Any:
        """
        获取或创建向量集合
        
        Args:
            project_id: 项目ID
            
        Returns:
            ChromaDB集合对象
        """
        collection_name = f"project_{project_id}"
        
        try:
            # 尝试获取现有集合
            collection = self.chroma_client.get_collection(name=collection_name)
            # 清除现有数据（如果是更新）- 修复空 where 条件的问题
            collection.delete(where={"project_id": project_id})
            return collection
        except:
            # 创建新集合
            return self.chroma_client.create_collection(name=collection_name)
    
    def _save_index_metadata(self, project_id: str, project_path: str, file_count: int) -> None:
        """
        保存索引元数据，包括文件修改时间
        
        Args:
            project_id: 项目ID
            project_path: 项目路径
            file_count: 文件数量
            
        Returns:
            无返回值
        """
        metadata_file = self.vector_db_path / f"metadata_{project_id}.json"
        
        # 获取所有文件的最后修改时间
        indexed_files = {}
        files = self._scan_project_files(project_path)
        for file_path in files:
            try:
                indexed_files[file_path] = os.path.getmtime(file_path)
            except:
                continue
        
        metadata = {
            "project_id": project_id,
            "project_path": project_path,
            "indexed_at": datetime.now().isoformat(),
            "file_count": file_count,
            "embedding_model": {
                "name": "bge-large-zh",  # 更强大的中文代码理解模型
                "dimension": 1024,
                "quantization": "int8",  # 使用量化减少内存占用
                "batch_size": 32,
                "max_length": 2048  # 支持更长的代码块
            },
            "indexed_files": indexed_files,
            "languages": list(self.EXTENSION_TO_LANGUAGE.values()),
            "version": "0.2.0"  # 添加版本号便于后续升级
        }
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        except:
            pass
    
    def _load_indexing_status(self) -> None:
        """
        从磁盘加载索引状态
        
        Returns:
            无返回值
        """
        try:
            status_file = self.status_path / "status.json"
            if status_file.exists():
                with open(status_file, 'r', encoding='utf-8') as f:
                    self.indexing_status = json.load(f)
                logging.info(f"从磁盘加载了 {len(self.indexing_status)} 个项目的索引状态")
        except Exception as e:
            logging.error(f"加载索引状态失败: {str(e)}")
            self.indexing_status = {}
    
    def _save_indexing_status(self) -> None:
        """
        将索引状态保存到磁盘
        
        Returns:
            无返回值
        """
        try:
            status_file = self.status_path / "status.json"
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(self.indexing_status, f, indent=2)
            logging.info(f"保存了 {len(self.indexing_status)} 个项目的索引状态到磁盘")
        except Exception as e:
            logging.error(f"保存索引状态失败: {str(e)}")
    
    def get_indexing_status(self, project_id: str) -> Tuple[str, float]:
        """
        获取项目索引状态
        
        Args:
            project_id: 项目ID
            
        Returns:
            元组(状态, 进度)
        """
        with self.indexing_lock:
            status = self.indexing_status.get(project_id, IndexingStatus.NEW)
        
        # 如果已完成，进度为1.0，否则为0.0
        progress = 1.0 if status == IndexingStatus.COMPLETED else 0.0
        
        # 检查元数据文件是否存在，如果存在且状态为NEW，则更新为COMPLETED
        metadata_file = self.vector_db_path / f"metadata_{project_id}.json"
        if metadata_file.exists() and status == IndexingStatus.NEW:
            try:
                # 验证向量数据库集合
                collection_name = f"project_{project_id}"
                collection = self.chroma_client.get_collection(name=collection_name)
                if collection.count() > 0:
                    with self.indexing_lock:
                        self.indexing_status[project_id] = IndexingStatus.COMPLETED
                        self._save_indexing_status()
                    status = IndexingStatus.COMPLETED
                    progress = 1.0
            except Exception as e:
                logging.error(f"验证索引状态失败: {str(e)}")
        
        return status, progress
    
    # 查询缓存
    _query_cache = {}
    _cache_lock = threading.Lock()
    _cache_max_size = 100  # 最大缓存条目数
    _cache_ttl = 300  # 缓存有效期（秒）
    
    def search(self, project_id: str, query: str, limit: int = 10, timeout: int = 30) -> List[Dict[str, Any]]:
        """
        搜索代码
        
        Args:
            project_id: 项目ID
            query: 查询字符串
            limit: 返回结果数量限制
            timeout: 搜索超时时间（秒）
            
        Returns:
            代码块字典列表
        """
        # 获取logger
        logger = logging.getLogger(__name__)
        
        # 生成缓存键
        cache_key = f"{project_id}:{query}:{limit}"
        
        # 检查缓存
        with self._cache_lock:
            if cache_key in self._query_cache:
                cache_entry = self._query_cache[cache_key]
                # 检查缓存是否过期
                if time.time() - cache_entry['timestamp'] < self._cache_ttl:
                    logger.info(f"使用缓存结果: {cache_key}")
                    return cache_entry['results']
        
        # 加载嵌入模型
        self._load_embedding_model()
        
        collection_name = f"project_{project_id}"
        
        try:
            # 设置超时
            start_time = time.time()
            
            # 获取集合
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
            except Exception as e:
                logger.error(f"获取集合失败: {str(e)}")
                return []
            
            # 检查超时
            if time.time() - start_time > timeout:
                logger.warning(f"搜索超时: 获取集合阶段")
                return []
            
            # 生成查询嵌入
            try:
                # 使用更高效的编码设置
                query_embedding = self.embedding_model.encode(
                    query,
                    show_progress_bar=False,
                    convert_to_tensor=False,
                    normalize_embeddings=True
                ).tolist()
            except Exception as e:
                logger.error(f"生成查询嵌入失败: {str(e)}")
                return []
            
            # 检查超时
            if time.time() - start_time > timeout:
                logger.warning(f"搜索超时: 生成嵌入阶段")
                return []
            
            # 执行搜索
            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                    where={"project_id": project_id}
                )
            except Exception as e:
                logger.error(f"执行查询失败: {str(e)}")
                return []
            
            # 检查超时
            if time.time() - start_time > timeout:
                logger.warning(f"搜索超时: 执行查询阶段")
                return []
            
            # 处理结果
            code_chunks = []
            if results and results['metadatas']:
                for i, metadata in enumerate(results['metadatas'][0]):
                    try:
                        chunk_data = json.loads(metadata.get('chunk_data', '{}'))
                        chunk = CodeChunk.from_dict(chunk_data)
                        
                        # 添加相似度分数
                        result_dict = chunk.to_dict()
                        if 'distances' in results and len(results['distances']) > 0:
                            result_dict['similarity'] = 1.0 - results['distances'][0][i]
                        
                        code_chunks.append(result_dict)
                    except Exception as e:
                        logger.error(f"处理结果失败: {str(e)}")
            
            # 更新缓存
            with self._cache_lock:
                # 如果缓存已满，删除最旧的条目
                if len(self._query_cache) >= self._cache_max_size:
                    oldest_key = min(self._query_cache.keys(), key=lambda k: self._query_cache[k]['timestamp'])
                    del self._query_cache[oldest_key]
                
                # 添加新的缓存条目
                self._query_cache[cache_key] = {
                    'results': code_chunks,
                    'timestamp': time.time()
                }
            
            return code_chunks
            
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return []
    
    def delete_project_index(self, project_id: str) -> bool:
        """
        删除项目索引
        
        Args:
            project_id: 项目ID
            
        Returns:
            删除是否成功
        """
        collection_name = f"project_{project_id}"
        
        try:
            # 删除集合
            self.chroma_client.delete_collection(name=collection_name)
            
            # 删除元数据
            metadata_file = self.vector_db_path / f"metadata_{project_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            # 更新状态
            with self.indexing_lock:
                if project_id in self.indexing_status:
                    del self.indexing_status[project_id]
            
            return True
        except:
            return False
    
    def get_indexed_projects(self) -> List[Dict[str, Any]]:
        """
        获取所有已索引的项目
        
        Returns:
            项目信息字典列表
        """
        projects = []
        
        # 从元数据文件获取项目信息
        for metadata_file in self.vector_db_path.glob("metadata_*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                    # 获取索引状态
                    project_id = metadata.get("project_id")
                    status, progress = self.get_indexing_status(project_id)
                    
                    projects.append({
                        "project_id": project_id,
                        "project_path": metadata.get("project_path"),
                        "indexed_at": metadata.get("indexed_at"),
                        "file_count": metadata.get("file_count"),
                        "status": status,
                        "progress": progress
                    })
            except:
                pass
        
        return projects