"""
配置管理模块
负责加载、验证和提供应用配置
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Config:
    """
    配置管理类
    
    负责加载和管理应用配置，支持从配置文件、环境变量和默认值获取配置
    """
    
    # 默认配置
    DEFAULT_CONFIG = {
        "server": {
            "host": "127.0.0.1",
            "port": 5000,
            "debug": False,
            "workers": 4
        },
        "indexer": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 32,
            "max_tokens": 8192,
            "chunk_size": 1000,
            "chunk_overlap": 200
        },
        "storage": {
            "vector_db_path": "./vector_db",
            "project_data_path": "./project_data"
        },
        "exclude_patterns": [
            "node_modules/",
            "vendor/",
            ".git/",
            "venv/",
            "__pycache__/",
            "dist/",
            "build/",
            ".next/",
            "target/",
            ".idea/",
            ".vscode/",
            "logs/"
        ],
        "file_extensions": [
            ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp",
            ".cs", ".go", ".rb", ".php", ".swift", ".kt", ".rs", ".sh"
        ],
        "max_file_size_kb": 1024,
        "logging": {
            "level": "INFO",
            "file": "mcp_indexer.log"
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
            
        Returns:
            无返回值
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        # 加载配置文件
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
            
        # 从环境变量加载配置
        self._load_from_env()
        
        # 验证配置
        self._validate_config()
        
        logger.info(f"配置加载完成: {len(self.config)} 个配置项")
    
    def _load_from_file(self, config_path: str) -> None:
        """
        从YAML配置文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            无返回值
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._merge_config(self.config, file_config)
            logger.info(f"从配置文件加载配置: {config_path}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
    
    def _load_from_env(self) -> None:
        """
        从环境变量加载配置
        环境变量格式: MCP_SECTION_KEY=value
        
        Returns:
            无返回值
        """
        prefix = "MCP_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                parts = key[len(prefix):].lower().split('_', 1)
                if len(parts) == 2:
                    section, option = parts
                    if section in self.config:
                        if option in self.config[section]:
                            # 尝试转换值类型
                            try:
                                orig_type = type(self.config[section][option])
                                if orig_type == bool:
                                    self.config[section][option] = value.lower() in ('true', 'yes', '1')
                                elif orig_type == int:
                                    self.config[section][option] = int(value)
                                elif orig_type == float:
                                    self.config[section][option] = float(value)
                                else:
                                    self.config[section][option] = value
                            except Exception:
                                self.config[section][option] = value
                            
                            logger.debug(f"从环境变量加载配置: {key}={value}")
    
    def _merge_config(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        递归合并配置字典
        
        Args:
            target: 目标配置字典
            source: 源配置字典
            
        Returns:
            无返回值
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value
    
    def _validate_config(self) -> None:
        """
        验证配置有效性
        
        Returns:
            无返回值
        """
        # 确保必要的目录存在
        os.makedirs(self.get("storage.vector_db_path"), exist_ok=True)
        os.makedirs(self.get("storage.project_data_path"), exist_ok=True)
        
        # 验证嵌入模型配置
        embedding_model = self.get("indexer.embedding_model")
        if not embedding_model:
            logger.warning("未配置嵌入模型，将使用默认模型")
            self.config["indexer"]["embedding_model"] = self.DEFAULT_CONFIG["indexer"]["embedding_model"]
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key_path: 配置键路径，格式为"section.key"
            default: 默认值，如果配置不存在则返回此值
            
        Returns:
            配置值或默认值
        """
        parts = key_path.split('.')
        config = self.config
        
        for part in parts:
            if isinstance(config, dict) and part in config:
                config = config[part]
            else:
                return default
                
        return config
    
    def set(self, key_path: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key_path: 配置键路径，格式为"section.key"
            value: 要设置的值
            
        Returns:
            无返回值
        """
        parts = key_path.split('.')
        config = self.config
        
        for i, part in enumerate(parts[:-1]):
            if part not in config:
                config[part] = {}
            config = config[part]
            
        config[parts[-1]] = value
    
    def get_exclude_patterns(self) -> List[str]:
        """
        获取排除模式列表
        
        Returns:
            排除模式字符串列表
        """
        return self.get("exclude_patterns", [])
    
    def get_file_extensions(self) -> List[str]:
        """
        获取支持的文件扩展名列表
        
        Returns:
            文件扩展名字符串列表
        """
        return self.get("file_extensions", [])
    
    def save_to_file(self, file_path: str) -> bool:
        """
        将当前配置保存到文件
        
        Args:
            file_path: 保存的文件路径
            
        Returns:
            保存是否成功
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            return True
        except Exception as e:
            logger.error(f"保存配置文件失败: {str(e)}")
            return False