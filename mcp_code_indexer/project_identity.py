"""
项目识别模块
负责生成和识别项目唯一标识，确保跨会话识别同一项目
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from .config import Config

logger = logging.getLogger(__name__)

class ProjectIdentifier:
    """
    项目识别器类
    
    用于生成和识别项目唯一标识，通过多种机制确保跨会话识别同一项目
    """
    
    # 项目标识文件名
    PROJECT_ID_FILE = ".mcp_project"
    
    # 关键文件列表，用于生成项目指纹
    KEY_FILES = [
        "package.json",
        "setup.py",
        "pom.xml",
        "build.gradle",
        "Cargo.toml",
        "go.mod",
        "Gemfile",
        "composer.json",
        ".gitignore",
        "README.md",
        "Makefile",
        "CMakeLists.txt"
    ]
    
    def __init__(self, config: Config):
        """
        初始化项目识别器
        
        Args:
            config: 配置对象
            
        Returns:
            无返回值
        """
        self.config = config
        self.project_data_path = Path(config.get("storage.project_data_path"))
        self.project_data_path.mkdir(parents=True, exist_ok=True)
    
    def identify_project(self, project_path: str) -> Tuple[str, bool, Dict]:
        """
        识别项目，返回项目ID和是否为新项目
        
        Args:
            project_path: 项目路径
            
        Returns:
            元组(项目ID, 是否为新项目, 项目元数据)
        """
        project_path = os.path.abspath(project_path)
        logger.info(f"识别项目: {project_path}")
        
        # 检查项目标识文件
        id_file_path = os.path.join(project_path, self.PROJECT_ID_FILE)
        project_id = None
        is_new = False
        metadata = {}
        
        if os.path.exists(id_file_path):
            # 从标识文件读取项目ID
            try:
                with open(id_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    project_id = data.get('project_id')
                    metadata = data.get('metadata', {})
                logger.info(f"从标识文件读取项目ID: {project_id}")
            except Exception as e:
                logger.error(f"读取项目标识文件失败: {str(e)}")
        
        # 生成项目指纹
        fingerprint = self._generate_fingerprint(project_path)
        
        # 如果没有项目ID或指纹不匹配，则生成新ID
        if not project_id:
            project_id = self._generate_project_id(project_path, fingerprint)
            is_new = True
            metadata = self._collect_project_metadata(project_path)
            self._save_project_id(project_path, project_id, fingerprint, metadata)
            logger.info(f"生成新项目ID: {project_id}")
        else:
            # 验证指纹是否匹配
            stored_fingerprint = self._get_stored_fingerprint(project_id)
            if stored_fingerprint and stored_fingerprint != fingerprint:
                logger.warning(f"项目指纹不匹配，可能是项目已更改")
                # 更新指纹
                self._update_project_fingerprint(project_id, fingerprint)
        
        return project_id, is_new, metadata
    
    def _generate_fingerprint(self, project_path: str) -> str:
        """
        生成项目指纹
        
        基于关键文件内容和目录结构生成唯一指纹
        
        Args:
            project_path: 项目路径
            
        Returns:
            项目指纹字符串
        """
        fingerprint_data = []
        
        # 添加关键文件内容的哈希
        for key_file in self.KEY_FILES:
            file_path = os.path.join(project_path, key_file)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                        fingerprint_data.append(f"{key_file}:{file_hash}")
                except Exception as e:
                    logger.error(f"读取文件失败 {file_path}: {str(e)}")
        
        # 添加目录结构信息
        dir_structure = []
        for root, dirs, files in os.walk(project_path, topdown=True):
            # 排除忽略的目录
            dirs[:] = [d for d in dirs if not self._should_ignore(os.path.join(root, d))]
            
            rel_path = os.path.relpath(root, project_path)
            if rel_path == '.':
                rel_path = ''
                
            # 只使用目录名和文件数量，不包含具体文件名
            dir_info = f"{rel_path}:{len(files)}"
            dir_structure.append(dir_info)
            
            # 限制目录结构大小，避免过大
            if len(dir_structure) > 100:
                break
        
        fingerprint_data.extend(dir_structure)
        
        # 生成最终指纹
        fingerprint_str = "\n".join(sorted(fingerprint_data))
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()
    
    def _should_ignore(self, path: str) -> bool:
        """
        检查路径是否应该被忽略
        
        Args:
            path: 文件或目录路径
            
        Returns:
            如果应该忽略则返回True，否则返回False
        """
        rel_path = os.path.basename(path)
        exclude_patterns = self.config.get_exclude_patterns()
        
        # 检查是否匹配排除模式
        for pattern in exclude_patterns:
            if pattern.endswith('/'):
                pattern = pattern[:-1]
            if rel_path == pattern or rel_path.startswith(pattern + os.path.sep):
                return True
        
        return False
    
    def _generate_project_id(self, project_path: str, fingerprint: str) -> str:
        """
        生成项目ID
        
        Args:
            project_path: 项目路径
            fingerprint: 项目指纹
            
        Returns:
            项目ID字符串
        """
        # 使用项目名称和指纹生成ID
        project_name = os.path.basename(project_path)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        id_str = f"{project_name}_{fingerprint[:8]}_{timestamp}"
        return hashlib.md5(id_str.encode()).hexdigest()
    
    def _save_project_id(self, project_path: str, project_id: str, 
                         fingerprint: str, metadata: Dict) -> None:
        """
        保存项目ID到项目目录和存储
        
        Args:
            project_path: 项目路径
            project_id: 项目ID
            fingerprint: 项目指纹
            metadata: 项目元数据
            
        Returns:
            无返回值
        """
        # 保存到项目目录
        id_file_path = os.path.join(project_path, self.PROJECT_ID_FILE)
        data = {
            'project_id': project_id,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata
        }
        
        try:
            with open(id_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"保存项目标识文件失败: {str(e)}")
        
        # 保存到存储
        self._save_project_data(project_id, {
            'project_id': project_id,
            'project_path': project_path,
            'fingerprint': fingerprint,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'metadata': metadata
        })
    
    def _save_project_data(self, project_id: str, data: Dict) -> None:
        """
        保存项目数据到存储
        
        Args:
            project_id: 项目ID
            data: 项目数据
            
        Returns:
            无返回值
        """
        project_file = self.project_data_path / f"{project_id}.json"
        try:
            with open(project_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"保存项目数据失败: {str(e)}")
    
    def _get_stored_fingerprint(self, project_id: str) -> Optional[str]:
        """
        获取存储的项目指纹
        
        Args:
            project_id: 项目ID
            
        Returns:
            项目指纹字符串，如果不存在则返回None
        """
        project_file = self.project_data_path / f"{project_id}.json"
        if not project_file.exists():
            return None
            
        try:
            with open(project_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('fingerprint')
        except Exception as e:
            logger.error(f"读取项目数据失败: {str(e)}")
            return None
    
    def _update_project_fingerprint(self, project_id: str, fingerprint: str) -> None:
        """
        更新项目指纹
        
        Args:
            project_id: 项目ID
            fingerprint: 新的项目指纹
            
        Returns:
            无返回值
        """
        project_file = self.project_data_path / f"{project_id}.json"
        if not project_file.exists():
            logger.error(f"项目数据不存在: {project_id}")
            return
            
        try:
            with open(project_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data['fingerprint'] = fingerprint
            data['updated_at'] = datetime.now().isoformat()
            
            with open(project_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"更新项目指纹: {project_id}")
        except Exception as e:
            logger.error(f"更新项目指纹失败: {str(e)}")
    
    def _collect_project_metadata(self, project_path: str) -> Dict:
        """
        收集项目元数据
        
        Args:
            project_path: 项目路径
            
        Returns:
            项目元数据字典
        """
        metadata = {
            'name': os.path.basename(project_path),
            'path': project_path,
            'size': 0,
            'file_count': 0,
            'languages': {},
            'key_files': []
        }
        
        # 统计文件数量和大小
        for root, _, files in os.walk(project_path):
            if self._should_ignore(root):
                continue
                
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path) and not self._should_ignore(file_path):
                    metadata['file_count'] += 1
                    try:
                        metadata['size'] += os.path.getsize(file_path)
                    except:
                        pass
                    
                    # 统计语言
                    ext = os.path.splitext(file)[1].lower()
                    if ext:
                        metadata['languages'][ext] = metadata['languages'].get(ext, 0) + 1
                    
                    # 记录关键文件
                    if file in self.KEY_FILES:
                        rel_path = os.path.relpath(file_path, project_path)
                        metadata['key_files'].append(rel_path)
        
        return metadata
    
    def get_all_projects(self) -> List[Dict]:
        """
        获取所有已索引的项目
        
        Returns:
            项目信息字典列表
        """
        projects = []
        for project_file in self.project_data_path.glob("*.json"):
            try:
                with open(project_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    projects.append({
                        'project_id': data.get('project_id'),
                        'project_path': data.get('project_path'),
                        'created_at': data.get('created_at'),
                        'updated_at': data.get('updated_at'),
                        'metadata': data.get('metadata', {})
                    })
            except Exception as e:
                logger.error(f"读取项目数据失败 {project_file}: {str(e)}")
        
        return projects
    
    def get_project_by_id(self, project_id: str) -> Optional[Dict]:
        """
        根据ID获取项目信息
        
        Args:
            project_id: 项目ID
            
        Returns:
            项目信息字典，如果不存在则返回None
        """
        project_file = self.project_data_path / f"{project_id}.json"
        if not project_file.exists():
            return None
            
        try:
            with open(project_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取项目数据失败: {str(e)}")
            return None
    
    def delete_project(self, project_id: str) -> bool:
        """
        删除项目数据
        
        Args:
            project_id: 项目ID
            
        Returns:
            删除是否成功
        """
        project_file = self.project_data_path / f"{project_id}.json"
        if not project_file.exists():
            return False
            
        try:
            project_file.unlink()
            logger.info(f"删除项目数据: {project_id}")
            return True
        except Exception as e:
            logger.error(f"删除项目数据失败: {str(e)}")
            return False