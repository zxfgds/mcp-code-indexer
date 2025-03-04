"""
MCP代码检索工具安装脚本
用于安装MCP代码检索工具及其依赖
"""

from setuptools import setup, find_packages
import os

# Check if README.md exists, if not use a default description
readme_path = "README.md"
long_description = "MCP Code Indexer - A code search and indexing tool"
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="mcp_code_indexer",
    version="0.1.0",
    author="MCP Team",
    author_email="example@example.com",
    description="基于MCP协议的代码检索工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/mcp_code_indexer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "flask>=2.0.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "torch>=1.10.0",
        "tree-sitter>=0.20.0",
        "pygments>=2.10.0",
        # "mcp-protocol>=0.1.0",  # Removed as not available on PyPI
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.19.0",
    ],
    entry_points={
        "console_scripts": [
            "mcp-indexer=mcp_code_indexer.client.cli:main",
            "mcp-server=mcp_code_indexer.server.app:main",
        ],
    },
)