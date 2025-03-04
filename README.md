# MCP代码索引器 (MCP Code Indexer)

基于模型上下文协议(Model Context Protocol)的智能代码检索工具，为AI大语言模型提供高效精确的代码库检索能力。

[English Version](README_EN.md)

## 项目简介

MCP代码索引器是一个专为AI大语言模型设计的代码检索工具。它通过向量化索引和语义理解，帮助AI更好地理解和分析代码库，显著提升代码相关任务的处理效率和准确性。

项目地址：https://github.com/zxfgds/mcp-code-indexer

### 主要特性

- **智能代码检索**
  - 基于语义的代码搜索，理解代码含义而不仅是关键字匹配
  - 支持跨语言代码检索
  - 智能代码片段提取，自动识别完整的代码上下文

- **代码分析能力**
  - 代码结构分析（类、函数、依赖关系）
  - 代码质量评估
  - 文档和注释提取
  - 相似代码检测
  - 代码度量统计
  - 项目依赖分析

- **优化的上下文处理**
  - 智能Token消耗控制，减少不必要的上下文信息
  - 结构化的代码上下文提供
  - 多项目并行索引和检索支持
  - 持久化项目识别，避免重复索引

- **完整MCP协议支持**
  - 符合MCP数据交互规范
  - 提供丰富的工具集
  - 支持异步操作和进度反馈

## 应用场景

### Claude Desktop应用

- **增强代码理解**：帮助Claude更准确地理解用户的代码库，提供更精准的建议和解答
- **智能代码导航**：快速定位相关代码片段，提高问答效率
- **代码质量改进**：通过代码分析功能，提供更专业的代码优化建议
- **项目依赖管理**：帮助理解和管理项目依赖关系
- **文档生成辅助**：基于代码注释和结构分析，协助生成技术文档

### VSCode扩展

- **实时代码分析**：在编辑器中直接获取代码分析结果
- **智能代码推荐**：基于项目上下文提供更准确的代码建议
- **重构辅助**：识别可优化的代码模式，辅助代码重构
- **依赖关系可视化**：直观展示代码依赖关系
- **团队协作增强**：帮助团队成员更好地理解代码库

## 安装说明

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 安装工具
```bash
python setup.py install
```

## 配置说明

### Claude Desktop配置

编辑配置文件：`%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "code-indexer": {
      "command": "python",
      "args": ["-m", "server.app"],
      "cwd": "安装目录路径",
      "env": {},
      "disabled": false,
      "alwaysAllow": []
    }
  }
}
```

### VSCode扩展配置

编辑配置文件：`%APPDATA%\Code\User\globalStorage\rooveterinaryinc.roo-cline\settings\cline_mcp_settings.json`

```json
{
  "mcpServers": {
    "code-indexer": {
      "command": "python",
      "args": ["-m", "server.app"],
      "cwd": "安装目录路径",
      "env": {},
      "disabled": false,
      "alwaysAllow": []
    }
  }
}
```

## 使用说明

### 基本功能

1. 项目识别
```
使用identify_project工具识别项目
```

2. 索引项目
```
使用index_project工具索引项目代码
```

3. 搜索代码
```
使用search_code工具搜索相关代码片段
```

### 高级功能

1. 获取代码结构
```
使用get_code_structure工具分析代码结构
```

2. 分析代码质量
```
使用analyze_code_quality工具评估代码质量
```

3. 提取文档
```
使用extract_documentation工具提取代码文档
```

4. 查找相似代码
```
使用find_similar_code工具检测相似代码
```

5. 获取代码度量
```
使用get_code_metrics工具获取代码统计数据
```

6. 分析依赖关系
```
使用analyze_dependencies工具分析项目依赖
```

## 项目价值

1. **提升AI代码理解能力**
   - 更准确的代码语义理解
   - 更全面的项目上下文把握
   - 更智能的代码相关建议

2. **优化开发体验**
   - 减少重复工作
   - 提高代码质量
   - 加速开发流程

3. **增强团队协作**
   - 便于代码审查
   - 提升代码可维护性
   - 促进知识共享

4. **降低资源消耗**
   - 优化Token使用
   - 提高响应速度
   - 减少计算开销

## 贡献指南

欢迎提交Issue和代码贡献。

## 许可证

MIT License