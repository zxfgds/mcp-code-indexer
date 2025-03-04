# MCP Code Indexer

An intelligent code retrieval tool based on the Model Context Protocol (MCP) that provides efficient and accurate code repository retrieval capabilities for AI large language models.

[中文版](README.md)

## Project Overview

MCP Code Indexer is a code retrieval tool specifically designed for AI large language models. Through vector indexing and semantic understanding, it helps AI better understand and analyze code repositories, significantly improving the efficiency and accuracy of code-related tasks.

Project Repository: https://github.com/zxfgds/mcp-code-indexer

### Key Features

- **Intelligent Code Retrieval**
  - Semantic-based code search, understanding code meaning beyond keyword matching
  - Cross-language code retrieval support
  - Smart code snippet extraction with automatic context recognition

- **Code Analysis Capabilities**
  - Code structure analysis (classes, functions, dependencies)
  - Code quality assessment
  - Documentation and comment extraction
  - Similar code detection
  - Code metrics statistics
  - Project dependency analysis

- **Optimized Context Processing**
  - Intelligent Token consumption control
  - Structured code context provision
  - Multi-project parallel indexing and retrieval
  - Persistent project identification

- **Complete MCP Protocol Support**
  - Compliant with MCP data interaction specifications
  - Rich toolset provision
  - Asynchronous operation and progress feedback

## Use Cases

### Claude Desktop Application

- **Enhanced Code Understanding**: Helps Claude better understand user codebases, providing more accurate suggestions and answers
- **Intelligent Code Navigation**: Quickly locate relevant code snippets, improving Q&A efficiency
- **Code Quality Improvement**: Provide professional code optimization suggestions through code analysis
- **Project Dependency Management**: Help understand and manage project dependencies
- **Documentation Generation**: Assist in generating technical documentation based on code comments and structure analysis

### VSCode Extension

- **Real-time Code Analysis**: Get code analysis results directly in the editor
- **Intelligent Code Recommendations**: Provide more accurate code suggestions based on project context
- **Refactoring Assistance**: Identify optimizable code patterns
- **Dependency Visualization**: Intuitively display code dependencies
- **Team Collaboration Enhancement**: Help team members better understand the codebase

## Installation

1. Install Dependencies
```bash
pip install -r requirements.txt
```

2. Install Tool
```bash
python setup.py install
```

## Configuration

### Claude Desktop Configuration

Edit configuration file: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "code-indexer": {
      "command": "python",
      "args": ["-m", "server.app"],
      "cwd": "installation_path",
      "env": {},
      "disabled": false,
      "alwaysAllow": []
    }
  }
}
```

### VSCode Extension Configuration

Edit configuration file: `%APPDATA%\Code\User\globalStorage\rooveterinaryinc.roo-cline\settings\cline_mcp_settings.json`

```json
{
  "mcpServers": {
    "code-indexer": {
      "command": "python",
      "args": ["-m", "server.app"],
      "cwd": "installation_path",
      "env": {},
      "disabled": false,
      "alwaysAllow": []
    }
  }
}
```

## Usage Guide

### Basic Functions

1. Project Identification
```
Use identify_project tool to identify the project
```

2. Index Project
```
Use index_project tool to index project code
```

3. Search Code
```
Use search_code tool to search for relevant code snippets
```

### Advanced Functions

1. Get Code Structure
```
Use get_code_structure tool to analyze code structure
```

2. Analyze Code Quality
```
Use analyze_code_quality tool to assess code quality
```

3. Extract Documentation
```
Use extract_documentation tool to extract code documentation
```

4. Find Similar Code
```
Use find_similar_code tool to detect similar code
```

5. Get Code Metrics
```
Use get_code_metrics tool to get code statistics
```

6. Analyze Dependencies
```
Use analyze_dependencies tool to analyze project dependencies
```

## Project Value

1. **Enhance AI Code Understanding**
   - More accurate code semantic understanding
   - More comprehensive project context grasp
   - Smarter code-related suggestions

2. **Optimize Development Experience**
   - Reduce repetitive work
   - Improve code quality
   - Accelerate development process

3. **Strengthen Team Collaboration**
   - Facilitate code review
   - Enhance code maintainability
   - Promote knowledge sharing

4. **Reduce Resource Consumption**
   - Optimize Token usage
   - Improve response speed
   - Reduce computational overhead

## Contributing

Issues and code contributions are welcome.

## License

MIT License