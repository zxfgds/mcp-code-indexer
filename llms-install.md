# MCP Code Indexer Installation Guide for AI Agents

This guide provides step-by-step instructions for AI agents like Cline to install and configure the MCP Code Indexer.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/zxfgds/mcp-code-indexer.git
cd mcp-code-indexer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
python setup.py install
```

## Configuration

### For Claude Desktop

1. Edit the configuration file at `%APPDATA%\Claude\claude_desktop_config.json`
2. Add the following configuration under the "mcpServers" section:
```json
{
  "mcpServers": {
    "code-indexer": {
      "command": "python",
      "args": ["-m", "server.app"],
      "cwd": "INSTALLATION_PATH",
      "env": {},
      "disabled": false,
      "alwaysAllow": []
    }
  }
}
```
Note: Replace INSTALLATION_PATH with the actual path where you cloned the repository.

### For VSCode Extension

1. Edit the configuration file at `%APPDATA%\Code\User\globalStorage\rooveterinaryinc.roo-cline\settings\cline_mcp_settings.json`
2. Add the same configuration as above under the "mcpServers" section.

## Verification

To verify the installation:

1. Start the MCP server:
```bash
python -m server.app
```

2. The server should start without errors and be ready to accept MCP protocol requests.

## Common Issues

1. If you encounter encoding issues on Windows, run:
```bash
chcp 65001
python -m server.app
```

2. If the server fails to start, check:
- Python version compatibility
- All dependencies are installed correctly
- Configuration file paths are correct

## Additional Notes

- The server uses the bge-large-zh model for better Chinese code understanding
- Default storage location for indices is in the `vector_db` directory
- Project identification data is stored in the `project_data` directory