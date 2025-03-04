"""
StdioServerTransport模块

本模块为StdioServerTransport类提供兼容层，
该类在MCP库的旧版本中使用，但在新版本中不可用。
"""

import asyncio
from mcp.server.stdio import stdio_server

class StdioServerTransport:
    """
    一个兼容性类，模拟MCP库旧版本中StdioServerTransport类的行为，
    但使用MCP库新版本中的stdio_server函数。
    """
    
    def __init__(self):
        """
        初始化传输器。
        """
        self.read_stream = None
        self.write_stream = None
        self.server = None
        self.task = None
    
    async def _run_server(self):
        """
        使用stdio传输运行服务器。
        """
        async with stdio_server() as (read_stream, write_stream):
            self.read_stream = read_stream
            self.write_stream = write_stream
            
            # Run the server with the streams
            # 使用正确的初始化选项
            from mcp.server.models import InitializationOptions
            from mcp.server.lowlevel import NotificationOptions
            
            init_options = InitializationOptions(
                server_name="mcp-code-indexer",
                server_version="0.1.0",
                capabilities=self.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
            
            await self.server.run(read_stream, write_stream, init_options)
    
    def connect(self, server):
        """
        将服务器连接到此传输器。
        
        Args:
            server: 要连接的MCP服务器。
        """
        self.server = server
        
        # Create a task to run the server
        self.task = asyncio.create_task(self._run_server())
    
    def close(self):
        """
        关闭传输器。
        """
        if self.task:
            self.task.cancel()
            self.task = None