"""
服务端包
提供MCP代码检索服务的服务端实现
"""

# 不要在包初始化时直接导入模块，避免循环导入问题
# 改为在需要时导入

__all__ = [
    # 这些符号将在导入时动态加载
    "main",
    "setup_routes"
]

# 定义懒加载函数
def main(*args, **kwargs):
    from .app import main as app_main
    return app_main(*args, **kwargs)

def setup_routes(*args, **kwargs):
    from .api import setup_routes as api_setup_routes
    return api_setup_routes(*args, **kwargs)