import os
import json
from pathlib import Path

# 获取配置文件路径
def get_config_path() -> Path:
    return Path(os.path.expanduser('~')) / '.batago_config.json'

# 检查配置文件存在性和内容
config_path = get_config_path()
print(f"配置文件路径: {config_path}")
print(f"配置文件是否存在: {config_path.exists()}")

# 如果配置文件存在，尝试读取内容
if config_path.exists():
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        print(f"配置文件内容: {config_data}")
        print(f"TAVILY_API_KEY是否存在: {'TAVILY_API_KEY' in config_data}")
        if 'TAVILY_API_KEY' in config_data:
            print(f"API密钥长度: {len(config_data['TAVILY_API_KEY'])} 字符")
            # 不要打印实际的API密钥，只显示是否存在和长度
    except Exception as e:
        print(f"读取配置文件时出错: {e}")
else:
    print("配置文件不存在，API密钥未被保存")

# 检查环境变量
print(f"环境变量中的TAVILY_API_KEY: {'已设置' if os.environ.get('TAVILY_API_KEY') else '未设置'}")
if os.environ.get('TAVILY_API_KEY'):
    print(f"环境变量中的API密钥长度: {len(os.environ['TAVILY_API_KEY'])} 字符")
