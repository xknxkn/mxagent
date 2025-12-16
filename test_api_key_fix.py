import os
import json
from pathlib import Path

# 导入gradiostudentsum.py中的函数进行测试
from gradiostudentsum import _get_config_path, load_saved_api_key, set_tavily_api_key

print("=== 测试TAVILY_API_KEY保存和加载功能 ===")

# 1. 检查配置文件路径
config_path = _get_config_path()
print(f"配置文件路径: {config_path}")

# 2. 测试load_saved_api_key函数
saved_key = load_saved_api_key()
print(f"从配置文件加载的API密钥: {'已找到' if saved_key else '未找到'}")
if saved_key:
    print(f"API密钥长度: {len(saved_key)} 字符")

# 3. 测试设置环境变量
if saved_key:
    os.environ["TAVILY_API_KEY"] = saved_key
    print("已将保存的API密钥设置到环境变量")
    print(f"环境变量TAVILY_API_KEY: {'已设置' if os.environ.get('TAVILY_API_KEY') else '未设置'}")

# 4. 验证gradiostudentsum.py中的自动加载逻辑是否正常工作
print("\n=== 模拟程序启动时的自动加载逻辑 ===")
if not os.environ.get("TAVILY_API_KEY"):
    # 为了测试，先清除环境变量
    if 'TAVILY_API_KEY' in os.environ:
        del os.environ['TAVILY_API_KEY']
    
    # 模拟程序启动时的加载
    saved = load_saved_api_key()
    if saved:
        os.environ["TAVILY_API_KEY"] = saved
        print("成功从配置文件加载API密钥到环境变量")
    else:
        print("无法从配置文件加载API密钥")
else:
    print("环境变量中已有TAVILY_API_KEY")

print("\n=== 测试完成 ===")
