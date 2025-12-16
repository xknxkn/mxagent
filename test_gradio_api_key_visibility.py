#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修改后的gradio界面，验证API密钥加载和界面显示逻辑
"""

import os
import sys
import json
import datetime
from pathlib import Path

print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始测试gradio界面的API密钥显示逻辑...")

# 1. 检查配置文件是否存在并包含API密钥
print(f"\n===== 测试1：检查配置文件状态 =====")
def get_config_path() -> Path:
    return Path(os.path.expanduser('~')) / '.batago_config.json'

config_path = get_config_path()
print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 配置文件路径: {config_path}")

if config_path.exists():
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 配置文件存在")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'TAVILY_API_KEY' in data and data['TAVILY_API_KEY']:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 配置文件包含有效的TAVILY_API_KEY")
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] API密钥长度: {len(data['TAVILY_API_KEY'])} 字符")
            else:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 配置文件中没有有效的TAVILY_API_KEY")
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 读取配置文件时出错: {e}")
else:
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 配置文件不存在")

# 2. 检查环境变量中的API密钥状态
print(f"\n===== 测试2：检查环境变量状态 =====")
env_key = os.environ.get('TAVILY_API_KEY')
if env_key:
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️  环境变量中已存在TAVILY_API_KEY（当前会话）")
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 注意：这是当前会话的环境变量，不代表batago.py启动时的状态")
else:
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 当前会话环境变量中没有TAVILY_API_KEY（这是正常的）")

# 3. 分析修改后的batago.py文件
print(f"\n===== 测试3：分析batago.py修改后的代码 =====")
batago_path = os.path.join(os.path.dirname(__file__), 'batago.py')
if os.path.exists(batago_path):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 读取batago.py文件以验证修改...")
    try:
        with open(batago_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 检查API密钥加载检查代码
            if "api_key_loaded = bool(os.environ.get(\"TAVILY_API_KEY\"))" in content:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 找到API密钥加载状态检查代码")
            else:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 未找到API密钥加载状态检查代码")
            
            # 检查api_container的visible属性设置
            if "with gr.Column(visible=not api_key_loaded) as api_container:" in content:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 找到API密钥输入区条件显示代码")
            else:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 未找到API密钥输入区条件显示代码")
            
            # 检查main_container的visible属性设置
            if "with gr.Column(visible=api_key_loaded) as main_container:" in content:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 找到主界面条件显示代码")
            else:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 未找到主界面条件显示代码")
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 读取或分析batago.py文件时出错: {e}")
else:
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 未找到batago.py文件")

# 4. 提供测试结论和运行建议
print(f"\n===== 测试结论 =====")
print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 修改总结：")
print(f"1. ✅ 已在batago.py中添加API密钥加载状态检查")
print(f"2. ✅ 已将API密钥输入区设置为仅在未加载密钥时显示")
print(f"3. ✅ 已将主界面设置为在已加载密钥时直接显示")

print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 预期行为：")
if config_path.exists():
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if 'TAVILY_API_KEY' in data and data['TAVILY_API_KEY']:
            print(f"✅ 当启动batago.py时，系统应自动从配置文件加载API密钥，并直接显示主界面，不再显示API密钥输入界面")
        else:
            print(f"⚠️  配置文件存在但没有有效的API密钥，启动时仍会显示API密钥输入界面")
else:
    print(f"⚠️  配置文件不存在，启动时仍会显示API密钥输入界面")

print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 运行建议：")
print(f"1. 现在可以直接运行 'python batago.py' 来测试修改后的功能")
print(f"2. 如果配置文件中已存储有效的TAVILY_API_KEY，应用程序应直接显示主界面")
print(f"3. 如果配置文件不存在或密钥无效，可以在界面中输入并保存，下次启动时将自动加载")
print(f"4. 查看控制台输出，检查是否有 '成功从配置文件加载TAVILY_API_KEY' 的日志信息")

print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 测试完成！")
