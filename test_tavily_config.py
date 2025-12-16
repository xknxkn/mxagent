#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试TAVILY_API_KEY配置文件的存储和加载功能
"""

import os
import json
from pathlib import Path
import datetime

print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始测试TAVILY_API_KEY配置文件...")

# 模拟batago.py中的配置路径获取函数
def _get_config_path() -> Path:
    return Path(os.path.expanduser('~')) / '.batago_config.json'

# 检查配置文件是否存在
config_path = _get_config_path()
print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 配置文件路径: {config_path}")

if config_path.exists():
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 配置文件存在")
    
    # 检查文件权限
    try:
        if os.name == 'nt':  # Windows系统
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Windows系统，不检查文件权限")
        else:
            import stat
            file_mode = config_path.stat().st_mode
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 文件权限: {oct(file_mode)}")
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 检查文件权限时出错: {e}")
    
    # 读取并验证配置文件内容
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 文件大小: {len(content)} 字节")
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 文件内容长度: {len(content)} 字符")
            
            # 检查内容是否为空
            if not content.strip():
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 配置文件内容为空")
            else:
                # 尝试解析JSON
                try:
                    data = json.loads(content)
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ JSON解析成功")
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 配置文件中的键: {list(data.keys())}")
                    
                    # 检查是否包含TAVILY_API_KEY
                    if 'TAVILY_API_KEY' in data:
                        key_value = data['TAVILY_API_KEY']
                        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 找到TAVILY_API_KEY")
                        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] API密钥长度: {len(key_value)} 字符")
                        # 打印前5个和后5个字符作为安全验证，中间用*替代
                        if len(key_value) > 10:
                            masked_key = key_value[:5] + '*' * (len(key_value) - 10) + key_value[-5:]
                        else:
                            masked_key = '*' * len(key_value)
                        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 部分显示API密钥: {masked_key}")
                    else:
                        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 配置文件中不包含TAVILY_API_KEY")
                except json.JSONDecodeError as e:
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ JSON解析失败: {e}")
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 文件前100个字符: {content[:100]}...")
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 读取配置文件时出错: {e}")
else:
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 配置文件不存在")

# 检查环境变量
print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 检查环境变量TAVILY_API_KEY:")
env_key = os.environ.get('TAVILY_API_KEY')
if env_key:
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 环境变量中存在TAVILY_API_KEY")
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 环境变量API密钥长度: {len(env_key)} 字符")
else:
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 环境变量中不存在TAVILY_API_KEY")

# 模拟保存和加载功能
def test_save_and_load():
    print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 测试保存和加载API密钥功能...")
    
    test_key = "test_tavily_api_key_123456"
    
    # 保存测试
    try:
        cfg = config_path
        cfg.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg, 'w', encoding='utf-8') as f:
            json.dump({'TAVILY_API_KEY': test_key}, f)
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 测试API密钥保存成功")
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 测试API密钥保存失败: {e}")
        return False
    
    # 加载测试
    try:
        with open(cfg, 'r', encoding='utf-8') as f:
            data = json.load(f)
            loaded_key = data.get('TAVILY_API_KEY')
            if loaded_key == test_key:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 测试API密钥加载成功，验证通过")
                return True
            else:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 加载的API密钥与保存的不匹配")
                return False
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 测试API密钥加载失败: {e}")
        return False

# 运行保存和加载测试
save_load_result = test_save_and_load()

print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 测试总结:")
print(f"配置文件存在: {'✅' if config_path.exists() else '❌'}")
if config_path.exists():
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"包含TAVILY_API_KEY: {'✅' if 'TAVILY_API_KEY' in data else '❌'}")
    except:
        print(f"包含TAVILY_API_KEY: ❌ (JSON解析失败)")
print(f"环境变量设置: {'✅' if env_key else '❌'}")
print(f"保存/加载功能测试: {'✅' if save_load_result else '❌'}")

if config_path.exists() and save_load_result:
    print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🎉 配置文件功能正常工作！")
else:
    print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️  配置文件功能存在问题，可能需要修复")
