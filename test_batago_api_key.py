#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试batago.py中TAVILY_API_KEY的加载和使用功能
"""

import os
import sys
import json
import importlib.util
import datetime

print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始测试batago.py的TAVILY_API_KEY加载功能...")

# 测试1：直接导入batago模块并检查API密钥加载
print(f"\n===== 测试1：导入batago模块并检查API密钥加载 =====")

try:
    # 模拟导入batago模块
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 导入batago模块...")
    import batago
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ batago模块导入成功")
    
    # 检查环境变量
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 检查导入后的环境变量TAVILY_API_KEY...")
    env_key = os.environ.get('TAVILY_API_KEY')
    if env_key:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 环境变量中存在TAVILY_API_KEY")
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] API密钥长度: {len(env_key)} 字符")
    else:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 环境变量中不存在TAVILY_API_KEY")
    
    # 直接测试load_saved_api_key函数
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 测试batago.load_saved_api_key()函数...")
    if hasattr(batago, 'load_saved_api_key'):
        saved_key = batago.load_saved_api_key()
        if saved_key:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 成功通过函数加载到API密钥")
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 加载的API密钥长度: {len(saved_key)} 字符")
        else:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 函数未能加载到API密钥")
    else:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ batago模块中没有load_saved_api_key函数")
    
except Exception as e:
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 导入batago模块时出错: {e}")
    import traceback
    traceback.print_exc()

# 测试2：测试保存功能并验证
print(f"\n===== 测试2：测试save_api_key功能 =====")
try:
    if hasattr(batago, 'save_api_key'):
        test_key = "test_batago_api_key_validation_123456"
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 测试保存API密钥...")
        batago.save_api_key(test_key)
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 测试API密钥保存成功")
        
        # 立即重新加载验证
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 重新加载验证保存的API密钥...")
        loaded_key = batago.load_saved_api_key()
        if loaded_key == test_key:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 验证通过：保存和加载的API密钥一致")
        else:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 验证失败：保存和加载的API密钥不一致")
    else:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ batago模块中没有save_api_key函数")
except Exception as e:
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 测试save_api_key功能时出错: {e}")

# 测试3：模拟环境变量未设置的情况，强制从配置文件加载
print(f"\n===== 测试3：模拟环境变量未设置，强制从配置文件加载 =====")
try:
    # 备份当前环境变量
    original_env_key = os.environ.get('TAVILY_API_KEY')
    if original_env_key:
        del os.environ['TAVILY_API_KEY']
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 已临时移除环境变量TAVILY_API_KEY")
    
    # 重新加载模块（尝试，但可能不会完全重新初始化）
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 重新加载batago模块...")
    if 'batago' in sys.modules:
        importlib.reload(sys.modules['batago'])
    
    # 直接测试tavily_search函数的错误处理
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 测试tavily_search函数的错误处理...")
    if hasattr(batago, 'tavily_search'):
        result = batago.tavily_search("测试搜索")
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] tavily_search返回结果: {result[:100]}...")
        if "TAVILY_API_KEY 未配置" in result:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ API密钥未正确加载，仍显示未配置")
        elif "搜索失败" in result:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️  搜索失败，但可能是因为测试密钥无效")
        else:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ API密钥可能已成功加载")
    else:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ batago模块中没有tavily_search函数")
    
    # 恢复环境变量
    if original_env_key:
        os.environ['TAVILY_API_KEY'] = original_env_key
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 已恢复环境变量TAVILY_API_KEY")
    
except Exception as e:
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 测试环境变量未设置情况时出错: {e}")
    # 确保恢复环境变量
    if original_env_key:
        os.environ['TAVILY_API_KEY'] = original_env_key

# 测试4：完整的配置文件验证
print(f"\n===== 测试4：完整的配置文件验证 =====")
try:
    # 获取配置文件路径
    if hasattr(batago, '_get_config_path'):
        config_path = batago._get_config_path()
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 配置文件路径: {config_path}")
        
        if config_path.exists():
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 配置文件存在")
            # 读取并验证配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 配置文件JSON格式正确")
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 配置文件包含的键: {list(data.keys())}")
                if 'TAVILY_API_KEY' in data:
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 配置文件包含TAVILY_API_KEY")
                else:
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 配置文件不包含TAVILY_API_KEY")
        else:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 配置文件不存在")
    else:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ batago模块中没有_get_config_path函数")
except Exception as e:
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 测试配置文件时出错: {e}")

print(f"\n===== 测试完成 =====")
print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 建议: 现在可以运行batago.py主程序，检查API密钥加载是否正常工作")
print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 如果问题仍然存在，请确保在Gradio界面中正确输入并提交API密钥")
