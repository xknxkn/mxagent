#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单的文件分享测试脚本
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入batago模块
try:
    from batago import create_share_link
    print("✅ 成功导入create_share_link函数")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试函数
def test_create_share_link():
    """简单测试create_share_link函数"""
    print("\n开始测试create_share_link函数...")
    
    # 测试不存在的文件
    print("\n1. 测试不存在的文件:")
    result = create_share_link("non_existent_file.txt")
    print(f"结果: {result}")
    
    # 确保upload目录存在
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'upload')
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        print(f"\n✅ 创建了upload目录: {upload_dir}")
    
    # 创建一个测试文件
    test_file = os.path.join(upload_dir, "test_share.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("这是测试文件内容")
    
    print(f"\n✅ 创建了测试文件: {test_file}")
    
    # 测试存在的文件
    print("\n2. 测试存在的文件:")
    try:
        result = create_share_link("test_share.txt")
        print(f"结果: {result}")
        if result.startswith("http"):
            print("✅ 成功生成了分享链接!")
        else:
            print("⚠️  未能生成有效的HTTP分享链接")
    except Exception as e:
        print(f"❌ 调用create_share_link时出错: {e}")

if __name__ == "__main__":
    print("文件分享功能简单测试脚本")
    print("="*50)
    test_create_share_link()
    print("\n测试完成!")
