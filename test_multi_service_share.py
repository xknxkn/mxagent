#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多服务文件分享功能
"""

import os
import sys
import time
from batago import create_share_link

def create_test_file():
    """
    创建一个测试文件用于分享
    """
    test_content = "这是一个测试文件，用于测试文件分享功能。\n" * 5
    test_filename = "test_share_file.txt"
    
    # 创建测试文件
    with open(test_filename, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"创建测试文件: {os.path.abspath(test_filename)}")
    return test_filename

def test_file_sharing():
    """
    测试文件分享功能
    """
    # 创建测试文件
    test_file = create_test_file()
    
    try:
        # 测试文件分享
        print("\n开始测试文件分享功能...")
        start_time = time.time()
        
        # 调用分享函数
        share_link = create_share_link(test_file)
        
        end_time = time.time()
        
        # 打印结果
        print(f"\n分享结果: {share_link}")
        print(f"分享耗时: {end_time - start_time:.2f} 秒")
        
        # 评估结果
        if share_link.startswith("http://"):
            print("\n测试结果: 成功 - 成功创建了分享链接")
        else:
            print(f"\n测试结果: 失败 - {share_link}")
            
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n清理测试文件: {test_file}")

if __name__ == "__main__":
    print("=== 多服务文件分享功能测试 ===")
    print("这个测试将验证本地HTTP服务器分享功能")
    print("当前支持的分享服务：")
    print("1. 本地HTTP服务器（首选）")
    print("2. 蓝奏云（预留接口）")
    print("3. 123云盘（预留接口）")
    print("4. 百度网盘（预留接口）")
    print("")
    
    test_file_sharing()
    
    print("\n=== 测试完成 ===")
