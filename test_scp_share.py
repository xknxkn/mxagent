#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试SCP文件分享功能
"""

import os
import sys
import time
from batago import create_share_link

def create_test_file():
    """
    创建一个测试文件用于分享
    """
    test_content = "这是一个测试文件，用于测试SCP文件分享功能。\n" * 5
    test_filename = "test_scp_share.txt"
    
    # 创建测试文件
    with open(test_filename, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"创建测试文件: {os.path.abspath(test_filename)}")
    return test_filename

def test_scp_sharing():
    """
    测试SCP文件分享功能
    """
    # 创建测试文件
    test_file = create_test_file()
    
    try:
        # 测试文件分享
        print("\n开始测试SCP文件分享功能...")
        start_time = time.time()
        
        # 调用分享函数
        result = create_share_link(test_file)
        
        end_time = time.time()
        
        # 打印结果
        print(f"\n分享结果: {result}")
        print(f"分享耗时: {end_time - start_time:.2f} 秒")
        
        # 评估结果
        if result.startswith("scp"):
            print("\n测试结果: 成功 - 成功创建了SCP下载命令")
            print("\n使用方法:")
            print(f"1. 在终端中复制并执行以下命令:")
            print(f"   {result}")
            print("2. 系统会提示输入密码，输入后文件将下载到当前目录")
        else:
            print(f"\n测试结果: 失败 - {result}")
            print("\n请检查:")
            print("1. 网络连接是否正常")
            print("2. 服务器配置是否正确")
            print("3. SCP依赖库是否安装 (pip install -r scp_requirements.txt)")
            
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n清理测试文件: {test_file}")

if __name__ == "__main__":
    print("=== SCP文件分享功能测试 ===")
    print("这个测试将验证通过SCP协议上传文件到远程服务器的功能")
    print("服务器信息：")
    print("- 地址: 121.40.182.30")
    print("- 用户名: batago")
    print("- 密码: 4008737505")
    print("- 端口: 22")
    print("- 目标目录: /opt/redmine-3.0.1-0/apache2/htdocs/sharefile")
    print("")
    
    test_scp_sharing()
    
    print("\n=== 测试完成 ===")
