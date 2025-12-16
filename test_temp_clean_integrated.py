#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试整合到batago.py中的临时文件清理功能
"""

import os
import sys
import time
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入batago模块
import batago

# 测试临时文件清理功能
def test_temp_clean_function():
    """
    测试batago.py中的临时文件清理功能
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始测试临时文件清理功能...")
    
    # 定义career_plans目录路径
    career_plans_dir = os.path.join(os.path.dirname(__file__), 'career_plans')
    
    # 确保目录存在
    if not os.path.exists(career_plans_dir):
        os.makedirs(career_plans_dir)
    
    # 创建测试用的临时文件
    temp_files = []
    for i in range(3):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"temp_{timestamp}_{i}.md"
        temp_file_path = os.path.join(career_plans_dir, temp_filename)
        
        # 写入一些内容到临时文件
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(f"这是测试临时文件 {i}\n创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        temp_files.append(temp_file_path)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 创建测试临时文件: {temp_filename}")
        time.sleep(1)  # 等待1秒，确保文件名唯一
    
    # 调用batago中的clean_temp_files函数
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 调用clean_temp_files函数清理临时文件...")
    result = batago.clean_temp_files()
    
    # 显示清理结果
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 清理结果:")
    print(f"  成功: {result['success']}")
    print(f"  消息: {result['message']}")
    print(f"  删除文件数: {result['deleted_count']}")
    print(f"  检查文件总数: {result['total_files_checked']}")
    print(f"  跳过文件数: {result['skipped_files']}")
    print(f"  失败文件数: {result['failed_files']}")
    
    # 验证临时文件是否被删除
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 验证临时文件是否被删除:")
    all_deleted = True
    for file_path in temp_files:
        if os.path.exists(file_path):
            print(f"  ❌ 文件未被删除: {os.path.basename(file_path)}")
            all_deleted = False
        else:
            print(f"  ✅ 文件已被删除: {os.path.basename(file_path)}")
    
    # 测试总结
    if all_deleted and result['deleted_count'] == len(temp_files):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ 临时文件清理功能测试成功!")
        return True
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ 临时文件清理功能测试失败!")
        return False

if __name__ == "__main__":
    try:
        success = test_temp_clean_function()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
