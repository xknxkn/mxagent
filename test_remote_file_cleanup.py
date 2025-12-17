#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试远程文件清理功能
验证clean_remote_student_files函数是否能正确清理远程服务器上的旧文件
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入batago模块中的clean_remote_student_files函数
from batago import clean_remote_student_files

def test_remote_cleanup():
    """测试远程文件清理功能"""
    print("=" * 60)
    print("远程文件清理功能测试")
    print("=" * 60)
    
    # 测试参数
    test_student_name = "测试学生"
    test_file_types = ['career', 'summary']
    
    for file_type in test_file_types:
        print(f"\n测试清理类型: {file_type}")
        print(f"正在清理学生 '{test_student_name}' 的远程{file_type}类型文件...")
        
        try:
            # 调用远程文件清理函数
            result = clean_remote_student_files(test_student_name, file_type)
            
            # 打印结果
            print("\n清理结果:")
            print(f"  学生姓名: {result['student_name']}")
            print(f"  总文件数: {result['total_files']}")
            print(f"  删除文件数: {result['deleted_count']}")
            print(f"  保留文件: {result['kept_file']}")
            print(f"  状态: {result['status']}")
            print(f"  消息: {result['message']}")
            
            if result['deleted_files']:
                print("  删除的文件列表:")
                for deleted_file in result['deleted_files']:
                    print(f"    - {deleted_file}")
            else:
                print("  未删除任何文件")
                
            # 如果有错误，标记测试失败
            if result['status'] != 'success':
                print("❌ 测试失败")
            else:
                print("✅ 测试成功")
                
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {str(e)}")
    
    print("\n" + "=" * 60)
    print("远程文件清理功能测试完成")
    print("=" * 60)

if __name__ == "__main__":
    # 设置控制台编码为UTF-8
    if sys.platform.startswith('win'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    test_remote_cleanup()
