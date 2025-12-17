#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试远程文件清理功能
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的函数
from batago import clean_remote_student_files

def main():
    """主测试函数"""
    print("简单测试远程文件清理功能")
    print("=" * 50)
    
    # 测试清理career类型文件
    print("\n测试清理career类型文件...")
    result = clean_remote_student_files("测试学生", "career")
    
    # 打印基本结果
    print(f"状态: {result['status']}")
    print(f"总文件数: {result['total_files']}")
    print(f"删除文件数: {result['deleted_count']}")
    
    print("\n测试完成")

if __name__ == "__main__":
    main()
