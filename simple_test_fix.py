#!/usr/bin/env python3
# coding: utf-8
"""
简单测试修复后的清理功能
"""

import os
import sys

# 确保可以导入batago模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入需要的函数
from batago import clean_old_student_files

# 设置测试参数
STUDENT_NAME = "吴天昊"
CAREER_PLANS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'career_plans')

def list_student_files():
    """列出学生的所有文件"""
    print(f"\n当前{STUDENT_NAME}的文件:")
    files = [f for f in os.listdir(CAREER_PLANS_DIR) if f.startswith(STUDENT_NAME)]
    for file in files:
        file_path = os.path.join(CAREER_PLANS_DIR, file)
        # 获取文件大小和修改时间
        stat_info = os.stat(file_path)
        size = stat_info.st_size
        mtime = os.path.getmtime(file_path)
        import datetime
        mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"- {file} (大小: {size} 字节, 修改时间: {mtime_str})")
    return files

def main():
    """运行测试"""
    print(f"=== 测试{STUDENT_NAME}文件清理修复 ===")
    
    # 检查career_plans目录是否存在
    if not os.path.exists(CAREER_PLANS_DIR):
        print(f"错误: 目录 {CAREER_PLANS_DIR} 不存在")
        return
    
    # 列出修复前的文件
    files_before = list_student_files()
    
    # 执行清理函数
    print("\n执行清理函数...")
    result = clean_old_student_files(STUDENT_NAME)
    
    # 显示清理结果
    print("\n清理结果:")
    print(f"学生姓名: {result['student_name']}")
    print(f"总文件数: {result['total_files']}")
    print(f"删除文件数: {result['deleted_count']}")
    print(f"删除的文件: {result['deleted_files']}")
    print(f"保留的文件: {result['kept_files']}")
    
    # 列出修复后的文件
    files_after = list_student_files()
    
    # 分析结果
    print("\n分析结果:")
    if len(files_after) < len(files_before):
        print(f"✅ 修复成功! 清理了{len(files_before) - len(files_after)}个旧文件")
    else:
        print(f"❌ 修复失败! 没有删除任何文件")
        if len(files_before) <= 1:
            print("  可能原因: 只有一个文件，不需要清理")
        else:
            print("  可能原因: 匹配规则仍然有问题")

if __name__ == "__main__":
    main()
