#!/usr/bin/env python3
# coding: utf-8

"""
简单测试clean_old_student_files函数的PDF清理功能
"""

import os
import sys
import shutil
from pathlib import Path

# 导入函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from batago import clean_old_student_files, normalize_name

def test_pdf_clean():
    """简单测试PDF清理功能"""
    print("开始测试clean_old_student_files函数的PDF清理功能...")
    
    # 获取career_plans目录
    career_dir = os.path.join(os.path.dirname(__file__), 'career_plans')
    
    # 测试学生姓名
    test_name = "测试学生"
    normalized = normalize_name(test_name).replace(' ', '_')
    
    # 创建测试文件（如果目录存在）
    if os.path.exists(career_dir):
        print(f"在目录 {career_dir} 中创建测试文件...")
        
        # 创建两组测试文件
        old_files = [
            f"{normalized}_20251210_010000.docx",
            f"{normalized}_20251210_010000.pdf"
        ]
        new_files = [
            f"{normalized}_20251210_020000.docx",
            f"{normalized}_20251210_020000.pdf"
        ]
        
        # 创建文件
        for file_list in [old_files, new_files]:
            for filename in file_list:
                file_path = os.path.join(career_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Test content for {filename}")
                print(f"创建测试文件: {filename}")
        
        # 执行清理
        print("\n执行清理函数...")
        result = clean_old_student_files(test_name)
        print(f"清理结果: {result}")
        
        # 检查结果
        remaining_files = [f for f in os.listdir(career_dir) if normalized in f]
        print(f"\n剩余的测试文件: {remaining_files}")
        
        # 验证是否只保留了最新的文件
        expected_remaining = set(new_files)
        actual_remaining = set(remaining_files)
        
        if expected_remaining == actual_remaining:
            print("✅ 测试通过! 只保留了最新的Word和PDF文件")
        else:
            print("❌ 测试失败! 清理结果不符合预期")
            print(f"预期保留: {expected_remaining}")
            print(f"实际保留: {actual_remaining}")
        
        # 清理测试文件
        print("\n清理测试文件...")
        for file_list in [old_files, new_files]:
            for filename in file_list:
                file_path = os.path.join(career_dir, filename)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"删除测试文件: {filename}")
                    except Exception as e:
                        print(f"删除测试文件 {filename} 失败: {e}")
    else:
        print(f"错误: career_plans目录不存在: {career_dir}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_pdf_clean()
