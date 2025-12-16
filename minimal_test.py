#!/usr/bin/env python3
# coding: utf-8
"""
最小化测试脚本，直接检查文件名匹配问题
"""

import os
import re

# 设置测试参数
STUDENT_NAME = "吴天昊"
CAREER_PLANS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'career_plans')

def test_regex_matching():
    """测试不同的正则表达式匹配模式"""
    print("=== 测试正则表达式匹配 ===")
    
    # 列出目录中的所有文件
    try:
        all_files = os.listdir(CAREER_PLANS_DIR)
        print(f"目录中共有{len(all_files)}个文件")
        
        # 找出可能的吴天昊文件
        wth_files = [f for f in all_files if '吴天昊' in f]
        print(f"\n包含'吴天昊'的文件: {len(wth_files)}个")
        for file in wth_files:
            print(f"- {file}")
        
        # 测试不同的正则表达式
        patterns = [
            ("使用原始姓名", re.compile(r'^%s_(\d{8})_(\d{6})\.(docx|pdf)$' % re.escape(STUDENT_NAME))),
            ("宽松匹配", re.compile(r'.*%s.*(\d{8})_(\d{6})\.(docx|pdf)$' % re.escape(STUDENT_NAME))),
            ("只检查前缀", re.compile(r'^%s_.*$' % re.escape(STUDENT_NAME)))
        ]
        
        print("\n测试不同的匹配模式:")
        for pattern_name, pattern in patterns:
            print(f"\n{pattern_name}:")
            print(f"正则表达式: {pattern.pattern}")
            matched_files = []
            for file in all_files:
                match = pattern.match(file)
                if match:
                    matched_files.append(file)
                    print(f"✅ 匹配: {file}")
            if not matched_files:
                print("❌ 没有匹配的文件")
    
    except Exception as e:
        print(f"发生错误: {e}")

def main():
    """主函数"""
    print("=== 最小化测试 - 吴天昊文件名匹配 ===")
    
    # 检查目录是否存在
    if os.path.exists(CAREER_PLANS_DIR):
        test_regex_matching()
    else:
        print(f"错误: 目录 {CAREER_PLANS_DIR} 不存在")

if __name__ == "__main__":
    main()
