#!/usr/bin/env python3
# coding: utf-8
"""
测试吴天昊文件清理问题的脚本
"""

import os
import re
import sys
from pathlib import Path

# 确保可以导入batago模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入需要的函数
from batago import normalize_name, clean_old_student_files

# 设置测试参数
STUDENT_NAME = "吴天昊"
CAREER_PLANS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'career_plans')

def test_name_normalization():
    """测试姓名标准化"""
    print("=== 测试姓名标准化 ===")
    normalized = normalize_name(STUDENT_NAME)
    print(f"原始姓名: '{STUDENT_NAME}'")
    print(f"标准化后: '{normalized}'")
    print(f"标准化后(替换空格): '{normalized.replace(' ', '_')}'")
    return normalized

def test_file_pattern():
    """测试文件匹配模式"""
    print("\n=== 测试文件匹配模式 ===")
    normalized_name = normalize_name(STUDENT_NAME).replace(' ', '_')
    pattern = re.compile(r'^%s_(\d{8})_(\d{6})\.(docx|pdf)$' % re.escape(normalized_name))
    print(f"使用的正则表达式: {pattern.pattern}")
    
    # 列出目录中所有吴天昊的文件
    print("\n目录中的吴天昊文件:")
    files = [f for f in os.listdir(CAREER_PLANS_DIR) if f.startswith(STUDENT_NAME)]
    
    for file in files:
        match = pattern.match(file)
        result = "✅ 匹配" if match else "❌ 不匹配"
        print(f"- {file}: {result}")
        if match:
            print(f"  匹配组: {match.groups()}")
    
    return pattern, normalized_name

def debug_clean_old_student_files():
    """调试clean_old_student_files函数"""
    print("\n=== 调试clean_old_student_files函数 ===")
    print(f"调用clean_old_student_files('{STUDENT_NAME}')")
    
    # 导入datetime用于调试
    import datetime
    
    # 定义career_plans目录路径
    CAREER_PLANS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'career_plans')
    
    # 文件命名规则：学生姓名_年月日_时分秒.docx 或 学生姓名_年月日_时分秒.pdf
    normalized_name = normalize_name(STUDENT_NAME).replace(' ', '_')
    file_pattern = re.compile(r'^%s_(\d{8})_(\d{6})\.(docx|pdf)$' % re.escape(normalized_name))
    
    print(f"normalized_name: '{normalized_name}'")
    print(f"file_pattern: {file_pattern.pattern}")
    
    # 获取目录中所有文件
    all_files = os.listdir(CAREER_PLANS_DIR)
    print(f"目录中共有{len(all_files)}个文件")
    
    # 收集该学生的所有文件
    student_files = []
    for filename in all_files:
        match = file_pattern.match(filename)
        print(f"检查文件: '{filename}' -> 匹配: {bool(match)}")
        if match:
            date_str, time_str, file_ext = match.groups()
            # 组合成完整的时间字符串
            datetime_str = f'{date_str}{time_str}'
            try:
                # 转换为datetime对象用于比较
                file_datetime = datetime.datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
                student_files.append({
                    'filename': filename,
                    'datetime': file_datetime,
                    'timestamp': datetime_str,
                    'ext': file_ext
                })
                print(f"  添加到学生文件列表: {filename} - {file_datetime}")
            except ValueError:
                print(f"  日期格式不正确，跳过: {filename}")
    
    print(f"\n找到{len(student_files)}个匹配的学生文件")
    if student_files:
        # 按时间戳分组
        timestamp_groups = {}
        for file_info in student_files:
            if file_info['timestamp'] not in timestamp_groups:
                timestamp_groups[file_info['timestamp']] = []
            timestamp_groups[file_info['timestamp']].append(file_info)
        
        print(f"时间戳分组: {len(timestamp_groups)}组")
        for ts, files in timestamp_groups.items():
            print(f"  时间戳 {ts}: {[f['filename'] for f in files]}")
        
        # 获取所有时间戳并按降序排序
        sorted_timestamps = sorted(timestamp_groups.keys(), reverse=True)
        print(f"排序后的时间戳: {sorted_timestamps}")
    
    # 执行实际的清理函数并显示结果
    print("\n执行实际的清理函数...")
    result = clean_old_student_files(STUDENT_NAME)
    print("\n清理结果:")
    for key, value in result.items():
        print(f"{key}: {value}")

def test_with_actual_filename():
    """使用实际文件名测试匹配"""
    print("\n=== 使用实际文件名测试匹配 ===")
    
    # 假设文件名
    filenames = ["吴天昊_20251213_194336.docx", "吴天昊_20251213_202949.docx"]
    
    # 测试不同的名称标准化方法
    methods = [
        ("原始标准化", lambda name: normalize_name(name).replace(' ', '_')),
        ("直接使用原始名称", lambda name: name)
    ]
    
    for method_name, method_func in methods:
        print(f"\n{method_name}:")
        normalized_name = method_func(STUDENT_NAME)
        pattern = re.compile(r'^%s_(\d{8})_(\d{6})\.(docx|pdf)$' % re.escape(normalized_name))
        print(f"  标准化名称: '{normalized_name}'")
        print(f"  正则表达式: {pattern.pattern}")
        
        for filename in filenames:
            match = pattern.match(filename)
            result = "✅ 匹配" if match else "❌ 不匹配"
            print(f"  {filename}: {result}")

def main():
    """运行所有测试"""
    print("=== 吴天昊文件清理问题分析 ===\n")
    
    # 检查career_plans目录是否存在
    if not os.path.exists(CAREER_PLANS_DIR):
        print(f"错误: 目录 {CAREER_PLANS_DIR} 不存在")
        return
    
    # 测试1: 姓名标准化
    test_name_normalization()
    
    # 测试2: 文件匹配模式
    test_file_pattern()
    
    # 测试3: 使用实际文件名测试不同的匹配方法
    test_with_actual_filename()
    
    # 测试4: 调试clean_old_student_files函数
    debug_clean_old_student_files()
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()
