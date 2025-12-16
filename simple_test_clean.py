#!/usr/bin/env python3
# coding: utf-8

"""
简单测试clean_old_student_files函数在只有Word文件时的行为
"""

import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入需要测试的函数
from batago import clean_old_student_files


def test_word_only_cleanup():
    """测试只有Word文件时的清理功能"""
    print("开始简单测试...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    student_name = "测试学生"
    
    try:
        # 保存原始的CAREER_PLANS_DIR
        import batago
        original_dir = batago.clean_old_student_files.__globals__.get('CAREER_PLANS_DIR')
        
        # 修改为临时目录
        batago.clean_old_student_files.__globals__['CAREER_PLANS_DIR'] = temp_dir
        
        # 创建3个不同时间戳的Word文件
        now = datetime.now()
        
        # 创建最旧的文件
        old_time = now - timedelta(days=2)
        old_filename = f"{student_name}_{old_time.strftime('%Y%m%d_%H%M%S')}.docx"
        with open(os.path.join(temp_dir, old_filename), 'w') as f:
            f.write('old file')
        
        # 创建中间的文件
        mid_time = now - timedelta(days=1)
        mid_filename = f"{student_name}_{mid_time.strftime('%Y%m%d_%H%M%S')}.docx"
        with open(os.path.join(temp_dir, mid_filename), 'w') as f:
            f.write('middle file')
        
        # 创建最新的文件
        newest_filename = f"{student_name}_{now.strftime('%Y%m%d_%H%M%S')}.docx"
        with open(os.path.join(temp_dir, newest_filename), 'w') as f:
            f.write('newest file')
        
        print(f"创建了3个测试文件: {old_filename}, {mid_filename}, {newest_filename}")
        
        # 运行清理函数
        result = clean_old_student_files(student_name)
        print(f"\n清理结果: {result}")
        
        # 检查剩余文件
        remaining_files = os.listdir(temp_dir)
        print(f"剩余文件: {remaining_files}")
        
        # 验证只保留了最新的文件
        if len(remaining_files) == 1 and newest_filename in remaining_files:
            print("✅ 测试通过: 成功删除了旧的Word文件，只保留了最新的文件")
        else:
            print("❌ 测试失败: 未正确删除旧文件")
            
    finally:
        # 恢复原始设置
        if original_dir:
            batago.clean_old_student_files.__globals__['CAREER_PLANS_DIR'] = original_dir
        # 清理临时目录
        shutil.rmtree(temp_dir)
        print("\n测试环境已清理")


if __name__ == "__main__":
    test_word_only_cleanup()
