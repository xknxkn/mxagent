#!/usr/bin/env python3
# coding: utf-8

"""
测试clean_old_student_files函数在只有Word文件时的行为
"""

import os
import sys
import shutil
from datetime import datetime, timedelta

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入需要测试的函数
from batago import clean_old_student_files

# 测试目录路径
TEST_CAREER_PLANS_DIR = os.path.join(os.path.dirname(__file__), 'career_plans_test')

# 备份原始目录
ORIGINAL_CAREER_PLANS_DIR = os.path.join(os.path.dirname(__file__), 'career_plans')


def setup_test_environment():
    """设置测试环境，创建测试目录和文件"""
    print("设置测试环境...")
    
    # 确保测试目录存在
    if os.path.exists(TEST_CAREER_PLANS_DIR):
        shutil.rmtree(TEST_CAREER_PLANS_DIR)
    os.makedirs(TEST_CAREER_PLANS_DIR)
    
    # 备份原始目录
    if os.path.exists(ORIGINAL_CAREER_PLANS_DIR):
        shutil.copytree(ORIGINAL_CAREER_PLANS_DIR, TEST_CAREER_PLANS_DIR + '_original_backup')


def create_test_files(student_name):
    """创建测试文件"""
    print(f"创建测试文件 for {student_name}...")
    
    # 获取当前时间
    now = datetime.now()
    
    # 创建3个不同时间戳的Word文件（没有PDF文件）
    # 最旧的文件：2天前
    old_date = (now - timedelta(days=2)).strftime('%Y%m%d_%H%M%S')
    old_file = f"{student_name}_{old_date}.docx"
    with open(os.path.join(TEST_CAREER_PLANS_DIR, old_file), 'w', encoding='utf-8') as f:
        f.write(f"这是一个旧的测试文件: {old_file}")
    print(f"创建文件: {old_file}")
    
    # 中间的文件：1天前
    mid_date = (now - timedelta(days=1)).strftime('%Y%m%d_%H%M%S')
    mid_file = f"{student_name}_{mid_date}.docx"
    with open(os.path.join(TEST_CAREER_PLANS_DIR, mid_file), 'w', encoding='utf-8') as f:
        f.write(f"这是一个中间的测试文件: {mid_file}")
    print(f"创建文件: {mid_file}")
    
    # 最新的文件：现在
    newest_date = now.strftime('%Y%m%d_%H%M%S')
    newest_file = f"{student_name}_{newest_date}.docx"
    with open(os.path.join(TEST_CAREER_PLANS_DIR, newest_file), 'w', encoding='utf-8') as f:
        f.write(f"这是一个最新的测试文件: {newest_file}")
    print(f"创建文件: {newest_file}")
    
    return old_file, mid_file, newest_file


def run_test(student_name):
    """运行测试"""
    print(f"\n开始测试 clean_old_student_files 函数 for {student_name}...")
    
    # 保存原始的career_plans_dir路径
    import batago
    original_dir = batago.clean_old_student_files.__globals__.get('CAREER_PLANS_DIR')
    
    try:
        # 修改函数内的CAREER_PLANS_DIR全局变量指向测试目录
        batago.clean_old_student_files.__globals__['CAREER_PLANS_DIR'] = TEST_CAREER_PLANS_DIR
        
        # 运行清理函数
        result = clean_old_student_files(student_name)
        print(f"\n清理结果: {result}")
        
        # 检查剩余的文件
        remaining_files = os.listdir(TEST_CAREER_PLANS_DIR)
        print(f"\n清理后剩余的文件: {remaining_files}")
        
        # 验证结果
        if result['deleted_count'] == 2 and result['kept_files'].get('docx') and len(remaining_files) == 1:
            print("✅ 测试通过: 成功删除了2个旧文件，只保留了最新的Word文件")
            return True
        else:
            print("❌ 测试失败: 未按预期删除文件或保留文件")
            print(f"  - 删除的文件数: {result['deleted_count']}")
            print(f"  - 保留的文件: {result['kept_files']}")
            print(f"  - 剩余文件数: {len(remaining_files)}")
            return False
            
    finally:
        # 恢复原始的CAREER_PLANS_DIR
        if original_dir:
            batago.clean_old_student_files.__globals__['CAREER_PLANS_DIR'] = original_dir


def cleanup_test_environment():
    """清理测试环境"""
    print("\n清理测试环境...")
    if os.path.exists(TEST_CAREER_PLANS_DIR):
        shutil.rmtree(TEST_CAREER_PLANS_DIR)
    # 可以选择恢复原始目录，但我们先不做这个操作
    # if os.path.exists(TEST_CAREER_PLANS_DIR + '_original_backup'):
    #     if os.path.exists(ORIGINAL_CAREER_PLANS_DIR):
    #         shutil.rmtree(ORIGINAL_CAREER_PLANS_DIR)
    #     shutil.move(TEST_CAREER_PLANS_DIR + '_original_backup', ORIGINAL_CAREER_PLANS_DIR)


def main():
    """主函数"""
    student_name = "测试学生"
    
    try:
        setup_test_environment()
        create_test_files(student_name)
        success = run_test(student_name)
        
        if success:
            print("\n🎉 所有测试通过！")
        else:
            print("\n❌ 测试失败！")
            
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
    finally:
        cleanup_test_environment()


if __name__ == "__main__":
    main()
