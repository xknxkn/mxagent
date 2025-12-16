#!/usr/bin/env python3
# coding: utf-8

"""
删除career_plans目录中所有以temp开头的临时文件
"""

import os
import sys
from datetime import datetime

# 设置career_plans目录路径
CAREER_PLANS_DIR = os.path.join(os.path.dirname(__file__), 'career_plans')

def clean_temp_files():
    """删除以temp开头的临时文件"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始清理临时文件...")
    print(f"清理目录: {CAREER_PLANS_DIR}")
    
    # 检查目录是否存在
    if not os.path.exists(CAREER_PLANS_DIR):
        print(f"错误: 目录 {CAREER_PLANS_DIR} 不存在")
        return 1
    
    # 统计信息
    deleted_count = 0
    skipped_count = 0
    failed_count = 0
    deleted_files = []
    
    try:
        # 获取目录中的所有文件
        all_files = os.listdir(CAREER_PLANS_DIR)
        print(f"找到 {len(all_files)} 个文件")
        
        # 遍历并删除以temp开头的文件
        for filename in all_files:
            if filename.lower().startswith('temp'):
                file_path = os.path.join(CAREER_PLANS_DIR, filename)
                try:
                    # 检查是否为文件
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted_count += 1
                        deleted_files.append(filename)
                        print(f"✅ 已删除: {filename}")
                    else:
                        skipped_count += 1
                        print(f"⚠️  跳过(非文件): {filename}")
                except Exception as e:
                    failed_count += 1
                    print(f"❌ 删除失败: {filename}, 错误: {str(e)}")
            else:
                skipped_count += 1
        
        # 打印总结
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 清理完成:")
        print(f"  - 已删除: {deleted_count} 个临时文件")
        print(f"  - 跳过: {skipped_count} 个文件")
        print(f"  - 删除失败: {failed_count} 个文件")
        
        if deleted_count > 0:
            print(f"\n已删除的文件列表:")
            for file in deleted_files:
                print(f"  - {file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 清理过程中发生错误: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(clean_temp_files())
