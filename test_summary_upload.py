#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试generate_summary函数的SCP上传功能
"""

import os
import sys
import time

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from batago import generate_summary

def test_summary_upload():
    """
    测试generate_summary函数生成摘要并上传到远程服务器
    """
    try:
        print("===== 开始测试generate_summary函数的SCP上传功能 =====")
        
        # 使用已知的学生姓名和时间范围进行测试
        # 注意：这里使用了"最近"作为时间范围，可以根据实际情况调整
        student_name = "吴天昊"
        time_range = "最近"
        
        print(f"测试学生: {student_name}")
        print(f"测试时间范围: {time_range}")
        print("开始生成摘要...")
        
        # 调用generate_summary函数
        result = generate_summary(student_name, time_range)
        
        print("\n===== 摘要生成结果 =====")
        print(result)
        
        # 检查结果中是否包含文件上传相关信息
        if "文件已成功通过SCP上传到远程服务器" in result:
            print("\n✅ 测试成功: 文件已成功通过SCP上传到远程服务器")
        elif "SCP上传失败" in result:
            print("\n❌ 测试失败: SCP上传失败")
        else:
            print("\n⚠️ 测试警告: 结果中未包含明确的SCP上传状态信息")
        
        # 检查是否包含分享链接
        if "文件分享链接" in result:
            print("✅ 测试成功: 生成了文件分享链接")
        else:
            print("❌ 测试失败: 未生成文件分享链接")
            
        print("\n===== 测试完成 =====")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_summary_upload()
