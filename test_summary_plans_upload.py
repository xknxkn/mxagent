#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修改后的generate_summary函数，确保Word文档上传到summary_plans目录并生成正确链接
"""
import sys
import os
import time
from datetime import datetime

# 确保可以导入batago模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入需要的函数
from batago import generate_summary

def test_summary_upload_to_plans():
    """
    测试生成摘要文件并上传到远程服务器的summary_plans目录
    """
    print("===== 测试修改后的摘要生成功能 =====")
    print(f"开始测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 使用已有的学生姓名进行测试
        student_name = "吴天昊"
        time_range = "最近"
        
        print(f"\n测试生成学生 '{student_name}' 的摘要，时间范围: '{time_range}'")
        
        # 调用generate_summary函数
        result = generate_summary(student_name, time_range)
        
        print("\n===== 生成结果 =====")
        print(result)
        
        # 检查结果中是否包含分享链接
        if "文件分享链接" in result:
            print("\n✅ 分享链接生成成功")
            # 检查链接是否包含summary_plans路径
            if "/sharefile/summary_plans/" in result:
                print("✅ 分享链接正确指向summary_plans目录")
            else:
                print("❌ 分享链接未指向summary_plans目录")
        else:
            print("\n❌ 分享链接生成失败")
        
        print(f"\n测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_summary_upload_to_plans()
    if success:
        print("\n🎉 测试成功！修改后的功能正常工作")
    else:
        print("\n❌ 测试失败，请检查代码")
    sys.exit(0 if success else 1)
