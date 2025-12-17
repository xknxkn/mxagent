#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修改后的career_planning函数是否正确将文件上传到career_plans目录并生成正确的HTTP链接
"""

import os
import sys
import re

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_career_planning_upload():
    """测试career_planning函数是否正确上传文件到career_plans目录并生成正确链接"""
    try:
        from batago import career_planning
        
        print("=== 测试career_planning函数上传到career_plans目录功能 ===")
        
        # 测试使用吴天昊学生，设置一个简单的职业目标
        student_name = "吴天昊"
        career_target = "我想成为一名软件工程师"
        
        print(f"正在为学生 {student_name} 生成职业规划，目标: {career_target}")
        print("注意：这个过程可能需要一些时间，请耐心等待...")
        
        # 调用career_planning函数
        result = career_planning(student_name, career_target)
        
        # 打印结果以便查看
        print("\n=== 函数返回结果 ===")
        print(result)
        
        # 检查结果中是否包含正确的career_plans路径的HTTP链接
        http_link_pattern = r'http://121\.40\.182\.30:8000/sharefile/career_plans/.*\.docx'
        http_links = re.findall(http_link_pattern, result)
        
        if http_links:
            print("\n✅ 成功：找到包含career_plans路径的HTTP分享链接：")
            for link in http_links:
                print(f"   - {link}")
        else:
            print("\n❌ 失败：未找到包含career_plans路径的HTTP分享链接")
            
        # 检查是否不再包含SCP下载命令
        if "SCP下载命令" in result:
            print("\n❌ 失败：结果中仍然包含SCP下载命令")
        else:
            print("\n✅ 成功：结果中已移除SCP下载命令")
        
        print("\n=== 测试完成 ===")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_career_planning_upload()
