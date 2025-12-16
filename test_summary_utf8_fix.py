#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：验证generate_summary函数的UTF-8编码修复
"""
import os
import sys
import datetime

def main():
    print("开始测试generate_summary函数的UTF-8编码修复...")
    print(f"当前Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"系统编码: {sys.stdout.encoding}")
    
    try:
        # 导入必要的模块
        from batago import generate_summary
        
        # 使用一个已知存在的学生姓名进行测试
        # 注意：需要根据实际数据调整学生姓名
        student_name = "吴天昊"  # 可以根据实际数据修改
        time_period = "最近"
        
        print(f"\n测试生成学生'{student_name}'的摘要...")
        print(f"时间范围: {time_period}")
        
        # 调用函数并捕获结果
        result = generate_summary(student_name, time_period)
        
        print("\n函数执行成功!")
        print(f"返回结果长度: {len(result)} 字符")
        
        # 检查是否成功生成了Word文档
        if "Word文档已成功生成" in result:
            print("✅ Word文档生成成功")
        elif "生成Word文档时出错" in result:
            print("❌ Word文档生成失败")
            error_start = result.find("生成Word文档时出错:")
            if error_start != -1:
                error_msg = result[error_start:].split("---")[0].strip()
                print(f"错误信息: {error_msg}")
        
        # 显示部分结果预览
        preview_length = min(500, len(result))
        print(f"\n结果预览 (前{preview_length}字符):")
        print("=" * 50)
        print(result[:preview_length])
        print("...")
        print("=" * 50)
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保batago.py模块在正确的路径中")
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
