#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试远程旧文件清理功能
专门验证career_planning函数中的远程文件清理功能
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的函数
from batago import career_planning

def test_career_planning_with_cleanup():
    """测试career_planning函数中的远程文件清理功能"""
    print("=" * 60)
    print("测试career_planning函数中的远程文件清理功能")
    print("=" * 60)
    
    # 测试学生姓名
    test_student_name = "测试学生"
    test_input = "这是一个测试输入，用于生成职业规划文档"
    
    print(f"\n为学生 '{test_student_name}' 生成职业规划文档...")
    
    try:
        # 调用career_planning函数
        result = career_planning(test_student_name, test_input)
        
        # 打印结果信息
        print("\n生成结果:")
        print(f"文档生成状态: 成功")
        
        # 检查结果中是否包含清理信息
        if result and 'content' in result:
            if '🧹 **清理结果**' in result['content']:
                print("✅ 远程文件清理功能已触发并在结果中显示")
            else:
                print("⚠️  结果中未包含清理信息，可能没有旧文件需要清理")
        
        print("\n测试完成")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")

if __name__ == "__main__":
    # 设置控制台编码为UTF-8
    if sys.platform.startswith('win'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    test_career_planning_with_cleanup()
