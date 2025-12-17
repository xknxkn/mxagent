#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修改后的career_planning函数SCP功能
"""

import os
import sys
import datetime

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_career_planning_scp():
    """测试career_planning函数的SCP功能"""
    try:
        # 这里我们只测试upload_file_via_scp函数和SCP命令生成逻辑
        from batago import upload_file_via_scp

        # 创建测试文件
        test_content = f"""# 职业规划测试文档

生成时间：{datetime.datetime.now()}

这是用于测试SCP上传功能的文档。

## 主要内容

1. 职业目标
2. 发展路径
3. 技能提升

---
测试完成
"""

        test_file = "test_career_plan.md"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)

        print(f"✓ 测试Markdown文件已创建: {test_file}")

        # 模拟career_planning中的文件名生成
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        remote_filename = f"career_plan_{timestamp}.docx"
        remote_path = f"/opt/redmine-3.0.1-0/apache2/htdocs/sharefile/{remote_filename}"

        # 测试上传
        if upload_file_via_scp(test_file, remote_path):
            # 生成SCP命令（模拟career_planning中的逻辑）
            scp_command = f"scp batago@121.40.182.30:{remote_path} ."
            print("✓ 文件上传成功！")
            print(f"✓ 生成的SCP下载命令: {scp_command}")
            print("\n=== career_planning函数将返回的信息 ===")
            print(f"SCP下载命令: {scp_command}")
            print("本地备份: test_career_plan.md")
            print("💡 提示: 复制上方SCP命令到终端执行即可下载Word文档。")
        else:
            print("❌ 文件上传失败")

    except Exception as e:
        print(f"❌ 测试过程中出错: {str(e)}")
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"✓ 测试文件已清理: {test_file}")

if __name__ == "__main__":
    test_career_planning_scp()