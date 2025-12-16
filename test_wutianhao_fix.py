import os
import sys

# 确保能导入当前目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from batago import clean_old_student_files

print("测试吴天昊文件清理功能...")
print("=" * 50)

# 直接调用清理函数
try:
    result = clean_old_student_files("吴天昊")
    print("\n测试完成！")
except Exception as e:
    print(f"\n测试出错: {e}")
    import traceback
    traceback.print_exc()

print("\n验证目录中的文件...")
career_plans_dir = os.path.join(os.path.dirname(__file__), 'career_plans')
if os.path.exists(career_plans_dir):
    wutianhao_files = [f for f in os.listdir(career_plans_dir) if f.startswith("吴天昊_")]
    print(f"当前目录中吴天昊的文件: {wutianhao_files}")
    print(f"文件数量: {len(wutianhao_files)}")
else:
    print(f"目录不存在: {career_plans_dir}")
