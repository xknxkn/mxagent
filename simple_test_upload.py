import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 模拟NamedString类
class NamedString:
    def __init__(self, name, value):
        self.name = name
        self.value = value
    
    def __str__(self):
        return self.value

# 导入upload_file函数和UPLOAD_DIR
from batago import upload_file, UPLOAD_DIR

# 确保上传目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 创建测试文件
print("创建测试文件...")
test_file = NamedString("测试文件.txt", "这是测试内容")

# 测试上传
print("测试上传...")
result = upload_file(test_file)
print(f"上传结果: {result}")

# 验证文件是否存在
test_path = os.path.join(UPLOAD_DIR, "测试文件.txt")
if os.path.exists(test_path):
    print(f"✅ 文件已上传: {test_path}")
    # 清理
    os.remove(test_path)
    print("已清理测试文件")
else:
    print(f"❌ 文件未上传")

print("测试完成!")
