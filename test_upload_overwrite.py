import os
import sys
import shutil
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 模拟上传文件的类
class MockFile:
    def __init__(self, name, content):
        self.name = name
        self._content = content.encode()
        
    def read(self):
        return self._content

# 导入upload_file函数（需要确保UPLOAD_DIR已定义）
from batago import upload_file, UPLOAD_DIR

def test_upload_overwrite():
    print("开始测试文件上传覆盖功能...")
    
    # 确保上传目录存在
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # 测试文件名称
    test_filename = "test_overwrite.txt"
    test_file_path = os.path.join(UPLOAD_DIR, test_filename)
    
    try:
        # 1. 创建第一个测试文件
        first_content = "这是第一次上传的内容"
        first_file = MockFile(test_filename, first_content)
        
        # 上传第一个文件
        result1 = upload_file([first_file])
        print(f"第一次上传结果: {result1}")
        
        # 验证第一个文件内容
        with open(test_file_path, 'r', encoding='utf-8') as f:
            content1 = f.read()
            print(f"第一个文件内容: {content1}")
        
        # 2. 创建第二个测试文件（同名）
        second_content = "这是第二次上传的内容 - 应该覆盖第一个文件"
        second_file = MockFile(test_filename, second_content)
        
        # 上传第二个文件（应该覆盖第一个文件）
        result2 = upload_file([second_file])
        print(f"第二次上传结果: {result2}")
        
        # 验证文件被覆盖
        with open(test_file_path, 'r', encoding='utf-8') as f:
            content2 = f.read()
            print(f"覆盖后的文件内容: {content2}")
        
        # 检查是否只有一个文件存在（不应该有带时间戳的文件）
        files_in_dir = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(test_filename.split('.')[0])]
        print(f"目录中以'{test_filename.split('.')[0]}'开头的文件数量: {len(files_in_dir)}")
        print(f"这些文件是: {files_in_dir}")
        
        # 验证功能是否成功
        if len(files_in_dir) == 1 and content2 == second_content:
            print("\n✅ 测试成功: 同名文件上传成功覆盖原文件")
        else:
            print("\n❌ 测试失败: 同名文件未能正确覆盖")
    
except Exception as e:
        print(f"测试过程中出错: {str(e)}")
    
    finally:
        # 清理测试文件
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            print(f"清理测试文件: {test_file_path}")

if __name__ == "__main__":
    test_upload_overwrite()
