import os
import sys
import shutil

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 模拟NamedString类（根据错误信息推测）
class NamedString:
    def __init__(self, name, value):
        self.name = name
        self.value = value
    
    def __str__(self):
        return self.value

# 导入upload_file函数和UPLOAD_DIR
from batago import upload_file, UPLOAD_DIR

def test_named_string_upload():
    print("开始测试NamedString对象上传...")
    
    # 确保上传目录存在
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # 测试文件名
    test_filename = "上课反馈20220101to202511123.xlsx"
    test_file_path = os.path.join(UPLOAD_DIR, test_filename)
    
    try:
        # 创建一个模拟的NamedString对象
        test_content = "这是测试文件内容"
        named_string_file = NamedString(test_filename, test_content)
        
        # 测试上传单个文件
        print("\n测试1: 上传单个NamedString对象")
        result1 = upload_file(named_string_file)
        print(f"上传结果: {result1}")
        
        # 验证文件是否被创建
        if os.path.exists(test_file_path):
            print(f"✅ 文件已成功创建: {test_file_path}")
            # 读取文件内容进行验证
            with open(test_file_path, 'rb') as f:
                content = f.read().decode('utf-8')
                print(f"文件内容预览: {content[:50]}..." if len(content) > 50 else f"文件内容: {content}")
        else:
            print(f"❌ 文件未创建: {test_file_path}")
        
        # 测试上传文件列表
        print("\n测试2: 上传NamedString对象列表")
        named_string_file2 = NamedString("另一个测试文件.txt", "这是第二个测试文件的内容")
        result2 = upload_file([named_string_file, named_string_file2])
        print(f"上传结果: {result2}")
        
        # 测试文件是否被覆盖
        print("\n测试3: 覆盖已存在的文件")
        new_content = "这是新的文件内容 - 应该覆盖原文件"
        named_string_file_updated = NamedString(test_filename, new_content)
        result3 = upload_file(named_string_file_updated)
        print(f"上传结果: {result3}")
        
        # 验证文件是否被覆盖
        if os.path.exists(test_file_path):
            with open(test_file_path, 'rb') as f:
                updated_content = f.read().decode('utf-8')
                if updated_content == new_content:
                    print("✅ 文件成功被覆盖，内容正确")
                else:
                    print("❌ 文件内容未正确更新")
        
        print("\n测试完成！")
    
except Exception as e:
        print(f"测试过程中出错: {str(e)}")
    
    finally:
        # 清理测试文件
        files_to_clean = [
            test_file_path,
            os.path.join(UPLOAD_DIR, "另一个测试文件.txt")
        ]
        for file_path in files_to_clean:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"清理测试文件: {file_path}")

if __name__ == "__main__":
    test_named_string_upload()
