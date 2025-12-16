import os
from batago import upload_file

# 创建一个测试文件
with open('test_upload.txt', 'w') as f:
    f.write('This is a test file for upload functionality.')

# 测试上传功能
try:
    result = upload_file('test_upload.txt')
    print(f"上传结果: {result}")
    
    # 检查文件是否成功上传
    if os.path.exists(os.path.join('upload', 'test_upload.txt')):
        print("测试成功: 文件已成功上传到upload目录")
    else:
        print("测试失败: 文件未在upload目录中找到")
        
except Exception as e:
    print(f"测试失败: {str(e)}")
finally:
    # 清理测试文件
    if os.path.exists('test_upload.txt'):
        os.remove('test_upload.txt')
    if os.path.exists(os.path.join('upload', 'test_upload.txt')):
        os.remove(os.path.join('upload', 'test_upload.txt'))
