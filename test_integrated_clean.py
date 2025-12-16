import os
import sys
import datetime

# 导入gradiostudentsum模块中的clean_old_student_files函数
sys.path.append('.')
try:
    from gradiostudentsum import clean_old_student_files, normalize_name
    print("成功导入clean_old_student_files函数")
except ImportError as e:
    print(f"导入函数失败: {e}")
    print("请确保gradiostudentsum.py文件存在且包含clean_old_student_files函数")
    sys.exit(1)

def create_test_files(student_name, count=3):
    """
    为指定学生创建测试文件，用于测试删除功能
    
    Args:
        student_name: 学生姓名
        count: 创建的文件数量
    
    Returns:
        list: 创建的文件路径列表
    """
    # 定义career_plans目录路径
    career_dir = os.path.join(os.path.dirname(__file__), 'career_plans')
    if not os.path.exists(career_dir):
        os.makedirs(career_dir)
    
    created_files = []
    normalized_name = normalize_name(student_name).replace(' ', '_')
    
    # 创建测试文件，时间间隔为1小时
    base_time = datetime.datetime.now()
    for i in range(count):
        # 生成不同时间的文件名
        test_time = base_time - datetime.timedelta(hours=i)
        timestamp = test_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{normalized_name}_{timestamp}.docx"
        file_path = os.path.join(career_dir, filename)
        
        # 创建空文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"测试文件内容 - {student_name} - {test_time}")
        
        created_files.append(file_path)
        print(f"创建测试文件: {filename}")
    
    return created_files

def test_clean_function():
    """
    测试clean_old_student_files函数
    """
    # 测试学生姓名
    test_student = "测试学生"
    
    try:
        # 1. 创建测试文件
        print(f"\n=== 创建测试文件 ===")
        created_files = create_test_files(test_student, 3)
        
        # 2. 运行清理函数
        print(f"\n=== 运行清理函数 ===")
        result = clean_old_student_files(test_student)
        print(f"清理结果: {result}")
        
        # 3. 验证结果
        print(f"\n=== 验证结果 ===")
        normalized_name = normalize_name(test_student).replace(' ', '_')
        career_dir = os.path.join(os.path.dirname(__file__), 'career_plans')
        
        # 获取剩余的文件
        remaining_files = []
        for filename in os.listdir(career_dir):
            if filename.startswith(normalized_name) and filename.endswith('.docx'):
                remaining_files.append(filename)
        
        print(f"清理后剩余文件数量: {len(remaining_files)}")
        print(f"剩余文件: {remaining_files}")
        
        # 验证是否只保留了一个文件
        if len(remaining_files) == 1:
            print("✅ 测试通过: 成功只保留了一个最新的文件")
            # 检查是否保留的是最新的文件
            if result['kept_file'] in remaining_files:
                print(f"✅ 保留的是预期的最新文件: {result['kept_file']}")
        else:
            print(f"❌ 测试失败: 清理后应该只保留一个文件，但实际保留了 {len(remaining_files)} 个文件")
        
        # 4. 清理测试文件
        print(f"\n=== 清理测试文件 ===")
        for file_path in created_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"删除测试文件: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"删除文件失败 {file_path}: {e}")
        
        return True
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试clean_old_student_files函数...")
    success = test_clean_function()
    
    if success:
        print("\n🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 测试失败！")
        sys.exit(1)
