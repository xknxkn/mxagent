import os
import sys
import shutil
from datetime import datetime, timedelta

# 导入修复后的函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from batago import clean_old_student_files, normalize_name

# 测试目录和学生姓名
TEST_DIR = os.path.join(os.path.dirname(__file__), 'career_plans')
TEST_STUDENT = "测试学生"

def setup_test_files():
    """创建测试文件"""
    # 确保测试目录存在
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # 创建测试文件，使用不同的时间戳
    test_files = []
    normalized_name = normalize_name(TEST_STUDENT).replace(' ', '_')
    
    # 创建3个测试文件，分别是3天前、2天前和1天前的
    for i in range(3):
        timestamp = (datetime.now() - timedelta(days=3-i)).strftime("%Y%m%d_%H%M%S")
        filename = f"{normalized_name}_{timestamp}.docx"
        file_path = os.path.join(TEST_DIR, filename)
        
        # 创建空文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Test content for {filename}")
        
        test_files.append(file_path)
        print(f"创建测试文件: {filename}")
    
    return test_files

def cleanup_test_files(files):
    """清理测试文件"""
    for file_path in files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"删除测试文件: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"删除测试文件失败 {file_path}: {e}")

def main():
    print("=== 开始测试修复后的clean_old_student_files函数 ===")
    
    # 创建测试文件
    test_files = setup_test_files()
    
    try:
        # 列出修复前的文件
        print("\n修复前的文件:")
        files_before = [f for f in os.listdir(TEST_DIR) if f.startswith(normalize_name(TEST_STUDENT).replace(' ', '_'))]
        print(f"找到 {len(files_before)} 个文件: {files_before}")
        
        # 执行清理函数
        print("\n执行clean_old_student_files函数...")
        result = clean_old_student_files(TEST_STUDENT)
        
        # 显示清理结果
        print("\n清理结果:")
        print(f"学生姓名: {result['student_name']}")
        print(f"总文件数: {result['total_files']}")
        print(f"删除文件数: {result['deleted_count']}")
        print(f"删除的文件: {result['deleted_files']}")
        print(f"保留的文件: {result['kept_files']}")
        
        # 列出修复后的文件
        print("\n修复后的文件:")
        files_after = [f for f in os.listdir(TEST_DIR) if f.startswith(normalize_name(TEST_STUDENT).replace(' ', '_'))]
        print(f"找到 {len(files_after)} 个文件: {files_after}")
        
        # 分析结果
        print("\n分析结果:")
        if len(files_after) < len(files_before):
            print(f"✅ 测试通过! 成功删除了{len(files_before) - len(files_after)}个旧文件")
        else:
            print(f"❌ 测试失败! 没有删除任何文件")
    finally:
        # 清理测试文件
        print("\n清理测试环境...")
        cleanup_test_files(test_files)
        print("测试完成!")

if __name__ == "__main__":
    main()
