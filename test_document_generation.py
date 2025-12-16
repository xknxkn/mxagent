import os
import sys
import time
from datetime import datetime, timedelta

# 导入相关函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from batago import normalize_name, clean_old_student_files

# 测试目录和学生姓名
TEST_DIR = os.path.join(os.path.dirname(__file__), 'career_plans')
TEST_STUDENT = "测试学生"

def setup_test_environment():
    """设置测试环境，创建一些旧文件"""
    # 确保测试目录存在
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # 生成标准化的学生姓名
    normalized_name = normalize_name(TEST_STUDENT).replace(' ', '_')
    
    # 创建2个旧的测试文件（Word和PDF）
    old_files = []
    # 使用2天前的时间戳
    old_timestamp = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d_%H%M%S")
    
    # 创建旧的Word文件
    old_docx = f"{normalized_name}_{old_timestamp}.docx"
    old_docx_path = os.path.join(TEST_DIR, old_docx)
    with open(old_docx_path, 'w', encoding='utf-8') as f:
        f.write("This is an old Word document.")
    old_files.append(old_docx_path)
    
    # 创建旧的PDF文件
    old_pdf = f"{normalized_name}_{old_timestamp}.pdf"
    old_pdf_path = os.path.join(TEST_DIR, old_pdf)
    with open(old_pdf_path, 'w', encoding='utf-8') as f:
        f.write("This is an old PDF document.")
    old_files.append(old_pdf_path)
    
    print(f"创建了2个旧文件用于测试: {old_docx}, {old_pdf}")
    return old_files

def simulate_document_generation():
    """模拟文档生成过程"""
    # 生成标准化的学生姓名
    normalized_name = normalize_name(TEST_STUDENT).replace(' ', '_')
    
    # 使用当前时间戳作为新文件
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建新的Word文件
    new_docx = f"{normalized_name}_{current_timestamp}.docx"
    new_docx_path = os.path.join(TEST_DIR, new_docx)
    with open(new_docx_path, 'w', encoding='utf-8') as f:
        f.write("This is a new Word document.")
    
    # 创建新的PDF文件
    new_pdf = f"{normalized_name}_{current_timestamp}.pdf"
    new_pdf_path = os.path.join(TEST_DIR, new_pdf)
    with open(new_pdf_path, 'w', encoding='utf-8') as f:
        f.write("This is a new PDF document.")
    
    print(f"模拟生成了2个新文件: {new_docx}, {new_pdf}")
    return new_docx_path, new_pdf_path

def list_student_files():
    """列出学生的所有文件"""
    normalized_name = normalize_name(TEST_STUDENT).replace(' ', '_')
    return [f for f in os.listdir(TEST_DIR) if f.startswith(normalized_name)]

def main():
    print("=== 开始测试修改后的文档生成和清理功能 ===")
    
    # 步骤1: 设置测试环境，创建旧文件
    old_files = setup_test_environment()
    
    # 步骤2: 列出初始状态的文件
    print("\n初始状态的文件:")
    initial_files = list_student_files()
    print(f"找到 {len(initial_files)} 个文件: {initial_files}")
    
    # 步骤3: 模拟文档生成过程
    print("\n模拟文档生成过程...")
    new_docx_path, new_pdf_path = simulate_document_generation()
    
    # 步骤4: 在文档生成后调用clean_old_student_files（模拟修改后的行为）
    print("\n在文档生成后调用clean_old_student_files...")
    clean_result = clean_old_student_files(TEST_STUDENT)
    
    # 步骤5: 显示清理结果
    print("\n清理结果:")
    print(f"学生姓名: {clean_result['student_name']}")
    print(f"总文件数: {clean_result['total_files']}")
    print(f"删除文件数: {clean_result['deleted_count']}")
    print(f"删除的文件: {clean_result['deleted_files']}")
    print(f"保留的文件: {clean_result['kept_files']}")
    
    # 步骤6: 列出最终状态的文件
    print("\n最终状态的文件:")
    final_files = list_student_files()
    print(f"找到 {len(final_files)} 个文件: {final_files}")
    
    # 步骤7: 验证新文件是否被保留
    new_files_exist = os.path.exists(new_docx_path) and os.path.exists(new_pdf_path)
    
    # 步骤8: 验证旧文件是否被删除
    old_files_deleted = all(not os.path.exists(f) for f in old_files)
    
    # 步骤9: 显示测试结果
    print("\n=== 测试结果 ===")
    if new_files_exist:
        print("✅ 新文件被正确保留")
    else:
        print("❌ 新文件未被保留")
    
    if old_files_deleted:
        print("✅ 旧文件被正确删除")
    else:
        print("❌ 旧文件未被删除")
    
    if new_files_exist and old_files_deleted:
        print("\n🎉 测试通过! clean_old_student_files在文档生成后正确调用，保留了新文件并删除了旧文件。")
    else:
        print("\n❌ 测试失败! 功能未按预期工作。")
    
    # 清理测试文件
    print("\n清理测试文件...")
    for file_path in [new_docx_path, new_pdf_path]:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"删除测试文件: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"删除测试文件失败 {file_path}: {e}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    main()
