import os
import re
import datetime
from batago import normalize_name

# 模拟测试文件名生成逻辑
def test_filename_generation():
    print("开始测试文件名格式生成...")
    
    # 测试不同情况下的文件名生成
    test_cases = [
        {"name": "测试学生", "mapping": ["2025年11月 -> 2025-11-01 ~ 2025-11-30"]},
        {"name": "张三李四", "mapping": ["最近三个月 -> 2025-09-14 ~ 2025-12-14"]},
        {"name": "王五", "mapping": []}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        correct_student_name = test_case["name"]
        mapping = test_case["mapping"]
        
        print(f"\n测试用例 {i}: 学生 '{correct_student_name}', mapping = {mapping}")
        
        # 复制修改后的文件名生成逻辑
        start_date_str = ""
        end_date_str = ""
        if mapping and len(mapping) > 0:
            mapping_str = mapping[0]
            date_range_match = re.search(r'(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})', mapping_str)
            if date_range_match:
                start_date_str = date_range_match.group(1).replace('-', '')
                end_date_str = date_range_match.group(2).replace('-', '')
            else:
                current_date = datetime.datetime.now().strftime('%Y%m%d')
                start_date_str = current_date
                end_date_str = current_date
        else:
            current_date = datetime.datetime.now().strftime('%Y%m%d')
            start_date_str = current_date
            end_date_str = current_date
        
        # 生成精确到秒的时间戳
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 生成测试文件名
        safe_filename = f"{normalize_name(correct_student_name).replace(' ', '_')}_总结_{start_date_str}_{end_date_str}_{timestamp}.docx"
        print(f"生成的文件名: {safe_filename}")
        
        # 验证文件名格式
        expected_pattern = r'^[\u4e00-\u9fa5a-zA-Z0-9_]+_总结_\d{8}_\d{8}_\d{8}_\d{6}\.docx$'
        if re.match(expected_pattern, safe_filename):
            print("✅ 文件名格式验证通过")
            
            # 提取并验证各部分内容
            parts = safe_filename.split('_')
            if len(parts) >= 6 and parts[-2].isdigit() and parts[-3].isdigit() and parts[-4].isdigit():
                print(f"  - 学生姓名部分: {'_'.join(parts[:-5])}")
                print(f"  - 开始日期: {parts[-4]}")
                print(f"  - 结束日期: {parts[-3]}")
                print(f"  - 时间戳: {parts[-2]}")
                print(f"  - 文件扩展名: {parts[-1]}")
        else:
            print("❌ 文件名格式验证失败")
    
    # 测试实际生成文件（可选，这里只做模拟）
    print("\n文件名格式测试完成。新格式将在实际调用generate_summary时生效。")
    print("新文件名格式: 学生姓名_总结_开始日期_结束日期_时间戳.docx")
    print("其中时间戳格式: YYYYMMDD_HHMMSS")

if __name__ == "__main__":
    test_filename_generation()
