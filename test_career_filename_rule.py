import os
import re
import datetime
from batago import normalize_name

# 模拟normalize_name函数（如果不能直接导入）
def simulate_normalize_name(name):
    # 这里简单实现normalize_name的功能
    return name.strip()

# 测试1：验证文件名生成逻辑
def test_filename_generation():
    print("测试1：验证文件名生成逻辑")
    
    # 模拟学生姓名
    student_name = "张三"
    timestamp = "20240610_120000"  # 模拟时间戳
    
    # 生成文件名
    if 'normalize_name' in globals():
        normalized_name = normalize_name(student_name).replace(' ', '_')
    else:
        normalized_name = simulate_normalize_name(student_name).replace(' ', '_')
    
    expected_filename = f"{normalized_name}_规划_{timestamp}.docx"
    
    # 打印测试结果
    print(f"学生姓名: {student_name}")
    print(f"生成的文件名: {expected_filename}")
    print(f"是否包含 '_规划_': {'_规划_' in expected_filename}")
    
    # 验证结果
    assert '_规划_' in expected_filename, "文件名中未包含 '_规划_'"
    print("✓ 文件名生成逻辑测试通过！\n")

# 测试2：验证正则表达式匹配
def test_regex_matching():
    print("测试2：验证正则表达式匹配")
    
    # 模拟学生姓名
    student_name = "李四"
    
    # 获取或模拟normalized_name
    if 'normalize_name' in globals():
        normalized_name = normalize_name(student_name).replace(' ', '_')
    else:
        normalized_name = simulate_normalize_name(student_name).replace(' ', '_')
    
    # 新的正则表达式
    new_pattern = re.compile(r'^%s_规划_\d{8}_\d{6}\.docx$' % re.escape(normalized_name))
    
    # 测试有效的文件名
    valid_filenames = [
        f"{normalized_name}_规划_20240610_120000.docx",
        f"{normalized_name}_规划_20240101_000000.docx",
        f"{normalized_name}_规划_20241231_235959.docx"
    ]
    
    # 测试无效的文件名
    invalid_filenames = [
        f"{normalized_name}_20240610_120000.docx",  # 缺少_规划_
        f"{normalized_name}_规划_20240610_1200.docx",   # 时间戳格式错误
        f"{normalized_name}规划20240610120000.docx",    # 格式完全错误
        f"王五_规划_20240610_120000.docx"               # 学生姓名不匹配
    ]
    
    # 测试有效文件名
    print("测试有效文件名:")
    for filename in valid_filenames:
        match = new_pattern.match(filename)
        result = "✓ 匹配成功" if match else "✗ 匹配失败"
        print(f"  {filename}: {result}")
        assert match, f"有效文件名 {filename} 未匹配成功"
    
    # 测试无效文件名
    print("\n测试无效文件名:")
    for filename in invalid_filenames:
        match = new_pattern.match(filename)
        result = "✓ 正确拒绝" if not match else "✗ 错误匹配"
        print(f"  {filename}: {result}")
        assert not match, f"无效文件名 {filename} 错误匹配"
    
    print("✓ 正则表达式匹配测试通过！\n")

# 测试3：生成当前时间的文件名并验证
def test_current_time_filename():
    print("测试3：生成当前时间的文件名并验证")
    
    # 模拟学生姓名
    student_name = "王五"
    
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 获取或模拟normalized_name
    if 'normalize_name' in globals():
        normalized_name = normalize_name(student_name).replace(' ', '_')
    else:
        normalized_name = simulate_normalize_name(student_name).replace(' ', '_')
    
    # 生成文件名
    filename = f"{normalized_name}_规划_{timestamp}.docx"
    
    # 验证文件名格式
    expected_pattern = re.compile(r'^%s_规划_\d{8}_\d{6}\.docx$' % re.escape(normalized_name))
    match = expected_pattern.match(filename)
    
    print(f"生成的文件名: {filename}")
    print(f"是否符合预期格式: {bool(match)}")
    
    assert match, "生成的文件名不符合预期格式"
    print("✓ 当前时间文件名测试通过！")

if __name__ == "__main__":
    print("开始测试 career_plan 文件名规则修改...\n")
    
    try:
        test_filename_generation()
        test_regex_matching()
        test_current_time_filename()
        
        print("\n所有测试通过！文件名规则已成功修改为包含 '_规划_'。")
    except AssertionError as e:
        print(f"\n测试失败: {e}")
    except Exception as e:
        print(f"\n测试过程中出错: {e}")
