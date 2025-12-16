import sys
import datetime

# 导入必要的函数和变量
sys.path.append('.')
from gradiostudentsum import parse_time_signal

# 模拟generate_summary函数中的关键部分
def test_mapping():
    print("测试mapping变量定义和格式化...")
    
    # 测试各种时间输入
    test_inputs = ['最近一周', '最近', '近期', '今年', '']
    
    for time_input in test_inputs:
        try:
            print(f"\n测试输入: '{time_input}'")
            # 调用parse_time_signal函数
            sname, (start, end) = parse_time_signal(time_input)
            
            # 定义mapping变量
            mapping = [f"{sname}: {start.strftime('%Y-%m-%d')} 至 {end.strftime('%Y-%m-%d')}"]
            
            print(f"sname = {sname}")
            print(f"start = {start}")
            print(f"end = {end}")
            print(f"mapping = {mapping}")
            print(f"格式化后的时间映射: {'; '.join(mapping)}")
            print("测试成功!")
        except Exception as e:
            print(f"测试失败: {e}")

if __name__ == "__main__":
    test_mapping()
