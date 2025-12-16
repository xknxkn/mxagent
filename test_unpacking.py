import sys
import datetime
import re

# 导入parse_time_signal函数
sys.path.append('.')
from gradiostudentsum import parse_time_signal

# 测试函数
def test_unpacking():
    print("测试parse_time_signal函数的解包功能...")
    
    # 测试各种时间输入
    test_inputs = ['最近一周', '最近', '近期', '今年', '']
    
    for time_input in test_inputs:
        try:
            print(f"\n测试输入: '{time_input}'")
            # 解包函数返回值
            sname, (start, end) = parse_time_signal(time_input)
            
            print(f"sname = {sname}")
            print(f"start = {start}")
            print(f"end = {end}")
            print("解包成功!")
        except Exception as e:
            print(f"解包失败: {e}")

if __name__ == "__main__":
    test_unpacking()
