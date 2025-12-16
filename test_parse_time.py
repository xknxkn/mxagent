import sys
import datetime
import re

def quarter_bounds(now, offset):
    # 简化的quarter_bounds函数实现
    q = (now.month - 1) // 3
    year = now.year
    start_month = q * 3 + 1 + offset * 3
    if start_month > 12:
        start_month -= 12
        year += 1
    elif start_month <= 0:
        start_month += 12
        year -= 1
    start = datetime.datetime(year, start_month, 1)
    end_month = start_month + 2
    if end_month > 12:
        end_month = 12
    if end_month in [1,3,5,7,8,10,12]:
        last = 31
    elif end_month in [4,6,9,11]:
        last = 30
    else:
        last = 29 if (year%4==0 and (year%100!=0 or year%400==0)) else 28
    end = datetime.datetime(year, end_month, last)
    return start, end

# 复制修复后的parse_time_signal函数
def parse_time_signal(ts: str):
    s = (ts or '').strip()
    
    now = datetime.datetime.now()
    
    # 空字符串返回默认值
    if not s:
        return ('默认:上个季度', quarter_bounds(now, -1))
    
    # 按优先级检查时间范围，一旦匹配就立即返回
    if re.search(r'最近一周|近一周|上周', s):
        return ('最近一周', (now - datetime.timedelta(days=7), now))
    
    if re.search(r'最近三个月|近三个月|过去三个月', s):
        return ('最近三个月', (now - datetime.timedelta(days=90), now))
    
    if re.search(r'近[一二三四五六七八九十]?天', s):
        return ('最近几天', (now - datetime.timedelta(days=7), now))
    
    if re.search(r'今年', s):
        return ('今年', (datetime.datetime(now.year,1,1), now))
    
    if re.search(r'这一季度|本季度|这季度', s):
        st, _ = quarter_bounds(now, 0)
        return ('这一季度', (st, now))
    
    if re.search(r'上个季度|上一季度|上季度', s):
        return ('上个季度', quarter_bounds(now, -1))
    
    if re.search(r'上个月|上月', s):
        y = now.year; m = now.month - 1
        if m == 0: m = 12; y -= 1
        if m in [1,3,5,7,8,10,12]: last = 31
        elif m in [4,6,9,11]: last = 30
        else: last = 29 if (y%4==0 and (y%100!=0 or y%400==0)) else 28
        return ('上个月', (datetime.datetime(y,m,1), datetime.datetime(y,m,last)))
    
    y = re.search(r'(20\d{2})', s)
    if y:
        yy = int(y.group(1))
        return (f'{yy}年', (datetime.datetime(yy,1,1), datetime.datetime(yy,12,31)))
    
    # 检查一般的"最近"、"近几"或"近期"
    if re.search(r'最近|近几|近期', s):
        return ('最近(30天)', (now - datetime.timedelta(days=30), now))
    
    # 如果没有任何匹配，返回默认值
    return ('默认(90天)', (now - datetime.timedelta(days=90), now))

# 测试函数
def test_parse_time():
    test_cases = [
        "最近一周",
        "最近30天",
        "最近",
        "近期",  # 添加近期测试用例
        "近三天",
        "最近三个月",
        "今年",
        ""
    ]
    
    print("测试parse_time_signal函数:")
    print("-" * 50)
    
    for test_case in test_cases:
        result = parse_time_signal(test_case)
        print(f"输入: '{test_case}'")
        print(f"输出: {result[0]}")  # 只打印描述部分
        print()
    
    # 特别检查"最近一周"的匹配结果
    week_result = parse_time_signal("最近一周")
    # 特别检查"最近"和"近期"是否匹配到"最近(30天)"
    recent_result = parse_time_signal("最近")
    nearby_result = parse_time_signal("近期")
    
    print("特别验证:")
    print(f"'最近一周'的匹配结果类型: {type(week_result).__name__}")
    print(f"'最近一周'匹配到的内容: {week_result[0]}")
    print(f"'最近'是否匹配到'最近(30天)': {'最近(30天)' == recent_result[0]}")
    print(f"'近期'是否匹配到'最近(30天)': {'最近(30天)' == nearby_result[0]}")

if __name__ == "__main__":
    test_parse_time()
