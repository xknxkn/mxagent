# 这是一个示例代码片段，演示如何修复 NameError
# 错误原因：student_df 变量没有在 sel = ... 之前定义

import pandas as pd
import datetime

# 示例数据（实际使用时需要从你的数据源加载）
# 假设 df 是你的 DataFrame
df = pd.DataFrame({
    '学生姓名_标准': ['张三', '李四', '王五'],
    '上课时间': [datetime.datetime(2025, 10, 1), datetime.datetime(2025, 11, 1), datetime.datetime(2025, 12, 1)]
})

# 定义 target（学生姓名）
target = '张三'

# 首先定义 student_df
student_df = df[df['学生姓名_标准'] == target]

# 定义时间范围
start = datetime.datetime(2025, 11, 1)
end = datetime.datetime(2025, 12, 1)

# 现在可以使用 student_df
sel = student_df[(student_df['上课时间'] >= start) & (student_df['上课时间'] <= end)]

print(f"找到 {len(sel)} 条记录")
print(sel)