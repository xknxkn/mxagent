import pandas as pd
import datetime
import os
import re

def normalize_name(s: str) -> str:
    """Normalize Chinese names by removing extra spaces and suffixes."""
    import unicodedata
    s = unicodedata.normalize('NFKC', str(s))
    s = ''.join(s.split())
    if s.endswith('-综评熊学科'):
        s = s[:-7]
    return s

def load_excel_data():
    """加载Excel课程反馈数据到全局变量，读取所有以上课反馈开头的xlsx文件并合并"""
    try:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 正在搜索并加载所有以上课反馈开头的xlsx文件...")
        
        # 搜索所有以上课反馈开头的xlsx文件
        excel_files = [f for f in os.listdir('.') if f.startswith('上课反馈') and f.endswith('.xlsx')]
        
        if not excel_files:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 错误: 未找到以上课反馈开头的xlsx文件")
            return None
        
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 找到{len(excel_files)}个以上课反馈开头的xlsx文件: {', '.join(excel_files)}")
        
        # 初始化空的DataFrame用于合并
        all_data_frames = []
        
        # 逐个读取并验证文件
        for excel_file in excel_files:
            try:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 正在读取文件: {excel_file}")
                df = pd.read_excel(excel_file)
                
                if df.empty:
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 警告: 文件{excel_file}为空")
                    continue
                
                # 验证必要的列是否存在
                required_columns = ['学生姓名', '上课时间']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 警告: 文件{excel_file}缺少必要的列: {missing_columns}，跳过该文件")
                    continue
                
                # 预先处理上课时间列
                df['上课时间'] = pd.to_datetime(df['上课时间'], errors='coerce')
                
                all_data_frames.append(df)
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 成功读取{excel_file}，共{len(df)}条记录")
                
            except Exception as e:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 读取文件{excel_file}时出错: {str(e)}")
                continue
        
        # 检查是否有成功读取的文件
        if not all_data_frames:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 错误: 没有成功读取的以上课反馈开头的xlsx文件")
            return None
        
        # 合并所有数据
        dfkcjl = pd.concat(all_data_frames, ignore_index=True)
        
        # 去重，避免内容重复
        original_length = len(dfkcjl)
        dfkcjl = dfkcjl.drop_duplicates()
        
        # 记录去重信息
        duplicates_removed = original_length - len(dfkcjl)
        if duplicates_removed > 0:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 移除了{duplicates_removed}条重复记录")
        
        # 记录加载成功信息
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 课程反馈数据加载成功，共{len(dfkcjl)}条不重复记录")
        return dfkcjl
        
    except Exception as e:
        print(f"加载课程反馈数据时发生未知错误: {str(e)}")
        return None

# 加载数据
df = load_excel_data()
if df is None:
    print("无法加载数据")
    exit()

# 指定学生姓名和时间范围
student_name = "吴天昊"
start_date = datetime.datetime(2025, 11, 13)
end_date = datetime.datetime(2025, 12, 13)

# 标准化学生姓名
target = normalize_name(student_name)

# 过滤学生记录
student_df = df[df['学生姓名'].astype(str).apply(normalize_name) == target]

print(f"学生 '{student_name}' (标准化: '{target}') 的总记录数: {len(student_df)}")

# 过滤时间范围
sel = student_df[(student_df['上课时间'] >= start_date) & (student_df['上课时间'] <= end_date)]

print(f"在时间范围 {start_date.date()} ~ {end_date.date()} 内的记录数: {len(sel)}")

if len(sel) > 0:
    print("记录详情:")
    for idx, row in sel.iterrows():
        print(f"- 上课时间: {row['上课时间']}, 内容: {row['内容'][:100]}...")
else:
    print("在指定时间范围内没有记录")