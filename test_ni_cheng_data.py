import pandas as pd
import os
from datetime import datetime

# 首先定义normalize_name函数，与batago.py中保持一致
def normalize_name(name):
    # 去除空格，转为小写
    return ''.join(name.split()).lower()

# 手动加载并检查数据
def check_ni_cheng_data():
    print("开始检查倪承同学的数据...")
    
    # 搜索Excel文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [current_dir, os.path.join(current_dir, "upload")]
    all_excel_files = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir) and os.path.isdir(search_dir):
            excel_files_in_dir = [
                os.path.join(search_dir, f) 
                for f in os.listdir(search_dir) 
                if f.startswith('上课反馈') and f.endswith('.xlsx')
            ]
            all_excel_files.extend(excel_files_in_dir)
    
    print(f"找到{len(all_excel_files)}个Excel文件")
    
    # 加载所有数据
    all_data_frames = []
    for excel_file in all_excel_files:
        try:
            print(f"读取文件: {os.path.basename(excel_file)}")
            df = pd.read_excel(excel_file)
            
            # 转换上课时间
            if '上课时间' in df.columns:
                df['上课时间'] = pd.to_datetime(df['上课时间'], errors='coerce')
            
            all_data_frames.append(df)
        except Exception as e:
            print(f"读取文件{os.path.basename(excel_file)}时出错: {str(e)}")
    
    if not all_data_frames:
        print("没有成功读取任何文件")
        return
    
    # 合并数据
    dfkcjl = pd.concat(all_data_frames, ignore_index=True)
    print(f"总记录数: {len(dfkcjl)}")
    
    # 查找倪承同学的记录
    dfkcjl['学生姓名_标准'] = dfkcjl['学生姓名'].astype(str).apply(normalize_name)
    ni_cheng = normalize_name('倪承')
    ni_cheng_df = dfkcjl[dfkcjl['学生姓名_标准'] == ni_cheng]
    
    print(f"倪承同学的记录数: {len(ni_cheng_df)}")
    
    # 分析日期范围
    if '上课时间' in ni_cheng_df.columns and not ni_cheng_df['上课时间'].empty:
        min_date = ni_cheng_df['上课时间'].min()
        max_date = ni_cheng_df['上课时间'].max()
        print(f"最早日期: {min_date}")
        print(f"最晚日期: {max_date}")
        
        # 检查是否有2022年的数据
        year_2022 = ni_cheng_df[ni_cheng_df['上课时间'].dt.year == 2022]
        print(f"2022年记录数: {len(year_2022)}")
        
        # 显示2022年的详细记录
        if len(year_2022) > 0:
            print("\n2022年详细记录:")
            print(year_2022[['上课时间', '课时消耗', '上课状态']].head(20))
        
        # 按年份分组统计
        print("\n按年份统计:")
        year_counts = ni_cheng_df.groupby(ni_cheng_df['上课时间'].dt.year)['上课时间'].count()
        print(year_counts)
        
        # 检查课时消耗
        if '课时消耗' in ni_cheng_df.columns:
            print("\n课时消耗分析:")
            print(f"总课时消耗: {ni_cheng_df['课时消耗'].sum()}")
            
            # 尝试数值转换后求和
            try:
                valid_hours = pd.to_numeric(ni_cheng_df['课时消耗'], errors='coerce')
                valid_hours = valid_hours[(valid_hours >= 0) & (valid_hours <= 10)]
                print(f"有效课时消耗(过滤后): {valid_hours.sum()}")
            except Exception as e:
                print(f"课时转换错误: {e}")
    
    # 检查是否有数据清洗或过滤的可能问题
    print("\n检查数据质量:")
    print(f"NaN值统计:")
    print(ni_cheng_df.isna().sum())
    
    # 检查上课状态
    if '上课状态' in ni_cheng_df.columns:
        print("\n上课状态分布:")
        print(ni_cheng_df['上课状态'].value_counts())

if __name__ == "__main__":
    check_ni_cheng_data()
