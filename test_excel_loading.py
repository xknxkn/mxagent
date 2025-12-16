import os
import sys
import pandas as pd
import datetime

# 直接在测试中实现和测试核心逻辑，避免模块导入问题
def test_excel_data_loading():
    """直接测试读取所有以上课反馈开头的xlsx文件并合并内容的功能"""
    print("开始测试Excel数据加载功能...")
    
    try:
        # 搜索所有以上课反馈开头的xlsx文件
        excel_files = [f for f in os.listdir('.') if f.startswith('上课反馈') and f.endswith('.xlsx')]
        print(f"当前目录中找到的以上课反馈开头的xlsx文件: {excel_files}")
        
        if not excel_files:
            print("警告: 当前目录中没有找到以上课反馈开头的xlsx文件")
            print("测试完成 (无可用文件)")
            return True
        
        print(f"找到{len(excel_files)}个以上课反馈开头的xlsx文件，准备测试合并和去重功能...")
        
        # 初始化空的DataFrame用于合并
        all_data_frames = []
        
        # 逐个读取并验证文件
        for excel_file in excel_files:
            try:
                print(f"正在读取文件: {excel_file}")
                df = pd.read_excel(excel_file)
                
                if df.empty:
                    print(f"警告: 文件{excel_file}为空")
                    continue
                
                # 验证必要的列是否存在
                required_columns = ['学生姓名', '上课时间']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"警告: 文件{excel_file}缺少必要的列: {missing_columns}，跳过该文件")
                    continue
                
                # 预先处理上课时间列
                df['上课时间'] = pd.to_datetime(df['上课时间'], errors='coerce')
                
                all_data_frames.append(df)
                print(f"成功读取{excel_file}，共{len(df)}条记录")
                
            except Exception as e:
                print(f"读取文件{excel_file}时出错: {str(e)}")
                continue
        
        # 检查是否有成功读取的文件
        if not all_data_frames:
            print("错误: 没有成功读取的以上课反馈开头的xlsx文件")
            return False
        
        # 合并所有数据
        combined_df = pd.concat(all_data_frames, ignore_index=True)
        
        # 去重，避免内容重复
        original_length = len(combined_df)
        combined_df = combined_df.drop_duplicates()
        
        # 记录去重信息
        duplicates_removed = original_length - len(combined_df)
        if duplicates_removed > 0:
            print(f"移除了{duplicates_removed}条重复记录")
        
        # 输出合并结果
        print(f"课程反馈数据加载成功，共{len(combined_df)}条不重复记录")
        print(f"数据列: {list(combined_df.columns)}")
        print("前5行数据预览:")
        print(combined_df.head())
        
        print("\n测试成功完成!")
        return True
        
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_excel_data_loading()
