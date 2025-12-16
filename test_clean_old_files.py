import os
import re
from datetime import datetime

# 定义career_plans目录路径
CAREER_PLANS_DIR = os.path.join(os.path.dirname(__file__), 'career_plans')

def clean_old_student_files(student_name):
    """
    删除指定学生的旧文件，只保留最新的文件
    
    Args:
        student_name: 学生姓名
    
    Returns:
        dict: 删除的文件信息
    """
    # 文件命名规则：学生姓名_年月日_时分秒.docx
    file_pattern = re.compile(r'^%s_(\d{8})_(\d{6})\.docx$' % re.escape(student_name))
    
    # 获取目录中所有文件
    all_files = os.listdir(CAREER_PLANS_DIR)
    
    # 收集该学生的所有文件
    student_files = []
    for filename in all_files:
        match = file_pattern.match(filename)
        if match:
            date_str, time_str = match.groups()
            # 组合成完整的时间字符串
            datetime_str = f'{date_str}{time_str}'
            # 转换为datetime对象用于比较
            file_datetime = datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
            student_files.append({
                'filename': filename,
                'datetime': file_datetime
            })
    
    # 如果该学生有多个文件
    if len(student_files) > 1:
        # 按时间排序，最新的在前
        student_files.sort(key=lambda x: x['datetime'], reverse=True)
        
        # 保留最新的文件，删除其余的
        files_to_delete = student_files[1:]
        deleted_count = 0
        deleted_files = []
        
        for file_info in files_to_delete:
            file_path = os.path.join(CAREER_PLANS_DIR, file_info['filename'])
            try:
                os.remove(file_path)
                deleted_count += 1
                deleted_files.append(file_info['filename'])
                print(f"已删除旧文件: {file_info['filename']}")
            except Exception as e:
                print(f"删除文件 {file_info['filename']} 失败: {e}")
        
        return {
            'student_name': student_name,
            'total_files': len(student_files),
            'deleted_count': deleted_count,
            'deleted_files': deleted_files,
            'kept_file': student_files[0]['filename']
        }
    else:
        # 没有或只有一个文件，不需要删除
        return {
            'student_name': student_name,
            'total_files': len(student_files),
            'deleted_count': 0,
            'deleted_files': [],
            'kept_file': student_files[0]['filename'] if student_files else None
        }

# 测试函数
def test_clean_old_files():
    # 测试几个学生
    students_to_test = ['倪承', '吴天昊', 'xkn']
    
    for student in students_to_test:
        print(f"\n处理学生: {student}")
        result = clean_old_student_files(student)
        print(f"结果: {result}")

if __name__ == '__main__':
    test_clean_old_files()
