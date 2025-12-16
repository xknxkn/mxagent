import os
import sys
import datetime

# 简单测试函数
def test_word_conversion():
    try:
        # 确保career_plans目录存在
        if not os.path.exists('career_plans'):
            os.makedirs('career_plans')
        
        # 使用career_planning函数的核心转换逻辑
        # 模拟一些数据进行测试
        student_name = "测试学生"
        career_target = "软件工程师"
        
        # 生成简单的markdown内容
        markdown_content = """
# 测试文档

## 章节一
这是一个测试文档，用于验证markdown转Word功能。

## 章节二
- 项目1
- 项目2
- 项目3

## 表格测试
| 名称 | 描述 |
|------|------|
| 项目A | 描述A |
| 项目B | 描述B |
"""
        
        # 生成Word文件名（使用学生姓名和当前时间）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"xkn_{timestamp}.docx"
        pdf_dir = "career_plans"
        docx_path = os.path.join(pdf_dir, safe_filename)
        
        # 导入必要的模块
        import subprocess
        import pypandoc
        
        # 检查pandoc是否安装
        try:
            result = subprocess.run(['pandoc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                pandoc_available = True
                print(f"Pandoc检查成功")
                print(f"Pandoc版本: {result.stdout.splitlines()[0]}")
            else:
                pandoc_available = False
                print(f"Pandoc命令返回错误码: {result.returncode}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            pandoc_available = False
            print(f"检测Pandoc时出错: {str(e)}")
            print("尝试直接使用pypandoc进行转换...")
        
        try:
            # 在markdown内容前添加标题和元信息
            enhanced_markdown = f"""# 学生职业规划 - {student_name}

## 基本信息

**目标职业:** {career_target}
**生成时间:** {datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

---

{markdown_content}
"""
            
            # 使用pypandoc将markdown转换为docx
            # 首先将markdown内容写入临时文件
            temp_md_path = os.path.join(pdf_dir, f"temp_{timestamp}.md")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(temp_md_path), exist_ok=True)
            
            try:
                with open(temp_md_path, 'w', encoding='utf-8') as f:
                    f.write(enhanced_markdown)
                print(f"临时Markdown文件已创建: {temp_md_path}")
            except Exception as file_error:
                raise Exception(f"创建临时Markdown文件失败: {str(file_error)}")
            
            # 使用pypandoc将markdown文件转换为docx
            try:
                # 设置额外的参数以优化转换，包括设置字体为宋体
                extra_args = [
                    '--standalone',
                    '--from=markdown',
                    '--to=docx',
                    '--wrap=none',
                    '--variable=mainfont="SimSun"',  # 设置主要字体为宋体
                    '--variable=fontfamily="SimSun"',  # 设置字体家族为宋体
                    '--variable=fontsize=12pt'  # 设置字号
                ]
                
                pypandoc.convert_file(
                    temp_md_path,
                    'docx',
                    outputfile=docx_path,
                    extra_args=extra_args
                )
                print(f"Word文档转换成功: {docx_path}")
            except Exception as convert_error:
                # 检查是否是因为pandoc本身的问题
                if 'pandoc' in str(convert_error).lower():
                    raise Exception(f"Pandoc转换失败: {str(convert_error)}。请确保已正确安装pandoc。")
                else:
                    raise Exception(f"文档转换过程中出错: {str(convert_error)}")
            finally:
                # 确保无论如何都删除临时文件
                try:
                    if os.path.exists(temp_md_path):
                        os.remove(temp_md_path)
                        print(f"临时文件已删除: {temp_md_path}")
                except Exception as cleanup_error:
                    print(f"清理临时文件时出错: {str(cleanup_error)}")
            
            # 验证生成的文件是否存在且非空
            if os.path.exists(docx_path) and os.path.getsize(docx_path) > 0:
                print(f"Word文档已成功生成并保存至: {docx_path}")
                print("测试成功!")
            else:
                raise Exception(f"生成的Word文档可能为空或未正确创建: {docx_path}")
                
        except Exception as docx_error:
            error_message = f"生成Word文档时出错: {str(docx_error)}"
            print(error_message)
            
            # 如果pandoc不可用，提供安装指南
            if not pandoc_available:
                print("\n注意：系统中未检测到pandoc命令行工具。请安装pandoc后重试。")
                print("Windows用户可以从 https://github.com/jgm/pandoc/releases 下载安装程序。")
                print("安装后可能需要重启计算机以更新环境变量。")
            print("测试失败!")
            
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        print("测试失败!")

if __name__ == "__main__":
    print("开始测试Word转换功能...")
    test_word_conversion()
