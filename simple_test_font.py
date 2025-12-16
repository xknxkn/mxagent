import os
import sys
import tempfile
import pypandoc

def test_font_settings():
    """测试Markdown转Word时的字体设置"""
    print("=== 简化版字体测试程序 ===")
    
    # 创建测试Markdown内容
    markdown_content = """# 测试文档

## 这是中文标题

这是一段中文内容，用于测试字体设置。

**加粗中文** 和 *斜体中文*
"""
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(markdown_content)
        md_path = f.name
    
    docx_path = md_path.replace('.md', '.docx')
    
    try:
        # 使用更有效的字体设置参数
        # 对于Word转换，使用docx格式时，应该使用不同的参数
        extra_args = [
            '--standalone',
            '--from=markdown',
            '--to=docx',
            '--wrap=none',
            # Word特定的字体设置方式
            '--variable=mainfont=SimSun',
            '--variable=fontfamily=SimSun'
        ]
        
        # 执行转换
        print(f"正在将Markdown转换为Word...")
        pypandoc.convert_file(
            md_path,
            'docx',
            outputfile=docx_path,
            extra_args=extra_args
        )
        
        print(f"✅ 转换完成: {docx_path}")
        print("\n注意：")
        print("1. python-docx库可能无法正确检测通过pandoc转换的文档字体")
        print("2. 请手动打开生成的Word文件检查字体是否为宋体")
        print("3. 在Word中，您可以通过'格式'或'字体'对话框查看字体信息")
        
        # 推荐的改进方法：使用参考文档
        print("\n推荐改进方案:")
        print("1. 创建一个已设置好宋体的Word模板文件(template.docx)")
        print("2. 使用 --reference-doc=template.docx 参数进行转换")
        print("3. 这样可以确保所有样式（包括字体）都正确应用")
        
    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(md_path):
            os.remove(md_path)
        
        print(f"\n测试文件位置: {docx_path}")
        print("请手动打开并检查字体设置")

def create_batago_fix_example():
    """生成batago.py中应该使用的改进后的字体设置代码"""
    fix_code = """
# 改进后的Word文档转换参数
# 对于docx格式，应该使用以下参数设置中文字体为宋体
extra_args = [
    '--standalone',
    '--from=markdown',
    '--to=docx',
    '--wrap=none',
    # 以下是Word文档特定的字体设置
    '--reference-doc=template.docx'  # 使用参考文档（推荐方式）
    # 或者使用以下变量设置
    # '--variable=mainfont=SimSun',
    # '--variable=fontfamily=SimSun'
]

# 重要说明：
# 1. 最好的方式是创建一个已设置好宋体的Word模板文件
# 2. 将其保存为template.docx放在程序目录下
# 3. 使用--reference-doc参数引用该模板
# 4. 这样可以确保所有样式都正确应用，包括标题、正文、表格等
"""
    
    print("\n=== batago.py 推荐的字体设置改进 ===")
    print(fix_code)

if __name__ == "__main__":
    # 检查pypandoc是否安装
    try:
        import pypandoc
        print("✅ pypandoc库已安装")
    except ImportError:
        print("❌ pypandoc库未安装，请运行: pip install pypandoc")
        sys.exit(1)
    
    # 运行测试
    test_font_settings()
    
    # 显示改进建议
    create_batago_fix_example()
