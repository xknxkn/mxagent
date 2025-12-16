import os
import sys
import tempfile
import pypandoc
from docx import Document
from docx.shared import Inches

def create_test_markdown():
    """创建一个包含中文内容的测试Markdown文件"""
    markdown_content = """# 测试文档

## 这是一个测试标题

这是一段包含中文的测试文本，用于验证字体设置是否正确。

### 列表测试
- 第一项内容
- 第二项内容
- 第三项内容

**加粗文本** 和 *斜体文本*

> 这是一段引用内容
"""
    
    # 创建临时Markdown文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(markdown_content)
        return f.name

def convert_md_to_docx(md_path, docx_path):
    """使用pypandoc将Markdown转换为Word，使用与batago.py相同的参数"""
    try:
        # 使用与batago.py相同的字体参数
        extra_args = [
            '--standalone',
            '--from=markdown',
            '--to=docx',
            '--wrap=none',
            '--variable=mainfont="SimSun"',  # 设置主要字体为宋体
            '--variable=fontfamily="SimSun"',  # 设置字体家族为宋体
            '--variable=fontsize=12pt',  # 设置字号
            '--variable=CJKmainfont="SimSun"',  # 明确设置CJK字体为宋体
            '--variable=CJKfontsize=12pt',  # 设置CJK字体大小
            '--variable=documentclass=article',
            '--variable=mainfontoptions="Mapping=tex-text"'
        ]
        
        pypandoc.convert_file(
            md_path,
            'docx',
            outputfile=docx_path,
            extra_args=extra_args
        )
        print(f"✅ Markdown已成功转换为Word文件: {docx_path}")
        return True
    except Exception as e:
        print(f"❌ Markdown转换失败: {str(e)}")
        return False

def check_chinese_font(docx_path):
    """读取Word文件，检查中文字体是否为宋体"""
    try:
        doc = Document(docx_path)
        all_fonts = []
        chinese_text_fonts = []
        
        print("\n开始检查文档中的字体...")
        
        # 遍历所有段落
        for para_index, paragraph in enumerate(doc.paragraphs):
            if not paragraph.text.strip():
                continue
                
            print(f"\n段落 {para_index + 1}: {paragraph.text[:50]}...")
            
            # 遍历段落中的所有运行(runs)
            for run in paragraph.runs:
                # 检查是否包含中文字符
                contains_chinese = any('\u4e00' <= char <= '\u9fff' for char in run.text)
                font_name = run.font.name if run.font.name else "未设置"
                
                print(f"  运行文本: {run.text}")
                print(f"  字体: {font_name}")
                print(f"  包含中文: {contains_chinese}")
                
                all_fonts.append(font_name)
                if contains_chinese:
                    chinese_text_fonts.append(font_name)
        
        # 统计结果
        print("\n=== 字体检查结果 ===")
        print(f"文档中的唯一字体: {set(all_fonts)}")
        print(f"中文文本使用的字体: {set(chinese_text_fonts)}")
        
        # 检查是否所有中文都使用宋体
        # 注意：宋体在不同系统中可能有不同的名称表示
        simsun_variants = ['SimSun', '宋体', 'SimSun-ExtB']
        all_chinese_in_simsun = all(font in simsun_variants for font in chinese_text_fonts)
        
        if all_chinese_in_simsun:
            print("✅ 测试通过: 所有中文文本都使用了宋体")
            return True
        else:
            print("❌ 测试失败: 部分中文文本未使用宋体")
            return False
            
    except Exception as e:
        print(f"❌ 读取Word文件失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("=== Markdown转Word字体测试程序 ===")
    print("目标: 验证转换后的Word文档中文字体为宋体")
    
    # 创建临时文件路径
    md_path = create_test_markdown()
    docx_path = md_path.replace('.md', '.docx')
    
    try:
        # 1. 转换Markdown到Word
        if not convert_md_to_docx(md_path, docx_path):
            print("测试失败: Markdown转换过程出错")
            return
        
        # 2. 检查生成的Word文件中的字体
        success = check_chinese_font(docx_path)
        
        if success:
            print("\n🎉 总体测试通过: Markdown转Word时中文字体正确设置为宋体")
        else:
            print("\n❌ 总体测试失败: 请检查字体设置参数")
            
    finally:
        # 清理临时文件
        if os.path.exists(md_path):
            os.remove(md_path)
            print(f"\n清理临时文件: {md_path}")
        # 保留Word文件以便手动检查
        print(f"\n测试生成的Word文件: {docx_path}")

if __name__ == "__main__":
    # 检查必要的依赖
    try:
        import docx
        print("✅ python-docx库已安装")
    except ImportError:
        print("❌ python-docx库未安装，请运行: pip install python-docx")
        sys.exit(1)
    
    try:
        import pypandoc
        print("✅ pypandoc库已安装")
    except ImportError:
        print("❌ pypandoc库未安装，请运行: pip install pypandoc")
        sys.exit(1)
    
    main()
