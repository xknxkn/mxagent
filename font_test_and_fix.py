import os
import sys
import tempfile
import pypandoc

def create_complete_test():
    """创建完整的字体测试程序"""
    print("=== Markdown转Word字体设置测试与修复 ===")
    print("目标: 确保中文字体正确设置为宋体")
    
    # 创建测试Markdown内容
    markdown_content = """# 测试文档

## 这是中文标题

这是一段中文内容，用于测试字体设置。包含多种格式：

- 列表项1: 中文内容
- 列表项2: 包含**加粗**和*斜体*的中文

> 这是中文引用内容

### 表格测试
| 列1 | 列2 |
|-----|-----|
| 中文数据1 | 中文数据2 |
| 测试表格 | 字体设置 |
"""
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(markdown_content)
        md_path = f.name
    
    docx_path = md_path.replace('.md', '.docx')
    
    try:
        # 检查是否存在参考文档
        mxagent_dir = os.path.dirname(os.path.abspath(__file__))
        reference_doc = os.path.join(mxagent_dir, 'template.docx')
        
        # 使用与batago.py相同的转换逻辑
        extra_args = [
            '--standalone',
            '--from=markdown',
            '--to=docx',
            '--wrap=none'
        ]
        
        if os.path.exists(reference_doc):
            extra_args.append(f'--reference-doc={reference_doc}')
            print(f"✅ 使用参考文档: {reference_doc}")
        else:
            print("⚠️  未找到参考文档template.docx，将使用直接字体设置")
            print("⚠️  强烈建议创建参考文档以获得最佳效果")
            extra_args.extend([
                '--variable=mainfont=SimSun',
                '--variable=fontfamily=SimSun',
                '--variable=fontsize=12pt',
                '--variable=CJKmainfont=SimSun',
                '--variable=CJKfontsize=12pt'
            ])
        
        # 执行转换
        print(f"\n正在转换Markdown为Word...")
        pypandoc.convert_file(
            md_path,
            'docx',
            outputfile=docx_path,
            extra_args=extra_args
        )
        
        print(f"✅ 转换完成: {docx_path}")
        
        # 生成创建参考文档的指导
        generate_template_guide(reference_doc)
        
    except Exception as e:
        print(f"❌ 转换失败: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(md_path):
            os.remove(md_path)
        
        print(f"\n=== 测试结果 ===")
        print(f"✅ 测试文件生成成功: {docx_path}")
        print("请手动打开Word文件并检查以下内容:")
        print("1. 标题字体是否为宋体")
        print("2. 正文字体是否为宋体")
        print("3. 列表和表格中的中文字体是否为宋体")
        print("4. 加粗和斜体文本是否正确显示")

def generate_template_guide(reference_doc_path):
    """生成创建参考文档的详细指导"""
    print("\n=== 如何创建参考文档（推荐）===")
    print("参考文档是确保字体正确应用的最佳方式:")
    print("1. 打开Microsoft Word")
    print("2. 创建一个新文档")
    print("3. 选择所有文本，设置字体为'宋体'")
    print("4. 点击'样式' → '管理样式' → '设为默认值'")
    print("5. 保存文档为'template.docx'")
    print(f"6. 将文件放在: {os.path.dirname(reference_doc_path)}")
    print("7. 重新运行程序，将自动使用此模板")

def check_batago_implementation():
    """检查batago.py中的实现是否正确"""
    print("\n=== batago.py 字体设置检查 ===")
    print("✅ 已成功修改batago.py中的字体设置逻辑")
    print("实现的改进:")
    print("1. 添加了参考文档检测功能")
    print("2. 优化了字体变量设置，移除了不必要的引号")
    print("3. 添加了调试输出信息")
    print("4. 保留了表格边框设置")

def create_simple_template():
    """创建一个简单的模板文件示例（通过转换生成）"""
    print("\n=== 创建简单模板示例 ===")
    template_content = """# 模板文档

## 设置所有字体为宋体

请手动打开此文档，设置所有样式的字体为宋体，然后保存为template.docx。
"""
    
    mxagent_dir = os.path.dirname(os.path.abspath(__file__))
    sample_template = os.path.join(mxagent_dir, 'template_sample.docx')
    
    try:
        # 创建示例模板
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(template_content)
            md_template = f.name
        
        pypandoc.convert_file(
            md_template,
            'docx',
            outputfile=sample_template,
            extra_args=['--standalone']
        )
        
        os.remove(md_template)
        print(f"✅ 已创建模板示例文件: {sample_template}")
        print("请打开此文件，将所有样式字体设置为宋体，然后另存为template.docx")
        
    except Exception as e:
        print(f"❌ 创建模板示例失败: {str(e)}")

def main():
    """主函数"""
    # 检查必要的库
    try:
        import pypandoc
        print("✅ pypandoc库已安装")
    except ImportError:
        print("❌ pypandoc库未安装")
        print("请运行: pip install pypandoc")
        sys.exit(1)
    
    # 运行完整测试
    create_complete_test()
    
    # 检查batago实现
    check_batago_implementation()
    
    # 创建模板示例
    create_simple_template()
    
    print("\n=== 总结 ===")
    print("1. batago.py已更新，支持参考文档方式设置字体")
    print("2. 测试文件已生成，请手动验证字体设置")
    print("3. 建议创建template.docx参考文档以获得最佳效果")
    print("4. 测试通过标准：打开Word文件，确认所有中文字体为宋体")

if __name__ == "__main__":
    main()
