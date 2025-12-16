#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试A4纸张格式的Word文档生成
验证我们修改的代码是否正确应用A4纸张格式
"""

import os
import sys
import subprocess
import pypandoc
import datetime

# 设置输出编码为UTF-8
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

def test_a4_paper_format():
    """测试A4纸张格式的Word文档生成"""
    print("开始测试A4纸张格式的Word文档生成...")
    
    # 创建测试目录
    test_dir = "test_a4_format"
    os.makedirs(test_dir, exist_ok=True)
    
    # 测试内容，包含中文字符
    test_markdown = """
# A4纸张格式测试文档

## 基本信息

**测试时间:** {}
**测试内容:** A4纸张格式验证

---

### 测试段落
这是一个测试文档，用于验证生成的Word文档是否使用A4纸张格式。

### 测试列表
- A4纸张尺寸: 210mm × 297mm
- 这应该是标准的A4纸张设置
- 文档应该能够正确显示所有内容

### 测试表格
| 列1 | 列2 | 列3 |
|-----|-----|-----|
| 测试 | 数据 | 内容 |
| 中文 | 支持 | 测试 |
""".format(datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S'))
    
    # 写入测试Markdown文件
    md_file_path = os.path.join(test_dir, "test_a4.md")
    with open(md_file_path, 'w', encoding='utf-8') as f:
        f.write(test_markdown)
    print(f"已创建测试Markdown文件: {md_file_path}")
    
    # 生成Word文件路径
    docx_file_path = os.path.join(test_dir, "test_a4.docx")
    
    # 使用与修改后相同的参数设置
    print("开始转换Markdown到Word (A4纸张格式)...")
    try:
        # 设置基本参数
        extra_args = [
            '--standalone',
            '--from=markdown+smart',
            '--to=docx',
            '--wrap=none',
            '--metadata=title=A4纸张格式测试'
        ]
        
        # 检查是否存在template.docx文件作为参考文档（确保A4纸张格式）
        template_path = os.path.join(os.path.dirname(__file__), 'template.docx')
        if os.path.exists(template_path):
            extra_args.append(f'--reference-doc={template_path}')
            print(f"使用参考文档确保A4纸张格式: {template_path}")
        else:
            print("警告: 未找到template.docx文件，将使用默认设置")
            # 如果没有template.docx，可以尝试设置变量（虽然对docx格式可能不起作用）
            extra_args.append('--variable=papersize=a4')
            print("尝试使用--variable=papersize=a4参数")
        
        # 执行转换
        pypandoc.convert_file(
            md_file_path,
            'docx',
            outputfile=docx_file_path,
            extra_args=extra_args
        )
        
        # 验证文件是否生成成功
        if os.path.exists(docx_file_path) and os.path.getsize(docx_file_path) > 0:
            file_size = os.path.getsize(docx_file_path) / 1024
            print(f"✓ 转换成功! Word文件已生成: {docx_file_path}")
            print(f"  文件大小: {file_size:.2f} KB")
            print("  请在Microsoft Word中打开并验证纸张格式是否为A4")
            return True
        else:
            print(f"✗ 转换失败: 文件未生成或为空")
            return False
    except Exception as e:
        print(f"✗ 转换过程中出错: {str(e)}")
        return False

if __name__ == "__main__":
    # 检查pandoc是否安装
    print("检查Pandoc是否可用...")
    try:
        result = subprocess.run(['pandoc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        if result.returncode == 0:
            print(f"✓ Pandoc检查成功: {result.stdout.splitlines()[0]}")
        else:
            print(f"✗ Pandoc命令返回错误码: {result.returncode}")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"✗ 检测Pandoc时出错: {str(e)}")
    
    # 运行测试
    success = test_a4_paper_format()
    
    if success:
        print("\n测试完成! 请在Word中打开生成的文档并验证纸张格式设置。")
    else:
        print("\n测试失败! 请检查错误信息并修复问题。")
        sys.exit(1)
