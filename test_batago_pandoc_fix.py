#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试batago.py中Pandoc转换修复后的功能
验证UTF-8编码问题是否已解决
"""

import os
import sys
import subprocess
import pypandoc
import datetime

# 设置输出编码为UTF-8
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

def check_pandoc_installation():
    """检查Pandoc是否正确安装"""
    print("检查Pandoc安装情况...")
    try:
        result = subprocess.run(['pandoc', '--version'], capture_output=True, text=True, check=True)
        print(f"Pandoc版本: {result.stdout.split('\n')[0]}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Pandoc未安装或不可用: {e}")
        return False

def test_pandoc_conversion():
    """测试Pandoc转换功能，使用修复后的参数设置"""
    # 创建测试目录
    test_dir = "test_batago_pandoc"
    os.makedirs(test_dir, exist_ok=True)
    
    # 测试内容，包含中文字符
    test_markdown = """
# 倍塔狗人工智能测试文档

## 测试基本信息

**学生姓名:** 测试学生  
**测试时间:** "+datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')+"  
**测试内容:** UTF-8编码转换测试

---

### 测试段落
这是一个包含中文的测试段落，用于验证Pandoc转换功能是否正常工作。
包含特殊字符：！@#￥%……&*（）——+  
以及标点符号：，。；：？！

### 测试列表
- 第一项测试内容
- 第二项测试内容
- 第三项测试内容

### 测试代码块
```python
print("这是中文测试代码")
for i in range(10):
    print(f"测试数字: {i}")
```
"""
    
    # 写入测试Markdown文件
    md_file_path = os.path.join(test_dir, "test_chinese.md")
    with open(md_file_path, 'w', encoding='utf-8') as f:
        f.write(test_markdown)
    print(f"已创建测试Markdown文件: {md_file_path}")
    
    # 生成Word文件路径
    docx_file_path = os.path.join(test_dir, "test_chinese.docx")
    
    # 使用修复后的参数进行转换
    print("开始转换Markdown到Word...")
    try:
        # 使用与修复后相同的参数设置
        extra_args = [
            '--standalone',
            '--from=markdown+smart',  # 修复后的参数
            '--to=docx',
            '--wrap=none'
        ]
        
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
            return True
        else:
            print(f"✗ 转换失败: 文件未生成或为空")
            return False
    except Exception as e:
        print(f"✗ 转换过程中出错: {str(e)}")
        return False

def main():
    """主函数"""
    print("===== Batago Pandoc UTF-8修复测试 =====")
    print(f"Python版本: {sys.version}")
    print(f"pypandoc版本: {pypandoc.__version__}")
    print(f"Python默认编码: {sys.getdefaultencoding()}")
    print(f"文件系统编码: {sys.getfilesystemencoding()}")
    print()
    
    # 检查Pandoc安装
    if not check_pandoc_installation():
        print("请先安装Pandoc再运行测试")
        print("Windows用户可以从 https://github.com/jgm/pandoc/releases 下载安装程序")
        return False
    
    print()
    
    # 测试转换功能
    conversion_success = test_pandoc_conversion()
    
    print()
    print("===== 测试结果 ====")
    if conversion_success:
        print("✅ Pandoc UTF-8编码修复验证成功!")
        print("✅ 使用markdown+smart参数可以成功转换包含中文的文档")
        return True
    else:
        print("❌ Pandoc UTF-8编码修复验证失败!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
