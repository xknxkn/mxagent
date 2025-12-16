#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试脚本：直接验证Pandoc UTF-8编码转换
"""
import os
import sys
import tempfile
import datetime

def test_pandoc_utf8():
    """测试Pandoc的UTF-8编码转换功能"""
    print("开始测试Pandoc UTF-8编码转换...")
    print(f"当前Python版本: {sys.version}")
    print(f"系统编码: {sys.stdout.encoding}")
    
    try:
        # 确保pypandoc已安装
        import pypandoc
        print(f"pypandoc版本: {pypandoc.get_pandoc_version()}")
        
        # 设置UTF-8环境变量
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # 创建临时文件目录
        temp_dir = "summary_plans"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # 生成文件名
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_md_path = os.path.join(temp_dir, f"test_{timestamp}.md")
        temp_docx_path = os.path.join(temp_dir, f"test_{timestamp}.docx")
        
        # 创建包含中文的测试内容
        test_content = """# 测试文档
## 中文UTF-8编码测试

这是一个包含**中文**、**特殊字符**和**格式**的测试文档。

### 测试列表
1. 第一项
2. 第二项
3. 第三项

### 测试表格
| 列1 | 列2 | 列3 |
|-----|-----|-----|
| 中文 | English | 数字123 |
| 测试 | Test | 测试数据 |

这是一段包含特殊符号的文本：!@#$%^&*()_+-=[]{}|;':",.<>/?\
"""
        
        print(f"\n创建临时Markdown文件: {temp_md_path}")
        # 写入临时Markdown文件
        with open(temp_md_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        print("✓ Markdown文件创建成功")
        
        # 设置pandoc参数
        extra_args = [
            '--standalone',
            '--from=markdown+smart',
            '--to=docx',
            '--wrap=none',
            '--metadata=title=Pandoc UTF-8测试文档',
            '--markdown-headings=atx'
        ]
        
        print(f"\n尝试转换为Word文档: {temp_docx_path}")
        print(f"使用参数: {extra_args}")
        
        # 执行转换
        pypandoc.convert_file(
            temp_md_path,
            'docx',
            outputfile=temp_docx_path,
            extra_args=extra_args
        )
        
        # 检查结果
        if os.path.exists(temp_docx_path) and os.path.getsize(temp_docx_path) > 0:
            file_size = os.path.getsize(temp_docx_path) / 1024
            print(f"✓ Word文档转换成功!")
            print(f"  文件大小: {file_size:.2f} KB")
            print(f"  文件路径: {temp_docx_path}")
            return True
        else:
            print(f"✗ Word文档转换失败: 文件未创建或为空")
            return False
            
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请安装pypandoc: pip install pypandoc")
        return False
    except Exception as e:
        print(f"转换过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时文件
        if 'temp_md_path' in locals() and os.path.exists(temp_md_path):
            try:
                os.remove(temp_md_path)
                print(f"清理临时Markdown文件: {temp_md_path}")
            except:
                pass

if __name__ == "__main__":
    success = test_pandoc_utf8()
    print(f"\n测试{'成功' if success else '失败'}")
    sys.exit(0 if success else 1)
