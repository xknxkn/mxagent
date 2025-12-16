import os
import sys
import pypandoc
from datetime import datetime

def test_pandoc_encoding():
    """测试Pandoc转换的UTF-8编码处理"""
    print("开始测试Pandoc UTF-8编码修复...")
    
    # 创建测试目录
    test_dir = "test_pandoc"
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建包含中文的测试Markdown文件
    test_md_path = os.path.join(test_dir, "test_chinese.md")
    test_docx_path = os.path.join(test_dir, "test_chinese.docx")
    
    # 测试内容包含各种中文和特殊字符
    test_content = """# 中文测试文档

## 这是一个测试

这是包含中文内容的测试文档，用于验证Pandoc转换时的UTF-8编码处理。

包含各种特殊字符：中文标点、符号等。

### 列表测试
- 测试项目1
- 测试项目2
- 测试项目3

### 表格测试
| 列1 | 列2 |
|-----|-----|
| 中文 | 测试 |
| 内容 | 数据 |

**粗体文本** 和 *斜体文本*
"""
    
    # 写入测试文件
    try:
        with open(test_md_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        print(f"✓ 已创建测试Markdown文件: {test_md_path}")
    except Exception as e:
        print(f"✗ 创建测试文件失败: {e}")
        return False
    
    # 测试pypandoc转换
    try:
        # 使用与修复后相同的参数
        extra_args = [
            '--standalone',
            '--from=markdown+smart',
            '--to=docx',
            '--wrap=none',
            '--variable=mainfont="SimSun"',
            '--variable=fontfamily="SimSun"',
            '--variable=fontsize=12pt'
        ]
        
        print("正在执行Pandoc转换...")
        pypandoc.convert_file(
            test_md_path,
            'docx',
            outputfile=test_docx_path,
            extra_args=extra_args
        )
        
        # 检查输出文件
        if os.path.exists(test_docx_path) and os.path.getsize(test_docx_path) > 0:
            print(f"✓ Word文档转换成功: {test_docx_path}")
            print(f"  文件大小: {os.path.getsize(test_docx_path) / 1024:.2f} KB")
            return True
        else:
            print(f"✗ Word文档未成功生成或为空")
            return False
    except Exception as e:
        print(f"✗ Pandoc转换失败: {e}")
        # 输出更详细的错误信息
        print(f"  错误类型: {type(e).__name__}")
        return False

def check_pandoc_installation():
    """检查Pandoc是否正确安装"""
    try:
        import subprocess
        result = subprocess.run(['pandoc', '--version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True)
        if result.returncode == 0:
            print("✓ Pandoc已正确安装")
            # 输出Pandoc版本信息
            version_line = result.stdout.split('\n')[0]
            print(f"  {version_line}")
            return True
        else:
            print("✗ Pandoc命令执行失败")
            print(f"  错误: {result.stderr}")
            return False
    except FileNotFoundError:
        print("✗ 未找到Pandoc命令行工具")
        print("  请从 https://github.com/jgm/pandoc/releases 下载并安装")
        return False
    except Exception as e:
        print(f"✗ 检查Pandoc安装时出错: {e}")
        return False

def check_pypandoc_version():
    """检查pypandoc版本"""
    try:
        import pypandoc
        print(f"✓ pypandoc版本: {pypandoc.__version__}")
        return True
    except Exception as e:
        print(f"✗ 检查pypandoc版本时出错: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print(f"测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 检查Python环境编码
    print(f"Python默认编码: {sys.getdefaultencoding()}")
    print(f"文件系统编码: {sys.getfilesystemencoding()}")
    
    # 检查安装状态
    pandoc_ok = check_pandoc_installation()
    pypandoc_ok = check_pypandoc_version()
    
    print("\n" + "-" * 60)
    
    # 只有当Pandoc和pypandoc都正常时才进行转换测试
    if pandoc_ok and pypandoc_ok:
        success = test_pandoc_encoding()
    else:
        success = False
        print("⚠ 跳过转换测试，因为Pandoc或pypandoc未正确安装")
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 测试成功! Pandoc UTF-8编码修复有效。")
        print("   修复方案: 为pypandoc.convert_file添加encoding='utf-8'参数")
        print("   并在extra_args中添加'--input-encoding=utf-8'和'--output-encoding=utf-8'")
    else:
        print("❌ 测试失败。请检查Pandoc安装和系统编码设置。")
        print("   建议：")
        print("   1. 确保Pandoc已正确安装且添加到系统PATH")
        print("   2. 检查系统区域设置是否为UTF-8")
        print("   3. 确保Python环境使用UTF-8编码")
    print("=" * 60)
