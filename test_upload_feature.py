#!/usr/bin/env python3
# coding: utf-8
"""
测试上传功能的脚本
"""

import os
import sys
import time

# 确保脚本可以找到batago模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_upload_folder_creation():
    """测试upload文件夹是否被正确创建"""
    # 导入batago模块以触发初始化
    print("正在导入batago模块以触发初始化...")
    try:
        import batago
        print("batago模块导入成功")
        
        # 检查upload文件夹是否存在
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'upload')
        if os.path.exists(upload_dir):
            print(f"✅ 测试通过：upload文件夹已成功创建在: {upload_dir}")
            # 列出文件夹内容
            try:
                files = os.listdir(upload_dir)
                if files:
                    print(f"  文件夹中当前有{len(files)}个文件：")
                    for file in files:
                        print(f"    - {file}")
                else:
                    print("  文件夹当前为空")
            except Exception as e:
                print(f"  读取文件夹内容时出错: {e}")
            return True
        else:
            print(f"❌ 测试失败：upload文件夹不存在于: {upload_dir}")
            return False
            
    except ImportError as e:
        print(f"❌ 导入batago模块失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 初始化过程中发生错误: {e}")
        return False

def test_file_upload_function():
    """测试文件上传函数是否可用"""
    try:
        from batago import upload_file
        print("✅ 测试通过：upload_file函数已成功导入")
        print("  注意：完整的文件上传功能需要在Gradio界面中测试")
        print("  建议启动batago.py并通过界面测试文件上传")
        return True
    except AttributeError:
        print("❌ 测试失败：upload_file函数未找到")
        return False
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        return False

def main():
    """运行所有测试"""
    print("=== upload功能测试 ===")
    print()
    
    # 运行文件夹创建测试
    folder_test_result = test_upload_folder_creation()
    print()
    
    # 运行文件上传函数测试
    upload_func_test_result = test_file_upload_function()
    print()
    
    # 总结
    print("=== 测试总结 ===")
    if folder_test_result and upload_func_test_result:
        print("✅ 所有功能测试通过！")
        print("\n使用说明：")
        print("1. 运行 batago.py 启动应用")
        print("2. 在Gradio界面中，您可以看到新的文件上传区域")
        print("3. 选择一个或多个文件，点击'开始上传'按钮")
        print("4. 上传成功后，文件将保存在upload文件夹中")
        print("5. 如果上传同名文件，系统会自动添加时间戳避免覆盖")
    else:
        print("❌ 部分测试失败，请检查相关代码")
    
    # 提示如何启动应用
    print("\n要启动应用进行完整测试，请运行:")
    print(f"  python {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'batago.py')}")

if __name__ == "__main__":
    main()
