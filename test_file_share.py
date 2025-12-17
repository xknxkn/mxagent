#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试文件分享功能
验证文件上传和分享链接生成是否正常工作
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# 添加当前目录到Python路径，确保可以导入batago模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入batago模块中的相关函数
from batago import upload_file, create_share_link, initialize_file_sharing

def test_file_sharing():
    """测试文件分享功能的完整流程"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始测试文件分享功能...")
    
    try:
        # 1. 初始化文件分享服务器
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 初始化文件分享服务器...")
        server_status = initialize_file_sharing()
        if server_status:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 文件分享服务器初始化成功")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 文件分享服务器初始化失败！")
            return False
        
        # 2. 创建临时测试文件
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 创建测试文件...")
        test_filename = "test_share_file.txt"
        test_content = "这是测试文件分享功能的内容。\nFile sharing test content."
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(test_content.encode('utf-8'))
            temp_file_path = temp_file.name
        
        # 重命名临时文件为我们想要的测试文件名
        final_test_path = os.path.join(os.path.dirname(temp_file_path), test_filename)
        os.rename(temp_file_path, final_test_path)
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 测试文件创建成功: {final_test_path}")
        
        # 3. 测试直接使用create_share_link函数（文件尚未上传）
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 测试create_share_link函数（文件未上传状态）...")
        share_link = create_share_link(test_filename)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] create_share_link结果: {share_link}")
        
        # 应该返回错误信息，因为文件尚未上传到upload目录
        if "不存在" in share_link:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ 预期行为：文件不存在时返回错误信息")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ 非预期行为：文件不存在时未返回正确的错误信息")
        
        # 4. 上传文件并获取分享链接
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 上传文件并获取分享链接...")
        
        # 定义一个简单的进度回调函数，替代gr.Progress()
        def simple_progress(value, desc=""):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 上传进度: {value:.1%} - {desc}")
        
        # 上传文件
        result = upload_file(final_test_path, progress=simple_progress)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] upload_file结果: {result}")
        
        # 5. 验证上传结果
        success = "文件上传成功" in result
        has_link = "文件分享链接" in result
        
        if success:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ 文件上传成功")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ 文件上传失败")
        
        if has_link:
            # 提取分享链接
            for line in result.split('\n'):
                if line.startswith("文件分享链接："):
                    share_link = line.replace("文件分享链接：", "").strip()
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ 分享链接生成成功: {share_link}")
                    break
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ 分享链接生成失败")
        
        # 6. 验证文件是否存在于upload目录
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'upload')
        uploaded_file_path = os.path.join(upload_dir, test_filename)
        if os.path.exists(uploaded_file_path):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ 文件成功上传到: {uploaded_file_path}")
            # 检查文件内容
            with open(uploaded_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content == test_content:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ 文件内容验证成功")
                else:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ 文件内容验证失败")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ 上传文件不存在: {uploaded_file_path}")
        
        # 7. 再次测试create_share_link（文件已上传）
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 再次测试create_share_link函数（文件已上传状态）...")
        share_link = create_share_link(test_filename)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] create_share_link结果: {share_link}")
        
        if share_link.startswith("http"):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ 分享链接生成成功: {share_link}")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ 分享链接生成失败: {share_link}")
        
        # 8. 清理测试文件
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 清理测试文件...")
        # 删除源测试文件
        if os.path.exists(final_test_path):
            os.remove(final_test_path)
        
        # 保留上传目录中的测试文件，以便用户可以手动验证
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 测试文件保留在upload目录中，可通过分享链接访问")
        
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 文件分享功能测试完成!")
        print(f"\n测试总结:")
        print(f"✅ 文件上传功能: {'成功' if success else '失败'}")
        print(f"✅ 分享链接生成: {'成功' if has_link else '失败'}")
        print(f"✅ 文件分享服务器: {'运行中' if server_status else '未运行'}")
        
        return success and has_link
        
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时文件
        if 'final_test_path' in locals() and os.path.exists(final_test_path):
            try:
                os.remove(final_test_path)
            except:
                pass

if __name__ == "__main__":
    success = test_file_sharing()
    print(f"\n测试 {'通过' if success else '失败'}")
    sys.exit(0 if success else 1)
