#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试HTTP文件下载功能

这个脚本用于测试通过HTTP链接下载已上传到服务器的文件
"""

import os
import sys
import time
import requests
from batago import create_share_link

def create_test_file():
    """
    创建一个测试文件用于上传和下载测试
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_content = f"这是一个用于测试HTTP下载功能的测试文件。\n"
    test_content += f"创建时间: {timestamp}\n"
    test_content += "\n这个文件将被上传到服务器，然后通过HTTP链接下载回来进行验证。\n"
    test_content += "如果下载成功，说明HTTP文件分享功能工作正常。"
    
    test_filename = f"test_http_download_{timestamp}.txt"
    
    # 创建测试文件
    with open(test_filename, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"创建测试文件: {os.path.abspath(test_filename)}")
    print(f"文件大小: {os.path.getsize(test_filename)} 字节")
    return test_filename, test_content

def test_http_link_sharing():
    """
    测试HTTP链接文件分享功能
    """
    # 创建测试文件
    test_file, test_content = create_test_file()
    
    try:
        # 测试文件分享
        print("\n开始测试HTTP文件分享功能...")
        start_time = time.time()
        
        # 调用分享函数获取HTTP链接
        share_result = create_share_link(test_file)
        
        end_time = time.time()
        
        # 打印结果
        print(f"\n分享结果: {share_result}")
        print(f"分享耗时: {end_time - start_time:.2f} 秒")
        
        # 检查是否返回了HTTP链接
        if share_result.startswith("http://"):
            print("\n测试结果: 成功 - 成功创建了HTTP下载链接")
            http_link = share_result
            
            # 尝试通过HTTP链接下载文件进行验证
            print(f"\n正在通过HTTP链接下载文件: {http_link}")
            try:
                download_start_time = time.time()
                response = requests.get(http_link, timeout=30)
                download_end_time = time.time()
                
                if response.status_code == 200:
                    print(f"下载成功！HTTP状态码: {response.status_code}")
                    print(f"下载耗时: {download_end_time - download_start_time:.2f} 秒")
                    print(f"下载的文件大小: {len(response.content)} 字节")
                    
                    # 验证下载的内容是否与原始内容匹配
                    downloaded_content = response.content.decode('utf-8')
                    if downloaded_content == test_content:
                        print("✅ 内容验证通过！下载的文件内容与原始文件完全一致")
                    else:
                        print("❌ 内容验证失败！下载的文件内容与原始文件不匹配")
                        print("原始文件长度:", len(test_content))
                        print("下载文件长度:", len(downloaded_content))
                        
                else:
                    print(f"❌ 下载失败！HTTP状态码: {response.status_code}")
                    print(f"响应内容: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ HTTP下载请求出错: {str(e)}")
                print("\n可能的原因:")
                print("1. 服务器未正确配置sharefile目录的访问权限")
                print("2. Apache服务器未正确映射/opt/redmine-3.0.1-0/apache2/htdocs目录")
                print("3. 网络连接问题")
                print("4. 文件未成功上传到服务器")
                
            # 提供使用说明
            print("\n使用方法:")
            print(f"1. 在浏览器中打开以下链接直接下载:")
            print(f"   {http_link}")
            print(f"2. 或者将链接分享给需要下载文件的用户")
            
        else:
            print(f"\n测试结果: 失败 - {share_result}")
            print("\n请检查:")
            print("1. SCP上传是否正常工作")
            print("2. 服务器配置是否正确")
            print("3. 网络连接是否正常")
            
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n清理测试文件: {test_file}")

def test_existing_file():
    """
    测试上传目录中已存在的文件的HTTP下载链接
    """
    # 检查upload目录中是否有文件
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'upload')
    if os.path.exists(upload_dir):
        files_in_upload = os.listdir(upload_dir)
        if files_in_upload:
            # 使用第一个找到的文件进行测试
            test_file = os.path.join(upload_dir, files_in_upload[0])
            print(f"\n在upload目录中找到文件: {test_file}")
            
            # 复制文件到当前目录进行测试
            temp_test_file = os.path.basename(test_file)
            import shutil
            shutil.copy2(test_file, temp_test_file)
            
            try:
                # 调用分享函数获取HTTP链接
                print(f"\n测试已存在文件的HTTP分享: {temp_test_file}")
                share_result = create_share_link(temp_test_file)
                print(f"分享结果: {share_result}")
            finally:
                # 清理临时复制的文件
                if os.path.exists(temp_test_file):
                    os.remove(temp_test_file)
                    print(f"清理临时文件: {temp_test_file}")

if __name__ == "__main__":
    print("=== HTTP文件下载功能测试 ===")
    print("服务器信息：")
    print("- 地址: 121.40.182.30")
    print("- HTTP端口: 8000")
    print("- 文件根目录: /opt/redmine-3.0.1-0/apache2/htdocs")
    print("- 分享文件目录: /opt/redmine-3.0.1-0/apache2/htdocs/sharefile")
    print("- HTTP访问路径: http://121.40.182.30:8000/sharefile/")
    print("")
    
    # 测试新文件的HTTP分享
    test_http_link_sharing()
    
    # 测试已存在文件的HTTP分享
    test_existing_file()
    
    print("\n=== 测试完成 ===")
