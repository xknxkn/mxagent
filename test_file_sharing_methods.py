#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试多种免费文件存储和分享链接的方法

这个脚本将测试几种常见的免费文件分享服务，找出可以正常工作的方案，并为batago.py提供一个可靠的文件分享实现。
"""

import os
import requests
import time
import json
import shutil
from datetime import datetime

# 配置
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'upload')
TEST_FILE_NAME = 'test_share_file.txt'
TEST_FILE_PATH = os.path.join(UPLOAD_DIR, TEST_FILE_NAME)

# 确保上传目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 创建测试文件
def create_test_file():
    """创建一个测试文件用于上传测试"""
    with open(TEST_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(f"这是一个测试文件，创建于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("用于测试多种文件分享服务的功能。\n")
        f.write("文件内容可以任意修改。\n")
    print(f"测试文件已创建: {TEST_FILE_PATH}")
    return TEST_FILE_PATH

# 测试1: anonfiles.com
def test_anonfiles(file_path):
    """测试anonfiles.com文件分享服务"""
    try:
        print("\n[1] 测试 anonfiles.com...")
        url = 'https://api.anonfiles.com/upload'
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=30)
            
        if response.status_code == 200:
            data = response.json()
            if data.get('status'):
                file_info = data.get('data', {}).get('file', {})
                share_link = file_info.get('url', {}).get('short')
                if share_link:
                    print(f"✅ anonfiles.com 成功! 分享链接: {share_link}")
                    return {'service': 'anonfiles', 'link': share_link, 'success': True}
                else:
                    print(f"❌ anonfiles.com 未获取到分享链接: {data}")
            else:
                error_msg = data.get('error', {}).get('message', '未知错误')
                print(f"❌ anonfiles.com 请求失败: {error_msg}")
        else:
            print(f"❌ anonfiles.com 请求错误: 状态码 {response.status_code}")
    except Exception as e:
        print(f"❌ anonfiles.com 异常: {str(e)}")
    return {'service': 'anonfiles', 'success': False, 'error': str(e) if 'e' in locals() else '未知错误'}

# 测试2: transfer.sh
def test_transfer_sh(file_path):
    """测试transfer.sh文件分享服务"""
    try:
        print("\n[2] 测试 transfer.sh...")
        file_name = os.path.basename(file_path)
        url = f'https://transfer.sh/{file_name}'
        
        with open(file_path, 'rb') as f:
            response = requests.put(url, data=f, timeout=30)
            
        if response.status_code == 200:
            share_link = response.text.strip()
            print(f"✅ transfer.sh 成功! 分享链接: {share_link}")
            return {'service': 'transfer.sh', 'link': share_link, 'success': True}
        else:
            print(f"❌ transfer.sh 请求失败: 状态码 {response.status_code}, 响应: {response.text}")
    except Exception as e:
        print(f"❌ transfer.sh 异常: {str(e)}")
    return {'service': 'transfer.sh', 'success': False, 'error': str(e) if 'e' in locals() else '未知错误'}

# 测试3: catbox.moe
def test_catbox(file_path):
    """测试catbox.moe文件分享服务"""
    try:
        print("\n[3] 测试 catbox.moe...")
        url = 'https://catbox.moe/user/api.php'
        
        with open(file_path, 'rb') as f:
            files = {'fileToUpload': f}
            data = {'reqtype': 'fileupload'}
            response = requests.post(url, files=files, data=data, timeout=30)
            
        if response.status_code == 200:
            share_link = response.text.strip()
            if share_link.startswith('https://files.catbox.moe/'):
                print(f"✅ catbox.moe 成功! 分享链接: {share_link}")
                return {'service': 'catbox.moe', 'link': share_link, 'success': True}
            else:
                print(f"❌ catbox.moe 返回非链接内容: {share_link}")
        else:
            print(f"❌ catbox.moe 请求失败: 状态码 {response.status_code}")
    except Exception as e:
        print(f"❌ catbox.moe 异常: {str(e)}")
    return {'service': 'catbox.moe', 'success': False, 'error': str(e) if 'e' in locals() else '未知错误'}

# 测试4: file.io
def test_file_io(file_path):
    """测试file.io文件分享服务"""
    try:
        print("\n[4] 测试 file.io...")
        url = 'https://file.io'
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=30)
            
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                share_link = data.get('link')
                print(f"✅ file.io 成功! 分享链接: {share_link}")
                return {'service': 'file.io', 'link': share_link, 'success': True}
            else:
                print(f"❌ file.io 请求失败: {data}")
        else:
            print(f"❌ file.io 请求错误: 状态码 {response.status_code}")
    except Exception as e:
        print(f"❌ file.io 异常: {str(e)}")
    return {'service': 'file.io', 'success': False, 'error': str(e) if 'e' in locals() else '未知错误'}

# 测试5: temporary-url.com
def test_temporary_url(file_path):
    """测试temporary-url.com文件分享服务"""
    try:
        print("\n[5] 测试 temporary-url.com...")
        url = 'https://temporary-url.com/upload'
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=30, allow_redirects=False)
            
        if response.status_code == 302:
            # 获取重定向后的URL
            location = response.headers.get('Location')
            # 提取文件ID
            import re
            match = re.search(r'\?id=(\w+)', location)
            if match:
                file_id = match.group(1)
                share_link = f"https://temporary-url.com/file/{file_id}"
                print(f"✅ temporary-url.com 成功! 分享链接: {share_link}")
                return {'service': 'temporary-url.com', 'link': share_link, 'success': True}
            else:
                print(f"❌ temporary-url.com 无法提取文件ID")
        else:
            print(f"❌ temporary-url.com 请求错误: 状态码 {response.status_code}")
    except Exception as e:
        print(f"❌ temporary-url.com 异常: {str(e)}")
    return {'service': 'temporary-url.com', 'success': False, 'error': str(e) if 'e' in locals() else '未知错误'}

# 主函数
def main():
    """主测试函数"""
    print("="*80)
    print("免费文件存储和分享链接服务测试")
    print("="*80)
    
    # 创建测试文件
    test_file = create_test_file()
    
    # 运行所有测试
    results = []
    
    # 运行每个测试并添加延迟以避免触发API限制
    tests = [
        test_anonfiles,
        test_transfer_sh,
        test_catbox,
        test_file_io,
        test_temporary_url
    ]
    
    for test_func in tests:
        result = test_func(test_file)
        results.append(result)
        # 添加间隔以避免过快的请求
        time.sleep(2)
    
    # 汇总结果
    print("\n" + "="*80)
    print("测试结果汇总:")
    print("="*80)
    
    successful_services = []
    failed_services = []
    
    for result in results:
        if result['success']:
            successful_services.append(result)
            print(f"✅ {result['service']}: {result['link']}")
        else:
            failed_services.append(result)
            print(f"❌ {result['service']}: {result.get('error', '未知错误')}")
    
    # 生成推荐方案
    print("\n" + "="*80)
    if successful_services:
        print(f"成功的服务数量: {len(successful_services)}")
        print("推荐的文件分享方案:")
        for i, service in enumerate(successful_services, 1):
            print(f"  {i}. {service['service']} - {service['link']}")
        
        # 选择第一个成功的服务作为首选方案
        primary_service = successful_services[0]
        print("\n首选方案代码示例:")
        
        # 生成相应的Python代码示例
        if primary_service['service'] == 'anonfiles':
            code_example = """
def create_share_link(filename):
    # 使用anonfiles.com创建分享链接
    try:
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'upload')
        file_path = os.path.join(upload_dir, filename)
        
        if not os.path.exists(file_path):
            return f"文件分享失败：{filename} 不存在"
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始使用anonfiles.com上传文件: {filename}")
        url = 'https://api.anonfiles.com/upload'
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=30)
            
        if response.status_code == 200:
            data = response.json()
            if data.get('status'):
                file_info = data.get('data', {}).get('file', {})
                share_link = file_info.get('url', {}).get('short')
                if share_link:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 文件上传成功，分享链接: {share_link}")
                    return share_link
                else:
                    return "获取分享链接失败"
            else:
                return f"上传失败: {data.get('error', {}).get('message', '未知错误')}"
        else:
            return f"请求失败: 状态码 {response.status_code}"
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 文件分享过程出错: {str(e)}")
        return f"文件分享失败：{str(e)}"
"""
        elif primary_service['service'] == 'transfer.sh':
            code_example = """
def create_share_link(filename):
    # 使用transfer.sh创建分享链接
    try:
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'upload')
        file_path = os.path.join(upload_dir, filename)
        
        if not os.path.exists(file_path):
            return f"文件分享失败：{filename} 不存在"
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始使用transfer.sh上传文件: {filename}")
        url = f'https://transfer.sh/{filename}'
        
        with open(file_path, 'rb') as f:
            response = requests.put(url, data=f, timeout=30)
            
        if response.status_code == 200:
            share_link = response.text.strip()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 文件上传成功，分享链接: {share_link}")
            return share_link
        else:
            return f"上传失败: 状态码 {response.status_code}"
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 文件分享过程出错: {str(e)}")
        return f"文件分享失败：{str(e)}"
"""
        elif primary_service['service'] == 'catbox.moe':
            code_example = """
def create_share_link(filename):
    # 使用catbox.moe创建分享链接
    try:
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'upload')
        file_path = os.path.join(upload_dir, filename)
        
        if not os.path.exists(file_path):
            return f"文件分享失败：{filename} 不存在"
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始使用catbox.moe上传文件: {filename}")
        url = 'https://catbox.moe/user/api.php'
        
        with open(file_path, 'rb') as f:
            files = {'fileToUpload': f}
            data = {'reqtype': 'fileupload'}
            response = requests.post(url, files=files, data=data, timeout=30)
            
        if response.status_code == 200:
            share_link = response.text.strip()
            if share_link.startswith('https://files.catbox.moe/'):
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 文件上传成功，分享链接: {share_link}")
                return share_link
            else:
                return f"获取分享链接失败: {share_link}"
        else:
            return f"上传失败: 状态码 {response.status_code}"
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 文件分享过程出错: {str(e)}")
        return f"文件分享失败：{str(e)}"
"""
        elif primary_service['service'] == 'file.io':
            code_example = """
def create_share_link(filename):
    # 使用file.io创建分享链接
    try:
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'upload')
        file_path = os.path.join(upload_dir, filename)
        
        if not os.path.exists(file_path):
            return f"文件分享失败：{filename} 不存在"
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始使用file.io上传文件: {filename}")
        url = 'https://file.io'
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=30)
            
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                share_link = data.get('link')
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 文件上传成功，分享链接: {share_link}")
                return share_link
            else:
                return f"上传失败: {data}"
        else:
            return f"请求失败: 状态码 {response.status_code}"
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 文件分享过程出错: {str(e)}")
        return f"文件分享失败：{str(e)}"
"""
        else:
            code_example = "# 请根据测试结果手动实现首选服务的代码"
        
        print(code_example)
    else:
        print("所有测试服务均失败! 建议:)")
        print("1. 检查网络连接")
        print("2. 考虑使用本地HTTP服务器作为备选方案")
        print("3. 尝试其他文件分享服务")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
