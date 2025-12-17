#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试SCP上传和HTTP下载功能

这个脚本独立于batago.py，直接使用paramiko和requests库来测试：
1. SCP上传文件到/opt/redmine-3.0.1-0/apache2/htdocs/sharefile/summary_plans目录
2. 通过HTTP链接下载上传的文件
"""

import os
import sys
import time
import paramiko
import requests
from scp import SCPClient

def create_test_file():
    """
    创建一个简单的测试文件
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_content = f"这是一个用于测试的文件。\n"
    test_content += f"创建时间: {timestamp}\n"
    test_content += "\n此文件将通过SCP直接上传到服务器的summary_plans目录，\n"
    test_content += "然后通过HTTP链接下载进行验证。"
    
    test_filename = f"test_scp_http_{timestamp}.txt"
    
    # 创建测试文件
    with open(test_filename, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"创建测试文件: {os.path.abspath(test_filename)}")
    print(f"文件内容: {test_content}")
    return test_filename, test_content

def connect_to_server():
    """
    连接到SSH服务器
    """
    try:
        # 创建SSH客户端
        ssh = paramiko.SSHClient()
        # 自动添加主机密钥
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        print("\n正在连接到SSH服务器...")
        # 连接服务器
        ssh.connect(
            hostname='121.40.182.30',
            port=22,
            username='batago',
            password='4008737505'
        )
        
        print("SSH连接成功!")
        return ssh
    except Exception as e:
        print(f"SSH连接失败: {str(e)}")
        return None

def create_remote_directory(ssh, directory):
    """
    在远程服务器上创建目录
    """
    try:
        print(f"\n正在创建远程目录: {directory}")
        # 使用shell命令创建目录（包括父目录）
        command = f"mkdir -p {directory}"
        stdin, stdout, stderr = ssh.exec_command(command)
        
        # 检查命令执行结果
        exit_code = stdout.channel.recv_exit_status()
        if exit_code == 0:
            print(f"远程目录创建成功: {directory}")
            return True
        else:
            error_msg = stderr.read().decode('utf-8')
            print(f"远程目录创建失败 (错误码: {exit_code}): {error_msg}")
            return False
    except Exception as e:
        print(f"创建远程目录时出错: {str(e)}")
        return False

def upload_file(ssh, local_file, remote_dir):
    """
    通过SCP上传文件
    """
    try:
        remote_filename = os.path.basename(local_file)
        remote_path = os.path.join(remote_dir, remote_filename).replace('\\', '/')
        
        print(f"\n正在通过SCP上传文件...")
        print(f"本地文件: {local_file}")
        print(f"远程路径: {remote_path}")
        
        # 创建SCP客户端
        with SCPClient(ssh.get_transport()) as scp:
            # 上传文件
            scp.put(local_file, remote_path)
        
        print("文件上传成功!")
        return remote_path
    except Exception as e:
        print(f"文件上传失败: {str(e)}")
        return None

def verify_file_exists(ssh, remote_path):
    """
    验证文件是否存在于远程服务器上
    """
    try:
        print(f"\n验证远程文件是否存在: {remote_path}")
        # 执行ls命令检查文件
        command = f"ls -la {remote_path}"
        stdin, stdout, stderr = ssh.exec_command(command)
        
        # 检查命令执行结果
        exit_code = stdout.channel.recv_exit_status()
        if exit_code == 0:
            file_info = stdout.read().decode('utf-8')
            print(f"✓ 远程文件存在: {remote_path}")
            print(f"文件信息: {file_info.strip()}")
            return True
        else:
            error_msg = stderr.read().decode('utf-8')
            print(f"✗ 远程文件不存在: {remote_path}")
            print(f"错误信息: {error_msg}")
            return False
    except Exception as e:
        print(f"验证远程文件时出错: {str(e)}")
        return False

def test_http_download(remote_filename):
    """
    测试通过HTTP链接下载文件
    """
    # 构建HTTP下载链接
    # 注意：根据目录结构，summary_plans应该是在sharefile下的子目录
    http_link = f"http://121.40.182.30:8000/sharefile/summary_plans/{remote_filename}"
    
    print(f"\n正在测试HTTP下载...")
    print(f"HTTP下载链接: {http_link}")
    
    try:
        # 发送HTTP GET请求
        print("发送HTTP请求...")
        response = requests.get(http_link, timeout=30)
        
        # 检查响应状态
        if response.status_code == 200:
            print(f"✓ HTTP下载成功! 状态码: {response.status_code}")
            print(f"下载的文件大小: {len(response.content)} 字节")
            
            # 显示下载的内容
            downloaded_content = response.content.decode('utf-8')
            print(f"下载的内容: {downloaded_content}")
            
            return True, downloaded_content
        else:
            print(f"✗ HTTP下载失败! 状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            
            # 检查sharefile目录列表
            print("\n检查sharefile目录内容...")
            list_url = "http://121.40.182.30:8000/sharefile"
            list_response = requests.get(list_url, timeout=10)
            print(f"目录列表URL: {list_url}")
            print(f"目录列表状态码: {list_response.status_code}")
            print(f"目录列表内容: {list_response.text[:500]}...")
            
            return False, None
    except Exception as e:
        print(f"HTTP下载请求出错: {str(e)}")
        return False, None

def main():
    """
    主函数
    """
    print("=== 直接测试SCP上传和HTTP下载 ===")
    
    # 服务器信息
    server_info = {
        'hostname': '121.40.182.30',
        'port': 22,
        'username': 'batago',
        'password': '4008737505'
    }
    
    print("服务器信息:")
    print(f"- 地址: {server_info['hostname']}")
    print(f"- 端口: {server_info['port']}")
    print(f"- 用户名: {server_info['username']}")
    print(f"- 目标目录: /opt/redmine-3.0.1-0/apache2/htdocs/sharefile/summary_plans")
    print(f"- HTTP访问: http://121.40.182.30:8000/sharefile/summary_plans/")
    
    # 创建测试文件
    test_file, original_content = create_test_file()
    test_file_cleanup = True
    
    # SSH连接
    ssh = connect_to_server()
    if not ssh:
        print("\n测试失败: 无法连接到SSH服务器")
        return
    
    try:
        # 定义远程目录
        remote_dir = "/opt/redmine-3.0.1-0/apache2/htdocs/sharefile/summary_plans"
        
        # 创建远程目录
        if not create_remote_directory(ssh, remote_dir):
            print("\n测试失败: 无法创建远程目录")
            return
        
        # 上传文件
        remote_path = upload_file(ssh, test_file, remote_dir)
        if not remote_path:
            print("\n测试失败: 文件上传失败")
            return
        
        # 验证文件是否存在
        if not verify_file_exists(ssh, remote_path):
            print("\n测试失败: 远程文件验证失败")
            return
        
        # 测试HTTP下载
        remote_filename = os.path.basename(test_file)
        download_success, downloaded_content = test_http_download(remote_filename)
        
        # 验证下载内容
        if download_success:
            # 标准化换行符并去除两端空白字符进行比较
            # 将所有换行符统一为\n
            # 标准化原始内容
            original_content_norm = original_content.replace('\r\n', '\n').replace('\r', '\n').strip()
            
            # 标准化下载内容
            downloaded_content_norm = downloaded_content.replace('\r\n', '\n').replace('\r', '\n').strip()
            
            print(f"\n内容比较详情:")
            print(f"- 原始内容长度: {len(original_content)} 字符")
            print(f"- 下载内容长度: {len(downloaded_content)} 字符")
            print(f"- 标准化后原始内容长度: {len(original_content_norm)} 字符")
            print(f"- 标准化后下载内容长度: {len(downloaded_content_norm)} 字符")
            
            # 详细比较每个字符
            if len(original_content_norm) == len(downloaded_content_norm):
                char_diff_found = False
                for i in range(len(original_content_norm)):
                    if original_content_norm[i] != downloaded_content_norm[i]:
                        print(f"字符位置 {i} 不同: '{original_content_norm[i]}' vs '{downloaded_content_norm[i]}'")
                        char_diff_found = True
                        break
                
                if not char_diff_found:
                    print("\n✓ 内容验证通过! 下载的文件内容与原始文件完全一致")
                    print("\n🎉 测试成功完成!")
                    print("\n使用方法:")
                    print(f"1. 通过HTTP直接下载: http://121.40.182.30:8000/sharefile/summary_plans/{remote_filename}")
                    print(f"2. 通过SCP下载: scp batago@121.40.182.30:{remote_path} .")
                else:
                    print("\n✗ 内容验证失败! 发现字符差异")
            else:
                print("\n✗ 内容验证失败! 标准化后的内容长度不同")
                print(f"- 标准化后原始内容: '{original_content_norm}'")
                print(f"- 标准化后下载内容: '{downloaded_content_norm}'")
        else:
            print("\n✗ HTTP下载测试失败")
            print("\n可能的原因:")
            print("1. Apache服务器未正确配置目录权限")
            print("2. summary_plans目录未被正确映射到HTTP路径")
            print("3. 文件权限问题")
            print("4. 网络连接问题")
    
    finally:
        # 关闭SSH连接
        if ssh:
            ssh.close()
            print("\nSSH连接已关闭")
        
        # 清理测试文件
        if test_file_cleanup and os.path.exists(test_file):
            os.remove(test_file)
            print(f"本地测试文件已清理: {test_file}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()