#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试SCP文件上传功能
"""

import os
import sys
import paramiko
from scp import SCPClient

def upload_file_via_scp(local_path, remote_path):
    """使用SCP将文件上传到远程服务器"""
    try:
        # SCP连接参数
        hostname = '121.40.182.30'
        username = 'batago'
        password = '4008737505'
        port = 22

        # 创建SSH客户端
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # 连接到服务器
        ssh.connect(hostname=hostname, username=username, password=password, port=port)
        print("SSH连接成功")

        # 确保远程目录存在
        remote_dir = os.path.dirname(remote_path)
        stdin, stdout, stderr = ssh.exec_command(f'mkdir -p {remote_dir}')
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            error_msg = stderr.read().decode('utf-8')
            print(f"创建远程目录失败: {error_msg}")
            ssh.close()
            return False
        print(f"远程目录创建成功: {remote_dir}")

        # 创建SCP客户端
        with SCPClient(ssh.get_transport()) as scp:
            # 上传文件
            scp.put(local_path, remote_path)
            print(f"文件已成功上传到服务器: {remote_path}")

        ssh.close()
        return True
    except Exception as e:
        print(f"SCP上传失败: {str(e)}")
        return False

def test_scp_upload():
    """测试SCP上传功能"""
    # 创建一个测试文件
    test_content = "这是一个测试文件，用于验证SCP上传功能。\n测试时间：" + str(datetime.datetime.now())
    test_file_path = "test_upload.txt"

    try:
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        print(f"测试文件已创建: {test_file_path}")

        # 上传文件
        remote_path = "/opt/redmine-3.0.1-0/apache2/htdocs/sharefile/test_upload.txt"
        if upload_file_via_scp(test_file_path, remote_path):
            print("SCP上传测试成功！")
            scp_command = f"scp batago@121.40.182.30:{remote_path} ."
            print(f"SCP下载命令: {scp_command}")
        else:
            print("SCP上传测试失败！")

    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
    finally:
        # 清理测试文件
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            print(f"测试文件已清理: {test_file_path}")

if __name__ == "__main__":
    import datetime
    test_scp_upload()