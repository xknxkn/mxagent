#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCP连接测试脚本
用于测试到远程服务器的SCP连接
"""

import sys
import os
import paramiko
from scp import SCPClient
import time

# 服务器配置
HOST = '121.40.182.30'
USERNAME = 'batago'
PASSWORD = '4008737505'
PORT = 22  # SSH默认端口

def test_ssh_connection():
    """测试SSH连接"""
    print("正在测试SSH连接...")
    try:
        # 创建SSH客户端
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # 连接到服务器
        ssh.connect(
            hostname=HOST,
            port=PORT,
            username=USERNAME,
            password=PASSWORD,
            timeout=10
        )

        print("✅ SSH连接成功！")

        # 执行简单命令测试
        stdin, stdout, stderr = ssh.exec_command('echo "Hello from remote server"')
        output = stdout.read().decode('utf-8').strip()
        print(f"远程服务器响应: {output}")

        ssh.close()
        return True

    except paramiko.AuthenticationException:
        print("❌ 认证失败：用户名或密码错误")
        return False
    except paramiko.SSHException as e:
        print(f"❌ SSH连接错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

def test_scp_connection():
    """测试SCP连接"""
    print("\n正在测试SCP连接...")
    try:
        # 创建SSH客户端
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # 连接到服务器
        ssh.connect(
            hostname=HOST,
            port=PORT,
            username=USERNAME,
            password=PASSWORD,
            timeout=10
        )

        print("✅ SCP连接成功！")

        # 创建SCP客户端
        with SCPClient(ssh.get_transport()) as scp:
            # 测试上传一个小文件
            test_content = "This is a test file for SCP connection.\n" + time.strftime("%Y-%m-%d %H:%M:%S")
            test_file = "scp_test.txt"

            # 创建本地测试文件
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)

            # 上传文件到远程服务器
            remote_path = f"/tmp/{test_file}"
            scp.put(test_file, remote_path)
            print(f"✅ 文件上传成功: {test_file} -> {remote_path}")

            # 下载文件回来验证
            downloaded_file = "downloaded_" + test_file
            scp.get(remote_path, downloaded_file)
            print(f"✅ 文件下载成功: {remote_path} -> {downloaded_file}")

            # 验证文件内容
            with open(downloaded_file, 'r', encoding='utf-8') as f:
                downloaded_content = f.read()

            if downloaded_content == test_content:
                print("✅ 文件内容验证成功！")
            else:
                print("⚠️ 文件内容不匹配")

            # 清理本地文件
            os.remove(test_file)
            os.remove(downloaded_file)

        ssh.close()
        return True

    except paramiko.AuthenticationException:
        print("❌ 认证失败：用户名或密码错误")
        return False
    except paramiko.SSHException as e:
        print(f"❌ SSH/SCP连接错误: {e}")
        return False
    except Exception as e:
        print(f"❌ SCP测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("SCP连接测试脚本")
    print(f"目标服务器: {HOST}")
    print(f"用户名: {USERNAME}")
    print("=" * 50)

    # 测试SSH连接
    ssh_success = test_ssh_connection()

    if ssh_success:
        # 测试SCP连接
        scp_success = test_scp_connection()

        if scp_success:
            print("\n🎉 所有测试通过！SCP连接正常。")
        else:
            print("\n⚠️ SSH连接成功，但SCP功能异常。")
    else:
        print("\n❌ SSH连接失败，无法进行SCP测试。")

    print("=" * 50)

if __name__ == "__main__":
    # 设置控制台编码
    if sys.platform.startswith('win'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    main()