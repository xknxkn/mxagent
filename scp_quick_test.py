#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的SCP连接测试脚本
"""

import paramiko
import sys

# 服务器配置
HOST = '121.40.182.30'
USERNAME = 'batago'
PASSWORD = '4008737505'
PORT = 22

def quick_test():
    """快速测试连接"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        print(f"连接到 {HOST}...")
        ssh.connect(
            hostname=HOST,
            port=PORT,
            username=USERNAME,
            password=PASSWORD,
            timeout=10
        )

        # 执行测试命令
        stdin, stdout, stderr = ssh.exec_command('pwd && whoami')
        output = stdout.read().decode('utf-8').strip()
        error = stderr.read().decode('utf-8').strip()

        ssh.close()

        print("✅ 连接成功！")
        print(f"服务器响应:\n{output}")
        if error:
            print(f"错误信息: {error}")

        return True

    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("SCP连接快速测试")
    print("=" * 30)
    success = quick_test()
    print("=" * 30)
    sys.exit(0 if success else 1)