#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查远程服务器web服务器配置
"""

import paramiko

def check_remote_server():
    """检查远程服务器配置"""
    try:
        # 连接到服务器
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('121.40.182.30', username='batago', password='4008737505', port=22)

        print("=== 远程服务器配置检查 ===")

        # 检查career_plans目录
        stdin, stdout, stderr = ssh.exec_command('ls -la /home/batago/career_plans/')
        print("\n远程目录内容 (/home/batago/career_plans/):")
        output = stdout.read().decode('utf-8')
        if output.strip():
            print(output)
        else:
            print("(目录为空)")

        # 检查是否有web服务器安装
        stdin, stdout, stderr = ssh.exec_command('which nginx 2>/dev/null || which apache2 2>/dev/null || which httpd 2>/dev/null || echo "未找到常见web服务器"')
        web_server = stdout.read().decode('utf-8').strip()
        print(f"\nWeb服务器检查: {web_server}")

        # 检查是否有运行中的web服务器
        stdin, stdout, stderr = ssh.exec_command('ps aux | grep -E "(nginx|apache|httpd)" | grep -v grep || echo "未发现运行中的web服务器"')
        running_servers = stdout.read().decode('utf-8').strip()
        print(f"\n运行中的web服务器: {running_servers}")

        # 检查是否有python的http.server运行
        stdin, stdout, stderr = ssh.exec_command('ps aux | grep "python.*http.server\|python.*-m.*http" | grep -v grep || echo "未发现Python HTTP服务器"')
        python_http = stdout.read().decode('utf-8').strip()
        print(f"\nPython HTTP服务器: {python_http}")

        # 检查防火墙设置
        stdin, stdout, stderr = ssh.exec_command('sudo ufw status 2>/dev/null || sudo firewall-cmd --state 2>/dev/null || echo "无法检查防火墙状态"')
        firewall = stdout.read().decode('utf-8').strip()
        print(f"\n防火墙状态: {firewall}")

        ssh.close()

    except Exception as e:
        print(f"检查过程中出错: {str(e)}")

if __name__ == "__main__":
    check_remote_server()