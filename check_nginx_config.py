#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查nginx配置
"""

import paramiko

def check_nginx_config():
    """检查nginx配置"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('121.40.182.30', username='batago', password='4008737505', port=22)

        print("=== Nginx配置检查 ===")

        # 检查nginx主配置文件中的career_plans配置
        stdin, stdout, stderr = ssh.exec_command('grep -n "career_plans" /alidata/server/nginx/conf/nginx.conf || echo "nginx.conf中未找到career_plans配置"')
        print("\nnginx.conf中的career_plans配置:")
        print(stdout.read().decode('utf-8').strip())

        # 检查vhosts目录
        stdin, stdout, stderr = ssh.exec_command('ls -la /alidata/server/nginx/conf/vhosts/ 2>/dev/null || echo "未找到vhosts目录"')
        print("\nvhosts目录内容:")
        print(stdout.read().decode('utf-8').strip())

        # 检查默认站点配置
        stdin, stdout, stderr = ssh.exec_command('cat /alidata/server/nginx/conf/vhosts/default.conf 2>/dev/null | grep -E "(location|root|server_name)" | head -20 || echo "未找到default.conf或相关配置"')
        print("\ndefault.conf中的关键配置:")
        print(stdout.read().decode('utf-8').strip())

        # 检查是否有其他配置文件引用career_plans
        stdin, stdout, stderr = ssh.exec_command('find /alidata/server/nginx/conf/ -name "*.conf" -exec grep -l "career_plans" {} \; 2>/dev/null || echo "未找到包含career_plans的配置文件"')
        print("\n包含career_plans的配置文件:")
        print(stdout.read().decode('utf-8').strip())

        # 测试HTTP访问
        stdin, stdout, stderr = ssh.exec_command('curl -I http://localhost/career_plans/test_upload.txt 2>/dev/null | head -5 || echo "无法通过localhost访问career_plans"')
        print("\n本地HTTP访问测试:")
        print(stdout.read().decode('utf-8').strip())

        ssh.close()

    except Exception as e:
        print(f"检查过程中出错: {str(e)}")

if __name__ == "__main__":
    check_nginx_config()