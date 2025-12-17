#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动配置nginx career_plans访问
"""

import paramiko

def configure_nginx_manual():
    """手动配置nginx"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('121.40.182.30', username='batago', password='4008737505', port=22)

        print("=== 手动配置nginx career_plans访问 ===")

        # 读取当前配置
        stdin, stdout, stderr = ssh.exec_command('cat /alidata/server/nginx/conf/vhosts/default.conf')
        current_config = stdout.read().decode('utf-8')
        print("当前配置:")
        print(current_config)
        print("---")

        # 替换配置，添加career_plans location
        new_config = current_config.replace(
            'if ( !-f $request_filename ) {\n        proxy_pass                http://127.0.0.1:8080;\n        break;\n      }\n}',
            'if ( !-f $request_filename ) {\n        proxy_pass                http://127.0.0.1:8080;\n        break;\n      }\n}\n\n    # 添加career_plans目录访问配置\n    location /career_plans/ {\n        alias /home/batago/career_plans/;\n        autoindex on;\n        autoindex_exact_size off;\n        autoindex_localtime on;\n        expires 30d;\n        add_header Cache-Control "public, immutable";\n    }'
        )

        print("新配置:")
        print(new_config)

        # 写入新配置
        command = "cat > /tmp/default.conf.new << 'EOF'\n" + new_config + "\nEOF"
        stdin, stdout, stderr = ssh.exec_command(command)

        # 备份原配置
        stdin, stdout, stderr = ssh.exec_command('sudo cp /alidata/server/nginx/conf/vhosts/default.conf /alidata/server/nginx/conf/vhosts/default.conf.backup_manual')

        # 替换配置
        stdin, stdout, stderr = ssh.exec_command('sudo mv /tmp/default.conf.new /alidata/server/nginx/conf/vhosts/default.conf')

        # 验证语法
        stdin, stdout, stderr = ssh.exec_command('sudo /alidata/server/nginx/sbin/nginx -t')
        syntax_check = stdout.read().decode('utf-8') + stderr.read().decode('utf-8')
        print("\n语法检查:")
        print(syntax_check)

        # 重新加载
        if 'successful' in syntax_check.lower():
            stdin, stdout, stderr = ssh.exec_command('sudo /alidata/server/nginx/sbin/nginx -s reload')
            reload_result = stdout.read().decode('utf-8') + stderr.read().decode('utf-8')
            print("重新加载结果:")
            print(reload_result)

            # 测试访问
            stdin, stdout, stderr = ssh.exec_command('curl -I http://localhost/career_plans/test_upload.txt 2>/dev/null | head -3')
            test_result = stdout.read().decode('utf-8').strip()
            print("\nHTTP访问测试:")
            print(test_result)

            if '200' in test_result:
                print("✓ 配置成功！")
            else:
                print("⚠️ 配置可能仍有问题")
        else:
            print("❌ 语法错误，配置失败")

        ssh.close()

    except Exception as e:
        print(f"配置过程中出错: {str(e)}")

if __name__ == "__main__":
    configure_nginx_manual()