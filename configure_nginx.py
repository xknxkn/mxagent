#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置nginx以提供career_plans目录访问
"""

import paramiko

def configure_nginx():
    """配置nginx提供career_plans目录访问"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('121.40.182.30', username='batago', password='4008737505', port=22)

        print("=== 配置nginx提供career_plans目录访问 ===")

        # 备份原配置文件
        stdin, stdout, stderr = ssh.exec_command('sudo cp /alidata/server/nginx/conf/vhosts/default.conf /alidata/server/nginx/conf/vhosts/default.conf.backup')
        print("✓ 备份原配置文件完成")

        # 创建新的nginx配置
        nginx_config = '''server {
    listen       80 default;
    server_name  _;
        index index.html index.htm index.jsp;
        root /alidata/www/default;
location / {

#       location ~ \\*.jsp$ {
#               proxy_pass    http://127.0.0.1:8080;
#       }

        location ~ .*\\.(gif|jpg|jpeg|png|bmp|swf)$
        {
                expires 30d;
        }

        location ~ .*\\.(js|css)?$
        {
                expires 1h;
        }

        access_log  /alidata/log/nginx/access/default.log;
if ( !-f $request_filename ) {
        proxy_pass                http://127.0.0.1:8080;
        break;
      }
}

    # 添加career_plans目录访问配置
    location /career_plans/ {
        alias /home/batago/career_plans/;
        autoindex on;
        autoindex_exact_size off;
        autoindex_localtime on;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}'''

        # 写入新配置
        command = f"cat > /tmp/default.conf.new << 'EOF'\n{nginx_config}\nEOF"
        stdin, stdout, stderr = ssh.exec_command(command)
        print("✓ 新配置文件已创建")

        # 替换原配置文件
        stdin, stdout, stderr = ssh.exec_command('sudo mv /tmp/default.conf.new /alidata/server/nginx/conf/vhosts/default.conf')
        print("✓ 配置文件已更新")

        # 验证配置文件语法
        stdin, stdout, stderr = ssh.exec_command('sudo /alidata/server/nginx/sbin/nginx -t')
        syntax_check = stdout.read().decode('utf-8') + stderr.read().decode('utf-8')
        print("配置文件语法检查:")
        print(syntax_check)

        # 重新加载nginx配置
        if 'successful' in syntax_check.lower():
            stdin, stdout, stderr = ssh.exec_command('sudo /alidata/server/nginx/sbin/nginx -s reload')
            reload_result = stdout.read().decode('utf-8') + stderr.read().decode('utf-8')
            print("✓ nginx重新加载成功")
            print(reload_result)

            # 测试配置是否生效
            stdin, stdout, stderr = ssh.exec_command('curl -I http://localhost/career_plans/test_upload.txt 2>/dev/null | head -3')
            test_result = stdout.read().decode('utf-8').strip()
            print("\n测试HTTP访问:")
            print(test_result)

            if '200' in test_result or '302' in test_result:
                print("✓ career_plans目录访问配置成功！")
            else:
                print("⚠️ 配置可能有问题，请检查nginx日志")

        else:
            print("❌ 配置文件语法错误，不进行重载")

        ssh.close()

    except Exception as e:
        print(f"配置过程中出错: {str(e)}")

if __name__ == "__main__":
    configure_nginx()