#!/usr/bin/env bash
# SCP命令行测试脚本 (适用于Linux/Mac)
# Windows用户请使用Git Bash或WSL运行

# 服务器配置
HOST="121.40.182.30"
USERNAME="batago"
PASSWORD="4008737505"
PORT="22"

echo "SCP命令行连接测试"
echo "=================="
echo "目标服务器: $HOST"
echo "用户名: $USERNAME"
echo "端口: $PORT"
echo "=================="

# 创建测试文件
echo "创建测试文件..."
echo "This is a test file created at $(date)" > scp_test_$(date +%s).txt
TEST_FILE="scp_test_$(date +%s).txt"

# 使用sshpass进行SCP测试 (需要先安装sshpass)
# 如果没有sshpass，可以使用expect或手动输入密码

echo "测试SSH连接..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $PORT $USERNAME@$HOST "echo 'SSH connection successful'"

if [ $? -eq 0 ]; then
    echo "✅ SSH连接成功"

    # 测试SCP上传
    echo "测试SCP上传..."
    scp -o StrictHostKeyChecking=no -P $PORT $TEST_FILE $USERNAME@$HOST:/tmp/

    if [ $? -eq 0 ]; then
        echo "✅ SCP上传成功"

        # 测试SCP下载
        echo "测试SCP下载..."
        scp -o StrictHostKeyChecking=no -P $PORT $USERNAME@$HOST:/tmp/$TEST_FILE downloaded_$TEST_FILE

        if [ $? -eq 0 ]; then
            echo "✅ SCP下载成功"

            # 验证文件
            if cmp -s $TEST_FILE downloaded_$TEST_FILE; then
                echo "✅ 文件内容验证成功"
            else
                echo "⚠️ 文件内容不匹配"
            fi

            # 清理远程文件
            ssh -o StrictHostKeyChecking=no -p $PORT $USERNAME@$HOST "rm /tmp/$TEST_FILE"
        else
            echo "❌ SCP下载失败"
        fi
    else
        echo "❌ SCP上传失败"
    fi

    # 清理本地文件
    rm -f $TEST_FILE downloaded_$TEST_FILE

else
    echo "❌ SSH连接失败"
fi

echo "=================="
echo "测试完成"