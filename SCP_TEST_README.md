# SCP连接测试工具

本目录包含用于测试到远程服务器SCP连接的工具。

## 服务器信息
- **地址**: 121.40.182.30
- **用户名**: batago
- **密码**: 4008737505
- **端口**: 22 (SSH默认端口)

## 测试脚本

### 1. Python完整测试脚本 (`test_scp_connection.py`)
功能最完整的测试脚本，包括：
- SSH连接测试
- SCP文件上传下载测试
- 文件内容验证

**使用方法**:
```bash
pip install -r scp_requirements.txt
python test_scp_connection.py
```

### 2. Python快速测试脚本 (`scp_quick_test.py`)
简化的连接测试，只验证SSH连接是否正常。

**使用方法**:
```bash
python scp_quick_test.py
```

### 3. 命令行测试脚本 (`test_scp_command.sh`)
使用系统SCP命令进行测试，适用于Linux/Mac环境。
Windows用户需要在Git Bash或WSL中运行。

**使用方法**:
```bash
chmod +x test_scp_command.sh
./test_scp_command.sh
```

## 依赖包
- paramiko >= 3.0.0
- scp >= 0.15.0

## 测试结果
✅ 所有测试均已验证通过，SCP连接正常工作。

## 注意事项
- 确保网络连接正常
- 服务器防火墙允许SSH连接
- 密码认证已启用（某些服务器可能只允许密钥认证）