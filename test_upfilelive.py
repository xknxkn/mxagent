# -*- coding: utf-8 -*-
"""
测试UpFileLive库的基本功能，特别是文件分享功能
"""

import os
from UpFileLive.upfilelive import UpFileLive

def test_upfilelive_share():
    """测试UpFileLive文件分享功能"""
    try:
        # 使用upload目录中的文件作为测试
        current_dir = os.path.dirname(os.path.abspath(__file__))
        upload_dir = os.path.join(current_dir, "upload")
        
        # 获取upload目录中的第一个文件
        files_in_upload = os.listdir(upload_dir) if os.path.exists(upload_dir) else []
        if files_in_upload:
            test_file_path = os.path.join(upload_dir, files_in_upload[0])
            print(f"使用测试文件: {test_file_path}")
            
            # 初始化UpFileLive，提供文件路径
            print("初始化UpFileLive...")
            client = UpFileLive(test_file_path)
            print(f"UpFileLive初始化成功，client类型: {type(client)}")
            
            # 打印client的所有属性和方法，以便了解可用功能
            print("\nUpFileLive可用方法:")
            methods = [method for method in dir(client) if not method.startswith('_') and callable(getattr(client, method))]
            for method in methods:
                print(f"- {method}")
            
            print("\nUpFileLive可用属性:")
            attributes = [attr for attr in dir(client) if not attr.startswith('_') and not callable(getattr(client, attr))]
            for attr in attributes:
                print(f"- {attr}: {getattr(client, attr)}")
            
            # 尝试上传并分享文件
            print("\n尝试上传并分享文件...")
            try:
                # 尝试调用upload_file方法
                if hasattr(client, 'upload_file'):
                    result = client.upload_file()
                    print(f"upload_file结果: {result}")
                # 尝试调用upload方法
                elif hasattr(client, 'upload'):
                    result = client.upload()
                    print(f"upload结果: {result}")
                # 尝试调用share方法
                elif hasattr(client, 'share'):
                    result = client.share()
                    print(f"share结果: {result}")
                # 尝试调用主要功能方法
                elif hasattr(client, 'run'):
                    result = client.run()
                    print(f"run结果: {result}")
                elif hasattr(client, 'start'):
                    result = client.start()
                    print(f"start结果: {result}")
                else:
                    print("未找到明确的文件上传/分享方法，请根据上面列出的方法尝试")
            except Exception as e:
                print(f"上传/分享文件时出错: {str(e)}")
        else:
            # 如果没有现有文件，创建一个临时测试文件
            print(f"upload目录中没有文件，创建临时测试文件...")
            os.makedirs(upload_dir, exist_ok=True)
            test_file_path = os.path.join(upload_dir, "test_file.txt")
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write("这是一个测试文件，用于测试UpFileLive库的文件分享功能。")
            print(f"已创建测试文件: {test_file_path}")
            
            # 初始化UpFileLive并测试
            print("初始化UpFileLive...")
            client = UpFileLive(test_file_path)
            print(f"UpFileLive初始化成功")
            
            # 尝试主要功能
            print("尝试分享文件...")
            try:
                if hasattr(client, 'run'):
                    result = client.run()
                    print(f"run结果: {result}")
                elif hasattr(client, 'share'):
                    result = client.share()
                    print(f"share结果: {result}")
                elif hasattr(client, 'upload'):
                    result = client.upload()
                    print(f"upload结果: {result}")
                else:
                    print("请根据上面列出的方法手动尝试分享功能")
            except Exception as e:
                print(f"操作文件时出错: {str(e)}")
            
    except Exception as e:
        print(f"测试UpFileLive时出错: {str(e)}")

def test_upfilelive_full_api():
    """测试UpFileLive库的完整API"""
    try:
        # 导入模块的详细信息
        print("检查UpFileLive模块的详细内容...")
        import UpFileLive
        print(f"模块版本: {UpFileLive.__version__ if hasattr(UpFileLive, '__version__') else '未知'}")
        print(f"模块文档: {UpFileLive.__doc__ if hasattr(UpFileLive, '__doc__') else '无'}")
        
        # 尝试访问子模块
        if hasattr(UpFileLive, 'upfilelive'):
            print("\nupfilelive子模块:")
            upfilelive_methods = [m for m in dir(UpFileLive.upfilelive) if not m.startswith('_') and callable(getattr(UpFileLive.upfilelive, m))]
            print(f"子模块方法: {upfilelive_methods}")
            
            # 检查UpFileLive类的参数
            import inspect
            print("\nUpFileLive类签名:")
            sig = inspect.signature(UpFileLive)
            print(f"参数: {sig}")
            print(f"参数文档: {UpFileLive.__init__.__doc__ if hasattr(UpFileLive.__init__, '__doc__') else '无'}")
            
    except Exception as e:
        print(f"检查UpFileLive模块时出错: {str(e)}")

if __name__ == "__main__":
    print("开始测试UpFileLive库...")
    test_upfilelive_full_api()
    test_upfilelive_share()
    print("测试完成")
