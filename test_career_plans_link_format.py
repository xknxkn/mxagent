import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from batago import career_planning

def test_career_planning_link_format():
    print("开始测试career_planning函数的文件分享链接格式...")
    
    try:
        # 使用学生名称"吴天昊"进行测试
        result = career_planning(student_name="吴天昊")
        
        print("\n测试结果:")
        print(result)
        
        # 检查返回结果中是否包含正确格式的分享链接
        if isinstance(result, dict) and 'content' in result:
            content = result['content']
            
            # 检查是否包含正确的图标和格式
            if "🔗 **文件分享链接**: " in content:
                print("\n✅ 测试成功：文件分享链接格式正确，使用了🔗图标并且链接没有被反引号包裹！")
                
                # 提取分享链接
                for line in content.split('\n'):
                    if "🔗 **文件分享链接**: " in line:
                        link = line.split("🔗 **文件分享链接**: ")[1]
                        print(f"\n提取到的分享链接: {link}")
                        
                        # 验证链接格式是否正确
                        if link.startswith("http://121.40.182.30:8000/sharefile/career_plans/"):
                            print("✅ 分享链接格式正确，指向了career_plans目录！")
                        else:
                            print("❌ 分享链接路径不正确！")
            else:
                print("❌ 测试失败：未找到正确格式的文件分享链接！")
                print("请检查career_planning函数是否已正确修改。")
        else:
            print("❌ 测试失败：返回结果格式不正确！")
            
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {str(e)}")

if __name__ == "__main__":
    test_career_planning_link_format()
