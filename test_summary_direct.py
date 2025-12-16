#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接测试脚本：仅测试generate_summary函数的文档生成部分
"""
import os
import sys
import datetime

def test_direct_summary_generation():
    """直接测试摘要文档生成功能"""
    print("开始直接测试摘要文档生成...")
    
    try:
        import pypandoc
        
        # 学生姓名和时间
        student_name = "测试学生"
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建summary_plans目录
        summary_dir = "summary_plans"
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        
        # 创建测试的Markdown内容
        test_content = f"""# 倍塔狗人工智能学生学习情况阶段性总结  
学生:{student_name}  
时间:最近(30天) -> 2025-11-14 ~ 2025-12-14  

## 学习内容总结

### 1. 树莓派学习与实践
学生成功安装了树莓派并配置了VNC远程访问，学习了Linux命令行基础知识，尝试安装py5库并解决安装过程中遇到的技术问题。这一阶段的学习培养了学生的动手能力和问题解决能力，学生对成功配置树莓派表示很有成就感。

### 2. 木结构制作与工程实践
学生参与了星际空间木头框架的设计与制作，学习了尺寸划线、角铁螺丝固定、电动工具使用等实用技能。通过组装完成场景总图框架并摆放亚克力半球穹顶，学生体验了从设计到实现的完整工程流程，锻炼了精细动作和团队协作能力。

### 3. 竞赛准备与项目规划
学生积极参与Robotex比赛准备工作，学习使用draw.io绘制系统框图、流程图和半透膜投影图，修改PPT以适配演讲需求，并编写了材料清单(BOM)。这一阶段的学习体现了目标导向的学习方法，学生通过可视化工具更好地理解了比赛内容的整体结构。

### 4. AI工具应用与编程实践
学生学习利用AI工具修改比赛文档、生成演示文稿，并为阿米巴项目添加了大小参数和寿命值等功能。通过使用VSCode的Copilot功能，学生探索了AI辅助编程的高效工作方式，实现了星际生物的软件设计。

### 5. 竞赛策略与知识应用
学生深入讨论了Robotex竞赛方案和星际奥德赛比赛规则，将前期学习的py5编程知识应用到阿米巴虫项目中，实现了互动性功能。这一阶段的学习帮助学生将理论知识与实践项目相结合，进一步理解和掌握了面向对象编程的核心概念。

## 学习特点与进步

1. **理论与实践结合**：学生能够将课堂学习的编程知识应用到实际项目中，特别是在阿米巴虫项目和Robotex竞赛准备中。

2. **问题解决能力**：面对技术挑战时，学生展现出积极的问题解决态度，如在py5安装和配置过程中不断尝试和调整。

3. **跨学科学习**：学生同时涉足编程、电子、机械设计等多个领域，体现了STEM综合素养的培养。

4. **目标导向学习**：在竞赛准备过程中，学生以明确目标为导向，高效组织学习内容和项目进度。

## 后续学习建议

1. 继续深入学习Python编程，特别是面向对象编程和图形界面开发。

2. 加强工程设计能力，尝试更复杂的机械结构设计和实现。

3. 探索AI与传统编程的结合应用，提升项目的智能化水平。

4. 针对Robotex竞赛进行有针对性的训练，优化项目细节和性能。"""
        
        # 生成文件名
        safe_filename = f"{student_name}_总结_测试_{timestamp}.docx"
        docx_path = os.path.join(summary_dir, safe_filename)
        temp_md_path = os.path.join(summary_dir, f"temp_{timestamp}.md")
        
        # 写入临时Markdown文件
        print(f"创建临时Markdown文件: {temp_md_path}")
        with open(temp_md_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # 设置UTF-8环境变量
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # 使用成功的pandoc配置
        extra_args = [
            '--standalone',
            '--from=markdown+smart',
            '--to=docx',
            '--wrap=none',
            f'--metadata=title={student_name}的学习总结',
            '--markdown-headings=atx'
        ]
        
        print(f"\n开始转换为Word文档: {docx_path}")
        print(f"使用参数: {extra_args}")
        
        # 执行转换
        pypandoc.convert_file(
            temp_md_path,
            'docx',
            outputfile=docx_path,
            extra_args=extra_args
        )
        
        # 检查结果
        if os.path.exists(docx_path) and os.path.getsize(docx_path) > 0:
            file_size = os.path.getsize(docx_path) / 1024
            print(f"✓ Word文档转换成功!")
            print(f"  文件大小: {file_size:.2f} KB")
            print(f"  文件路径: {docx_path}")
            return True
        else:
            print(f"✗ Word文档转换失败: 文件未创建或为空")
            return False
            
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时文件
        if 'temp_md_path' in locals() and os.path.exists(temp_md_path):
            try:
                os.remove(temp_md_path)
                print(f"清理临时Markdown文件: {temp_md_path}")
            except:
                pass

if __name__ == "__main__":
    success = test_direct_summary_generation()
    print(f"\n测试{'成功' if success else '失败'}")
    sys.exit(0 if success else 1)
