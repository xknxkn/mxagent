# 倍塔狗AI教育助手 (Batago AI Education Assistant)

倍塔狗AI教育助手是一个基于Python和Gradio构建的教育辅助工具，主要用于学生职业规划、学习内容分析和STEM课程规划。

## 主要功能

1. **学生职业规划**：根据学生的学习历史、职业目标、家长期望等信息，规划后续30小时STEM课程内容
2. **会话状态管理**：支持多会话管理，记录用户交互历史和偏好设置
3. **文件分享功能**：通过SCP实现文件上传和分享
4. **LLM集成**：集成了ChatOllama等大语言模型，提供智能对话和内容生成
5. **数据分析**：支持Excel文件读取和分析学生学习数据
6. **课程规划**：在构造、电路、编程、智能、设计、整合、创新七个维度规划平衡的课程内容

## 安装

### 环境要求

- Python 3.8+
- pip

### 安装步骤

1. 克隆或下载项目文件

2. 创建虚拟环境（推荐）
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate  # Windows
   ```

3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

## 使用

### 1. 启动主应用

```bash
python batago.py
```

### 2. 启动会话演示

```bash
python gradiosession.py
```

访问 http://127.0.0.1:7862 查看演示界面

## 项目结构

```
.
├── batago.py          # 主应用程序，包含核心业务逻辑
├── gradiosession.py   # 会话管理演示程序
├── requirements.txt   # 项目依赖
├── README.md          # 项目说明文档
└── career_plans/      # 生成的职业规划文档存储目录
```

## 核心模块说明

### 1. batago.py

主要功能模块：

- **career_planning**：学生职业规划生成功能
  - 参数：学生姓名、职业目标、家长希望、后续课程建议
  - 返回：markdown格式的课程规划

- **upload_file_via_scp**：文件上传功能
  - 通过SCP将文件上传到远程服务器

- **load_excel_data**：Excel数据加载
  - 读取学生上课反馈数据

### 2. gradiosession.py

会话管理演示：

- 模拟多会话管理
- 支持设置兴趣主题
- 会话信息查看和重置

## 功能使用示例

### 职业规划示例

```python
from batago import career_planning

# 基本使用
result = career_planning(
    student_name="张三",
    career_target="我想成为一名软件工程师"
)

# 包含家长希望和后续课程建议
result = career_planning(
    student_name="张三",
    career_target="我想成为一名软件工程师",
    parent_expect="希望孩子能够打下坚实的编程基础",
    following_course_suggest="建议加强Python编程和算法学习"
)
```

### 会话管理示例

1. 在浏览器中访问 http://127.0.0.1:7862
2. 使用以下命令：
   - `/set_topic 编程` - 设置兴趣主题
   - `/show_session` - 查看会话信息
   - `/reset_session` - 重置会话

## 依赖说明

主要依赖：

- gradio - Web界面框架
- langchain_ollama - LLM集成
- pandas - 数据分析
- openpyxl - Excel文件读取
- paramiko/scp - 文件传输
- pypandoc - 文档格式转换

完整依赖列表请查看 `requirements.txt`

## 配置说明

### 远程服务器配置

在 `batago.py` 中可以配置SCP服务器参数：

```python
hostname = '121.40.182.30'
username = 'batago'
password = '4008737505'
port = 22
```

### LLM配置

```python
llm = ChatOllama(model="qwen3-vl:235b-cloud", temperature=0.9)
```

## 注意事项

1. 确保已安装所有依赖
2. 使用前请确保Excel数据文件格式正确
3. 远程文件上传功能需要确保网络连接正常
4. LLM功能需要配置正确的模型参数

## 更新日志

- 添加了家长希望和后续课程建议参数到职业规划功能
- 修复了Excel文件读取错误
- 优化了会话管理功能
- 修复了SCP文件上传功能

## 许可证

MIT License
