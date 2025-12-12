from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from langchain.tools import tool
from tavily import TavilyClient
from typing import List
from pypinyin import lazy_pinyin
from difflib import SequenceMatcher
import markdown
from weasyprint import HTML

import unicodedata
import gradio as gr
import sys
import io
import datetime
import pandas as pd
import unicodedata
import re
import os

# 打印显示中文
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import os
def tavily_search(query: str) -> str:
    """Use this tool to search the web for recent information."""
    # 从环境变量获取API密钥，如果未设置则使用默认值
    api_key = os.environ.get("TAVILY_API_KEY", "tvly-dev-xxxxxxx")
    client = TavilyClient(api_key)
    try:
        response = client.search(
            query=query,
            max_results=1
        )
        print("response of tavily search is", response)
        if 'results' in response and len(response['results']) > 0 and 'content' in response['results'][0]:
            return response['results'][0]["content"]
        else:
            return "搜索结果格式异常，无法获取内容"
    except Exception as e:
        print(f"Tavily搜索错误: {e}")
        return f"搜索失败: {str(e)}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return tavily_search(f"今天{location}的天气,只需要温度，湿度，风级别")

@tool
def get_batago_school_address(batago_site:str)->str:
    """得到 倍塔狗人工智能校区地址."""
    return tavily_search(f"在上海的慕客信信息科技{batago_site}地址")

@tool
def find_batago_student(name:str) -> str:
    """
    Search for the corresponding student names in the Excel dataset through Chinese, English or pinyin.
    学生在倍塔狗也就是batago上学，用PBL方法学习STEM

    Args:
        name (str): The name string to search for.
    """
    
    def normalize_name(s: str) -> str:
        s = unicodedata.normalize('NFKC', str(s))
        s = ''.join(s.split())
        if s.endswith('-综评熊学科'):
            s = s[:-7]
        return s

    def to_pinyin_letters(s: str) -> str:
        # convert Chinese/other characters to pinyin letters, drop non-alpha
        p = ''.join(lazy_pinyin(s))
        p = re.sub(r'[^a-zA-Z]', '', p).lower()
        return p

    # load names from excel
    try:
        df = pd.read_excel('上课反馈20230101to20251102.xlsx')
    except Exception as e:
        return f'读取数据失败: {e}'

    names = df['学生姓名'].dropna().astype(str).unique().tolist()
    # build normalized and pinyin maps
    norm_to_orig = {}
    pinyin_index = {}
    for n in names:
        norm = normalize_name(n)
        norm_to_orig[norm] = n
        py = to_pinyin_letters(n)
        # keep list because different original names could share same pinyin
        pinyin_index.setdefault(py, []).append(n)

    # decide whether input is pinyin-like (only ascii letters, spaces, hyphens) or contains CJK
    input_raw = str(name).strip()
    input_norm = normalize_name(input_raw)

    # If input contains Chinese characters, try exact normalized match first
    if re.search(r'[\u4e00-\u9fff]', input_raw):
        if input_norm in norm_to_orig:
            return norm_to_orig[input_norm]+":是倍塔狗人工智能的学生，欢迎"
        # try case-insensitive contains in original names
        for n in names:
            if input_raw in n:
                return n+":是倍塔狗人工智能的学生，欢迎"

    # prepare pinyin query
    py_query = re.sub(r'[^a-zA-Z]', '', input_raw).lower()
    # if query empty after removing punctuation, fall back to normalized exact match
    if not py_query:
        if input_norm in norm_to_orig:
            return norm_to_orig[input_norm]+":是倍塔狗人工智能的学生，欢迎"
        # fallback: fuzzy compare normalized names
        best = None
        best_score = 0.0
        for norm, orig in norm_to_orig.items():
            score = SequenceMatcher(None, input_norm.lower(), norm.lower()).ratio()
            if score > best_score:
                best_score = score; best = orig
        if best_score >= 0.6:
            return best+":是倍塔狗人工智能的学生，欢迎"
        return f'未能识别名字：{name}. 建议检查拼写或使用中文姓名。'

    # exact pinyin match
    if py_query in pinyin_index:
        # if multiple names share same pinyin, return the first (could be refined)
        return pinyin_index[py_query][0]+":是倍塔狗人工智能的学生，欢迎"

    # fuzzy pinyin match: compute best ratio against all pinyin keys
    best = None
    best_score = 0.0
    for py_key, orig_list in pinyin_index.items():
        score = SequenceMatcher(None, py_query, py_key).ratio()
        if score > best_score:
            best_score = score; best = (py_key, orig_list)

    if best is None:
        return f'未找到候选学生。'

    # if score is reasonable, return first original name of best pinyin key
    py_key, orig_list = best
    if best_score >= 0.5:
        return orig_list[0]+":是倍塔狗人工智能的学生，欢迎"
    # else return message with top few suggestions
    # build top3 suggestions
    scored = []
    for py_key, orig_list in pinyin_index.items():
        scored.append((SequenceMatcher(None, py_query, py_key).ratio(), orig_list[0]))
    scored.sort(reverse=True)
    suggestions = [s for _, s in scored[:5]]
    return f'未能精确匹配"{name}"。最接近的候选: {"、".join(suggestions)} (相似度: {best_score:.2f})'

# normalize name helper
def normalize_name(name: str) -> str:
    n = unicodedata.normalize('NFKC', str(name))
    n = ''.join(n.split())
    if n.endswith('-综评熊学科'):
        n = n[:-7]
    return n


@tool
def generate_summary(student_name: str, time: str) -> str:
    """
    Read the table information and generate a summary of all the course content for the given student within the specified time period accordingly.

    Args:
        student_name (str): The name of the student to query.
        time (str): Time period signals, e.g., "上个季度", "最近一周", "今年", etc.
    Returns:
        str: A summary of the student's learning situation.
    """
    # 首先调用find_batago_student函数验证并获取正确的学生姓名
    student_check_result = find_batago_student(student_name)
    
    # 检查是否找到了学生（如果返回结果包含冒号，则说明找到了学生）
    if ':' not in student_check_result:
        # 如果没有找到学生，直接返回错误信息
        return student_check_result
    
    # 提取冒号前的正确学生姓名
    correct_student_name = student_check_result.split(':')[0]
    
    def quarter_bounds(ref: datetime.datetime, offset=0):
        q = (ref.month - 1) // 3 + 1 + offset
        y = ref.year
        while q <= 0:
            q += 4; y -= 1
        while q > 4:
            q -= 4; y += 1
        start_month = (q - 1) * 3 + 1
        end_month = q * 3
        start = datetime.datetime(y, start_month, 1)
        if end_month in [1,3,5,7,8,10,12]: last = 31
        elif end_month in [4,6,9,11]: last = 30
        else: last = 29 if (y%4==0 and (y%100!=0 or y%400==0)) else 28
        end = datetime.datetime(y, end_month, last)
        return start, end

    now = datetime.datetime.now()

    def parse_time_signals(ts: str):
        s = (ts or '').strip()
        cand = []
        if not s:
            cand.append(('默认:上个季度', quarter_bounds(now, -1)))
            return cand
        if re.search(r'最近|近几|近[一二三四五六七八九十]?天', s):
            cand.append(('最近(30天)', (now - datetime.timedelta(days=30), now)))
        if re.search(r'最近一周|近一周|上周', s):
            cand.append(('最近一周', (now - datetime.timedelta(days=7), now)))
        if re.search(r'最近三个月|近三个月|过去三个月', s):
            cand.append(('最近三个月', (now - datetime.timedelta(days=90), now)))
        if re.search(r'今年', s):
            cand.append(('今年', (datetime.datetime(now.year,1,1), now)))
        if re.search(r'这一季度|本季度|这季度', s):
            st, _ = quarter_bounds(now, 0)
            cand.append(('这一季度', (st, now)))
        if re.search(r'上个季度|上一季度|上季度', s):
            cand.append(('上个季度', quarter_bounds(now, -1)))
        if re.search(r'上个月|上月', s):
            y = now.year; m = now.month - 1
            if m == 0: m = 12; y -= 1
            if m in [1,3,5,7,8,10,12]: last = 31
            elif m in [4,6,9,11]: last = 30
            else: last = 29 if (y%4==0 and (y%100!=0 or y%400==0)) else 28
            cand.append(('上个月', (datetime.datetime(y,m,1), datetime.datetime(y,m,last))))
        y = re.search(r'(20\d{2})', s)
        if y:
            yy = int(y.group(1))
            cand.append((f'{yy}年', (datetime.datetime(yy,1,1), datetime.datetime(yy,12,31))))
        if not cand:
            cand.append(('默认(90天)', (now - datetime.timedelta(days=90), now)))
        return cand

    # load Excel
    try:
        df = pd.read_excel('上课反馈20230101to20251102.xlsx')
    except Exception as e:
        return f'读取数据失败: {e}'

    df['上课时间'] = pd.to_datetime(df['上课时间'], errors='coerce')
    df['学生姓名_标准'] = df['学生姓名'].astype(str).apply(normalize_name)
    target = normalize_name(correct_student_name)
    student_df = df[df['学生姓名_标准'] == target]
    if student_df.empty:
        candidates = list(df['学生姓名'].dropna().astype(str).unique()[:200])
        return f'未找到学生 {correct_student_name} 的记录。数据中可选学生（最多200个显示）: {"、".join(candidates)}'

    # 确保 mapping 变量始终被定义
    signals = parse_time_signals(time)
    mapping = [f"{sig} -> {st.date()} ~ {ed.date()}" for sig,(st,ed) in signals]
    start, end = signals[0][1]
    sel = student_df[(student_df['上课时间'] >= start) & (student_df['上课时间'] <= end)]
    if sel.empty:
        sel = student_df

    contents = sel['内容'].dropna().astype(str).tolist()
    if not contents:
        return f"{correct_student_name} 没有课程'内容'进行分析。"

    # limit length
    max_chars = 6000
    joined = ''
    count = 0
    for t in contents:
        if count + len(t) > max_chars:
            break
        joined += t + ' '
        count += len(t) + 1

    
    try:
        llm = ChatOllama(model="qwen3-vl:235b-cloud")
        result = llm.invoke(f"请把下面多次课程记录的内容合并并生成不超过200字的中文学习总结：{joined}")
        ai_summary = result.content
        return f"时间映射：{'; '.join(mapping)}\n摘要（{correct_student_name}）：{ai_summary[:200]}"
    except Exception as e:
        error_msg = str(e)
        if "Service Temporarily Unavailable" in error_msg or "503" in error_msg:
            return "AI服务暂时不可用，请稍后再试。"
        return f"生成摘要时出错：{error_msg}"

@tool
def career_planning(student_name: str, career_target: str) -> str:
    """
    阅读上课反馈xlsx文件,总结学生学习了什么内容，根据职业生涯目标给出后续课程建议
    
    Args:
        student_name (str): The name of the student to query.
        career_target (str): 职业生涯目标
    Returns:
        str: 后续课程建议，30小时课程计划
    """
    # load Excel
    try:
        df = pd.read_excel('上课反馈20230101to20251102.xlsx')
    except Exception as e:
        return f'读取数据失败: {e}'

    df['上课时间'] = pd.to_datetime(df['上课时间'], errors='coerce')
    df['学生姓名_标准'] = df['学生姓名'].astype(str).apply(normalize_name)
    target = normalize_name(student_name)
    student_df = df[df['学生姓名_标准'] == target]
    if student_df.empty:
        candidates = list(df['学生姓名'].dropna().astype(str).unique()[:200])
        return f'未找到学生 {student_name} 的记录。数据中可选学生（最多200个显示）: {"、".join(candidates)}'
    
    sel = student_df
    contents = sel['内容'].dropna().astype(str).tolist()
    if not contents:
        return f"时间映射：{'; '.join(mapping)}\n{student_name} 在所选时间段没有课程'内容'进行摘要。"
    # limit length
    max_chars = 6000
    joined = ''
    count = 0
    for t in contents:
        if count + len(t) > max_chars:
            break
        joined += t + ' '
        count += len(t) + 1
    try:
        llm = ChatOllama(model="qwen3-vl:235b-cloud", temperature=0.9)
        result = llm.invoke(f'''你是一个STEM教育顾问，首先根学生已经学习过的内容和特点总结，然后规划后续30小时课程内容。
                        针对的学生职业目标是{career_target}
                        要求在1.构造 2.电路 3.编程 4.智能 5.设计 6.整合 7.创新的七个维度规划课程，
                        在学生已有学习内容基础上，保持平衡，预留须继续学习的空间，引导学生持续续课学习至少120小时。
                        学生已经学习过的内容如下：
                        {joined} 
                        请写出课程计划并给出计划制定的理由。
                        计划用markdown格式输出
                        '''
                        )
        markdown_content = result.content
        
        # 创建PDF保存目录
        pdf_dir = "career_plans"
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
        
        # 生成PDF文件名（使用学生姓名和当前时间）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{normalize_name(student_name).replace(' ', '_')}_{timestamp}.pdf"
        pdf_path = os.path.join(pdf_dir, safe_filename)
        
        try:
            # 将markdown转换为HTML
            html_content = markdown.markdown(markdown_content)
            
            # 添加基本样式使PDF看起来更好
            html_with_style = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{
                        font-family: SimSun, serif;
                        margin: 2cm;
                        line-height: 1.6;
                    }}
                    h1, h2, h3, h4, h5, h6 {{
                        color: #333333;
                        margin-top: 1.5em;
                    }}
                    h1 {{
                        text-align: center;
                        border-bottom: 2px solid #333;
                        padding-bottom: 0.3em;
                    }}
                    h2 {{
                        border-bottom: 1px solid #ddd;
                    }}
                    p {{
                        text-align: justify;
                    }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                    }}
                    th {{
                        background-color: #f2f2f2;
                    }}
                    ul, ol {{
                        padding-left: 2em;
                    }}
                </style>
            </head>
            <body>
                <h1>学生职业规划 - {student_name}</h1>
                <p><strong>目标职业:</strong> {career_target}</p>
                <p><strong>生成时间:</strong> {datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
                {html_content}
            </body>
            </html>
            """
            
            # 确保使用正确的WeasyPrint API调用方式
            # 直接使用HTML类的write_pdf方法并传入文件路径
            HTML(string=html_with_style).write_pdf(pdf_path)
            pdf_status = f"PDF已生成并保存至: {pdf_path}"
        except Exception as pdf_error:
            pdf_status = f"生成PDF时出错: {str(pdf_error)}"
        
        # 返回markdown内容和PDF生成状态
        return f"{markdown_content}\n\n---\n\n{pdf_status}"
    except Exception as e:
        error_msg = str(e)
        if "Service Temporarily Unavailable" in error_msg or "503" in error_msg:
            return "AI服务暂时不可用，请稍后再试。"
        return f"生成职业规划时出错：{error_msg}"
    

tools_list = [get_weather,get_batago_school_address,find_batago_student,generate_summary,career_planning]

llm = ChatOllama(model="qwen3-vl:235b-cloud", temperature=0).bind_tools(tools_list)

#根据提示词调用llm invoke 然后处理, 没找到tool的显示回答的问题
def llmtool_invoke_tool(str_input: str):
    try:
        result = llm.invoke(str_input)
        print(result)
        if isinstance(result, AIMessage) and result.tool_calls:
            for call in result.tool_calls:
                tool_result = None
                if isinstance(call, dict):
                    tool_obj = call.get("tool") or call.get("tool_name") or call.get("name")
                    tool_input = call.get("tool_input") or call.get("input") or call.get("args") or {}
                    tool_callable = globals().get(tool_obj)
                    tool_result = tool_callable.invoke(tool_input)
                    return tool_result
        else:
            return result.content
    except Exception as e:
        error_msg = str(e)
        if "Service Temporarily Unavailable" in error_msg or "503" in error_msg:
            return "AI服务暂时不可用，请稍后再试。"
        return f"调用AI模型时出错：{error_msg}"

def chat_fn(message, history):
    # Convert Gradio history to LangChain format
    history_langchain = [
    HumanMessage(content=msg['content']) if msg['role'] == "user" else AIMessage(content=msg['content'])
    for msg in history
    ]
    # Add user input to history
    history_langchain.append(HumanMessage(content=message))
    # Get response from the model
    result = llmtool_invoke_tool(history_langchain)
    return result

demo = gr.ChatInterface(fn=chat_fn, title="Echo Bot")
demo.launch()
