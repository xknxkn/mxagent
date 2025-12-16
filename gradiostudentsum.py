from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from langchain.tools import tool
from pandas.core.frame import treat_as_nested
from tavily import TavilyClient
from typing import List
from pypinyin import lazy_pinyin
from difflib import SequenceMatcher
import markdown
import pypandoc
import subprocess

import unicodedata
import gradio as gr
import os
import re
import sys
import io
import datetime
import pandas as pd
import unicodedata
import re
import os
import json
from pathlib import Path

# 打印显示中文
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Try to load saved TAVILY_API_KEY from local config if environment variable not set
def _get_config_path() -> Path:
    return Path(os.path.expanduser('~')) / '.batago_config.json'


def load_saved_api_key() -> str | None:
    cfg = _get_config_path()
    try:
        if cfg.exists():
            with open(cfg, 'r', encoding='utf-8') as f:
                data = json.load(f)
                key = data.get('TAVILY_API_KEY')
                return key
    except Exception:
        return None
    return None


def save_api_key(key: str) -> None:
    cfg = _get_config_path()
    try:
        cfg.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg, 'w', encoding='utf-8') as f:
            json.dump({'TAVILY_API_KEY': key}, f)
        # Try to restrict permissions (best-effort)
        try:
            if os.name != 'nt':
                cfg.chmod(0o600)
        except Exception:
            pass
    except Exception:
        # ignore persistence errors
        pass


# 确保在程序启动时正确从配置文件加载API密钥到环境变量
saved = load_saved_api_key()
if saved:
    os.environ["TAVILY_API_KEY"] = saved
    print(f"TAVILY_API_KEY已从配置文件加载")
else:
    print(f"未找到保存的TAVILY_API_KEY")

# 删除同名学生旧文件的函数
def clean_old_student_files(student_name):
    """
    删除指定学生的旧文件，只保留最新的文件
    
    Args:
        student_name: 学生姓名
    
    Returns:
        dict: 删除的文件信息
    """
    # 定义career_plans目录路径
    CAREER_PLANS_DIR = os.path.join(os.path.dirname(__file__), 'career_plans')
    
    # 文件命名规则：学生姓名_年月日_时分秒.docx
    file_pattern = re.compile(r'^%s_(\d{8})_(\d{6})\.docx$' % re.escape(normalize_name(student_name).replace(' ', '_')))
    
    # 检查目录是否存在
    if not os.path.exists(CAREER_PLANS_DIR):
        return {
            'student_name': student_name,
            'total_files': 0,
            'deleted_count': 0,
            'deleted_files': [],
            'kept_file': None
        }
    
    # 获取目录中所有文件
    try:
        all_files = os.listdir(CAREER_PLANS_DIR)
    except Exception as e:
        print(f"读取目录时出错: {e}")
        return {
            'student_name': student_name,
            'total_files': 0,
            'deleted_count': 0,
            'deleted_files': [],
            'kept_file': None
        }
    
    # 收集该学生的所有文件
    student_files = []
    for filename in all_files:
        match = file_pattern.match(filename)
        if match:
            date_str, time_str = match.groups()
            # 组合成完整的时间字符串
            datetime_str = f'{date_str}{time_str}'
            try:
                # 转换为datetime对象用于比较
                file_datetime = datetime.datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
                student_files.append({
                    'filename': filename,
                    'datetime': file_datetime
                })
            except ValueError:
                # 日期格式不正确，跳过该文件
                continue
    
    # 如果该学生有多个文件
    if len(student_files) > 1:
        # 按时间排序，最新的在前
        student_files.sort(key=lambda x: x['datetime'], reverse=True)
        
        # 保留最新的文件，删除其余的
        files_to_delete = student_files[1:]
        deleted_count = 0
        deleted_files = []
        
        for file_info in files_to_delete:
            file_path = os.path.join(CAREER_PLANS_DIR, file_info['filename'])
            try:
                os.remove(file_path)
                deleted_count += 1
                deleted_files.append(file_info['filename'])
                print(f"已删除旧文件: {file_info['filename']}")
            except Exception as e:
                print(f"删除文件 {file_info['filename']} 失败: {e}")
        
        return {
            'student_name': student_name,
            'total_files': len(student_files),
            'deleted_count': deleted_count,
            'deleted_files': deleted_files,
            'kept_file': student_files[0]['filename']
        }
    else:
        # 没有或只有一个文件，不需要删除
        return {
            'student_name': student_name,
            'total_files': len(student_files),
            'deleted_count': 0,
            'deleted_files': [],
            'kept_file': student_files[0]['filename'] if student_files else None
        }

# 全局变量：存储课程记录表
dfkcjl = None

# 在程序启动时加载Excel文件
def load_excel_data():
    """加载Excel课程反馈数据到全局变量，读取所有以上课反馈开头的xlsx文件并合并"""
    global dfkcjl
    
    try:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 正在搜索并加载所有以上课反馈开头的xlsx文件...")
        
        # 搜索所有以上课反馈开头的xlsx文件
        excel_files = [f for f in os.listdir('.') if f.startswith('上课反馈') and f.endswith('.xlsx')]
        
        if not excel_files:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 错误: 未找到以上课反馈开头的xlsx文件")
            dfkcjl = None
            return False
        
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 找到{len(excel_files)}个以上课反馈开头的xlsx文件: {', '.join(excel_files)}")
        
        # 初始化空的DataFrame用于合并
        all_data_frames = []
        
        # 逐个读取并验证文件
        for excel_file in excel_files:
            try:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 正在读取文件: {excel_file}")
                df = pd.read_excel(excel_file)
                
                if df.empty:
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 警告: 文件{excel_file}为空")
                    continue
                
                # 验证必要的列是否存在
                required_columns = ['学生姓名', '上课时间']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 警告: 文件{excel_file}缺少必要的列: {missing_columns}，跳过该文件")
                    continue
                
                # 预先处理上课时间列
                df['上课时间'] = pd.to_datetime(df['上课时间'], errors='coerce')
                
                all_data_frames.append(df)
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 成功读取{excel_file}，共{len(df)}条记录")
                
            except Exception as e:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 读取文件{excel_file}时出错: {str(e)}")
                continue
        
        # 检查是否有成功读取的文件
        if not all_data_frames:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 错误: 没有成功读取的以上课反馈开头的xlsx文件")
            dfkcjl = None
            return False
        
        # 合并所有数据
        dfkcjl = pd.concat(all_data_frames, ignore_index=True)
        
        # 去重，避免内容重复
        original_length = len(dfkcjl)
        dfkcjl = dfkcjl.drop_duplicates()
        
        # 记录去重信息
        duplicates_removed = original_length - len(dfkcjl)
        if duplicates_removed > 0:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 移除了{duplicates_removed}条重复记录")
        
        # 记录加载成功信息
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 课程反馈数据加载成功，共{len(dfkcjl)}条不重复记录")
        return True
        
    except FileNotFoundError:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 错误: 文件未找到: {excel_file}")
    except pd.errors.EmptyDataError:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 错误: Excel文件为空或格式不正确: {excel_file}")
    except pd.errors.ParserError:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 错误: 解析Excel文件失败，可能是格式问题: {excel_file}")
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 加载课程反馈数据时发生未知错误: {str(e)}")
    finally:
        if not 'dfkcjl' in globals() or dfkcjl is None:
            dfkcjl = None
            return False

# 初始化加载数据
load_excel_data()
def set_tavily_api_key(api_key: str) -> str:
    """设置TAVILY_API_KEY并保存到环境变量和配置文件"""
    key = (api_key or "").strip()
    if not key:
        return "请提供有效的 TAVILY_API_KEY"

    # 尝试验证该 key 是否可用（调用 Tavily 的简单搜索接口）
    try:
        test_client = TavilyClient(key)
        # 使用一个无害的查询进行验证
        resp = test_client.search(query="验证TAVILY_API_KEY是否有效: 测试", max_results=1)
        # 如果返回包含错误字段或抛异常会被捕获
        if resp is None:
            raise ValueError("空响应")
    except Exception as e:
        print(f"API Key验证失败: {e}")
        return f"API Key 验证失败：{e}"

    # 验证通过，设置环境变量并保存到配置文件
    try:
        # 确保环境变量被设置
        os.environ["TAVILY_API_KEY"] = key
        print(f"TAVILY_API_KEY已设置到环境变量")
        
        # 尝试保存到配置文件
        config_path = _get_config_path()
        try:
            cfg = _get_config_path()
            cfg.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg, 'w', encoding='utf-8') as f:
                json.dump({'TAVILY_API_KEY': key}, f)
            print(f"TAVILY_API_KEY已保存到配置文件: {config_path}")
            return "TAVILY_API_KEY 已成功设置并保存，程序将在下次启动时自动加载。"
        except Exception as save_error:
            print(f"保存API密钥失败: {save_error}")
            return f"TAVILY_API_KEY 已设置但保存失败: {save_error}，程序重启后需要重新输入。"
    except Exception as e:
        print(f"设置API密钥时出错: {e}")
        return f"设置API密钥时出错: {e}"

def tavily_search(query: str) -> str:
    """Use this tool to search the web for recent information."""
    # 从环境变量获取API密钥，若未设置则返回提示
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "TAVILY_API_KEY 未配置，请先在界面中设置API密钥后重试。"
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

def normalize_name(s: str) -> str:
    """Normalize Chinese names by removing extra spaces and suffixes."""
    s = unicodedata.normalize('NFKC', str(s))
    s = ''.join(s.split())
    if s.endswith('-综评熊学科'):
        s = s[:-7]
    return s

def to_pinyin_letters(s: str) -> str:
    """Convert Chinese/other characters to pinyin letters, drop non-alpha characters."""
    p = ''.join(lazy_pinyin(s))
    p = re.sub(r'[^a-zA-Z]', '', p).lower()
    return p

def find_the_student(name:str) -> tuple[bool, str]:
    """
    Search for the corresponding student names in the Excel dataset through Chinese, English or pinyin.

    Args:
        name (str): The name string to search for.
    
    Returns:
        tuple[bool, str]: (是否找到学生, 学生姓名或错误信息)
    """

    # 使用全局变量dfkcjl
    global dfkcjl
    
    # 检查全局变量是否已加载，如果未加载则尝试重新加载
    if dfkcjl is None:
        print("全局数据未加载，尝试重新加载...")
        if not load_excel_data():
            return False, '数据未加载，无法查询学生信息'
    
    df = dfkcjl

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
            return True, norm_to_orig[input_norm]
        # try case-insensitive contains in original names
        for n in names:
            if input_raw in n:
                return True, n

    # prepare pinyin query
    py_query = re.sub(r'[^a-zA-Z]', '', input_raw).lower()
    # if query empty after removing punctuation, fall back to normalized exact match
    if not py_query:
        if input_norm in norm_to_orig:
            return True, norm_to_orig[input_norm]
        # fallback: fuzzy compare normalized names
        best = None
        best_score = 0.0
        for norm, orig in norm_to_orig.items():
            score = SequenceMatcher(None, input_norm.lower(), norm.lower()).ratio()
            if score > best_score:
                best_score = score; best = orig
        if best_score >= 0.6:
            return True, best
        return False, f'未能识别名字：{name}. 建议检查拼写或使用中文姓名。'

    # exact pinyin match
    if py_query in pinyin_index:
        # if multiple names share same pinyin, return the first (could be refined)
        return True, pinyin_index[py_query][0]

    # fuzzy pinyin match: compute best ratio against all pinyin keys
    best = None
    best_score = 0.0
    for py_key, orig_list in pinyin_index.items():
        score = SequenceMatcher(None, py_query, py_key).ratio()
        if score > best_score:
            best_score = score; best = (py_key, orig_list)

    if best is None:
        return False, f'未找到候选学生。'

    # if score is reasonable, return first original name of best pinyin key
    py_key, orig_list = best
    if best_score >= 0.5:
        return True, orig_list[0]
    # else return message with top few suggestions
    # build top3 suggestions
    scored = []
    for py_key, orig_list in pinyin_index.items():
        scored.append((SequenceMatcher(None, py_query, py_key).ratio(), orig_list[0]))
    scored.sort(reverse=True)
    suggestions = [s for _, s in scored[:5]]
    return False, f'未能精确匹配"{name}"。最接近的候选: {"、".join(suggestions)} (相似度: {best_score:.2f})'

@tool
def find_batago_student(name:str) -> str:
    """
    Search for the corresponding student names in the Excel dataset through Chinese, English or pinyin.
    学生在倍塔狗也就是batago上学，用PBL方法学习STEM

    Args:
        name (str): The name string to search for.
    """
    # 调用内部函数
    isfound, result = find_the_student(name)
    if isfound:
        return result + ":是倍塔狗人工智能的学生"
    else:
        return "没找到"+result
  


@tool
def generate_summary(student_name: str, time: str) -> str:
    """
    根据 学了什么 学习内容 学习 等关键词触发，得到学生的情况汇总
    Read the table information and generate a summary of all the course content for the given student within the specified time period accordingly.
    Args:
        student_name (str): The name of the student to query，学生姓名.
        time (str): Time period signals, e.g., "上个季度", “最近”，“近期”,"最近一周", "今年", etc.
    Returns:
        str: A summary of the student's learning situation.
    """
    # 首先调用find_batago_student函数验证并获取正确的学生姓名
    print("generate_summary", student_name, time)
    isfound, result = find_the_student(student_name)
    if isfound:
        correct_student_name=result
    else:
        return "错误的学生姓名"+student_name
    
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

    def parse_time_signal(ts: str):
        s = (ts or '').strip()
        
        # 空字符串返回默认值
        if not s:
            return ('默认:上个季度', quarter_bounds(now, -1))
        
        # 按优先级检查时间范围，一旦匹配就立即返回
        if re.search(r'最近一周|近一周|上周', s):
            return ('最近一周', (now - datetime.timedelta(days=7), now))
        
        if re.search(r'最近三个月|近三个月|过去三个月', s):
            return ('最近三个月', (now - datetime.timedelta(days=90), now))
        
        if re.search(r'近[一二三四五六七八九十]?天', s):
            return ('最近几天', (now - datetime.timedelta(days=7), now))
        
        if re.search(r'今年', s):
            return ('今年', (datetime.datetime(now.year,1,1), now))
        
        if re.search(r'这一季度|本季度|这季度', s):
            st, _ = quarter_bounds(now, 0)
            return ('这一季度', (st, now))
        
        if re.search(r'上个季度|上一季度|上季度', s):
            return ('上个季度', quarter_bounds(now, -1))
        
        if re.search(r'上个月|上月', s):
            y = now.year; m = now.month - 1
            if m == 0: m = 12; y -= 1
            if m in [1,3,5,7,8,10,12]: last = 31
            elif m in [4,6,9,11]: last = 30
            else: last = 29 if (y%4==0 and (y%100!=0 or y%400==0)) else 28
            return ('上个月', (datetime.datetime(y,m,1), datetime.datetime(y,m,last)))
        
        y = re.search(r'(20\d{2})', s)
        if y:
            yy = int(y.group(1))
            return (f'{yy}年', (datetime.datetime(yy,1,1), datetime.datetime(yy,12,31)))
        
        # 检查一般的"最近"、"近几"或"近期"
        if re.search(r'最近|近几|近期', s):
            return ('最近(30天)', (now - datetime.timedelta(days=30), now))
        
        # 如果没有任何匹配，返回默认值
        return ('默认(90天)', (now - datetime.timedelta(days=90), now))

    # 使用全局变量dfkcjl
    global dfkcjl
    
    # 检查全局变量是否已加载，如果未加载则尝试重新加载
    if dfkcjl is None:
        print("全局数据未加载，尝试重新加载...")
        if not load_excel_data():
            return '数据未加载，无法生成学生摘要'
    
    df = dfkcjl.copy()
    df['学生姓名_标准'] = df['学生姓名'].astype(str).apply(normalize_name)
    target = normalize_name(correct_student_name)
    student_df = df[df['学生姓名_标准'] == target]
    print("generate_summary", target)
    
    if student_df.empty:
        candidates = list(df['学生姓名'].dropna().astype(str).unique()[:200])
        return f'未找到学生 {correct_student_name} 的记录。数据中可选学生（最多200个显示）: {"、".join(candidates)}'

    # 获取时间信号并设置mapping变量
    try:
        signal = parse_time_signal(time)
        sname, (start, end) = signal
        print("generate_summary signal", sname)
        print("generate_summary start", start, "end", end)
        # 定义mapping变量，与langcchaincourse.py保持一致的格式
        mapping = [f"{sname} -> {start.date()} ~ {end.date()}"]
    except Exception as e:
        print(f"时间解析错误: {e}")
        # 设置默认值以确保mapping变量始终被定义
        mapping = ["未知时间范围"]
        start = datetime.datetime.now() - datetime.timedelta(days=90)
        end = datetime.datetime.now()
    sel = student_df[(student_df['上课时间'] >= start) & (student_df['上课时间'] <= end)]
    if sel.empty:
        sel = student_df

    contents = sel['内容'].dropna().astype(str).tolist()
    if not contents:
        return f"{correct_student_name} 没有课程'内容'进行分析。"

    # limit length
    # 学生最多上200次课，每次课课程记录大约300个字符
    max_chars = 200*300
    joined = ''
    count = 0
    for t in contents:
        if count + len(t) > max_chars:
            break
        joined += t + ' '
        count += len(t) + 1

    print(f"generate_summary {correct_student_name} {time} {joined[:200]}")
    try:
        llmsumary = ChatOllama(model="qwen3-vl:235b-cloud",temperature=0.5)
        result = llmsumary.invoke(f"请把下面多次课程记录的内容合并并生成不超过800字的中文学习总结：{joined}")
        ai_summary = result.content
        return f"时间映射：{'; '.join(mapping)}\n摘要（{correct_student_name}）：{ai_summary}"
    except Exception as e:
        error_msg = str(e)
        if "Service Temporarily Unavailable" in error_msg or "503" in error_msg:
            return "AI服务暂时不可用，请稍后再试。"
        return f"生成摘要时出错：{error_msg}"

@tool
def career_planning(student_name: str, career_target: str) -> str:
    """
    根据学生姓名和职业目标，阅读上课反馈xlsx文件,总结学生学习了什么内容，根据职业生涯目标给出后续课程建议
    职业目标关键词：我想当 我想成为 我的理想是 我准备 等
    Args:
        student_name (str): The name of the student to query.
        career_target (str): 职业目标，根据我想当 我想成为 我的理想是 我准备 等关键词获得
    Returns:
        str: 后续课程建议，30小时课程计划
    """

    print("career_planning", student_name)
    isfound, result = find_the_student(student_name)
    if isfound:
        correct_student_name=result
    else:
        return "错误的学生姓名"+student_name
    # 使用全局变量dfkcjl
    global dfkcjl
    
    # 检查全局变量是否已加载，如果未加载则尝试重新加载
    if dfkcjl is None:
        print("全局数据未加载，尝试重新加载...")
        if not load_excel_data():
            return '数据未加载，无法生成职业规划'
    
    df = dfkcjl.copy()
    df['学生姓名_标准'] = df['学生姓名'].astype(str).apply(normalize_name)
    target = normalize_name(correct_student_name)
    student_df = df[df['学生姓名_标准'] == target]
    if student_df.empty:
        candidates = list(df['学生姓名'].dropna().astype(str).unique()[:200])
        return f'未找到学生 {correct_student_name} 的记录。数据中可选学生（最多200个显示）: {"、".join(candidates)}'
    
    sel = student_df
    contents = sel['内容'].dropna().astype(str).tolist()
    if not contents:
        return f"{correct_student_name} 没有课程'内容'进行分析。"
    
    #在sel里面找到第一次上课时间和最近一次上课时间
    first_class_time=sel['上课时间'].min().strftime('%Y年%m月%d日') if not sel['上课时间'].empty else '无记录'
    last_class_time=sel['上课时间'].max().strftime('%Y年%m月%d日') if not sel['上课时间'].empty else '无记录'
    # 获取所有上课时间列表并生成紧凑型日期列表
    sksjs=sel['上课时间'].dropna().astype(str).tolist()
    # 生成紧凑型日期列表，格式为YYMMDD
    # 解析日期函数，用于排序
    def parse_date(date_str):
        """解析日期字符串为datetime对象以便排序"""
        try:
            # 假设日期格式为 YYYY-MM-DD
            if len(date_str) >= 10:
                year = int(date_str[:4])
                month = int(date_str[5:7])
                day = int(date_str[8:10])
                return (year, month, day)
        except:
            pass
        return (0, 0, 0)  # 解析失败时返回默认值
    
    # 对日期进行排序（从近到远，降序）
    sorted_dates = sorted(sksjs, key=lambda x: parse_date(x.split()[0]), reverse=True)
    
    # 生成排序后的紧凑日期字符串
    compact_dates=' '.join([date.split()[0][2:4]+date.split()[0][5:7]+date.split()[0][8:10] if len(date.split()[0]) >= 10 else date.split()[0] for date in sorted_dates])
    #根据sel计算总课时数，进行数据清洗移除异常值
    try:
        # 尝试将课时消耗转换为浮点数
        valid_hours = pd.to_numeric(sel['课时消耗'], errors='coerce')
        # 过滤掉异常值：负值
        valid_hours = valid_hours[valid_hours >= 0]
        # 计算有效课时总和
        total_class_time = valid_hours.sum()
        # 计算有效课次数（行数）
        class_count = len(valid_hours[valid_hours > 0])
    except Exception as e:
        # 如果转换失败，使用原始求和
        total_class_time = sel['课时消耗'].sum()
        # 计算原始课次数
        class_count = len(sel[sel['课时消耗'] > 0])

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
        result = llm.invoke(f'''你是一个STEM教育顾问，首先根学生已经学习过{total_class_time}课时内容和特点总结，然后规划后续30小时课程内容。
                        针对的学生职业目标是{career_target}
                        要求在1.构造 2.电路 3.编程 4.智能 5.设计 6.整合 7.创新的七个维度规划课程，
                        在学生已有学习内容基础上，保持平衡，预留须继续学习的空间，引导学生持续续课学习至少预期达到120小时。
                        学生已经学习过的内容如下：
                        {joined} 
                        请写出课程计划并给出计划制定的理由。
                        计划用markdown格式输出
                        '''
                        )
        markdown_content = result.content
        
        # 删除markdown_content中的一级标题（如果存在）
        lines = markdown_content.split('\n')
        filtered_lines = [line for line in lines if not line.strip().startswith('# ')]
        markdown_content = '\n'.join(filtered_lines)
        
        # 创建PDF保存目录
        pdf_dir = "career_plans"
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
        
        # 在生成新文件前，删除该学生的旧文件，只保留最新的
        print(f"清理学生 {correct_student_name} 的旧文件...")
        clean_result = clean_old_student_files(correct_student_name)
        print(f"清理结果: 删除了 {clean_result['deleted_count']} 个旧文件")
        
        # 生成Word文件名（使用学生姓名和当前时间）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{normalize_name(correct_student_name).replace(' ', '_')}_{timestamp}.docx"
        docx_path = os.path.join(pdf_dir, safe_filename)
        
        # 检查pandoc是否安装
        try:
            # 检查pandoc命令行工具是否可用
            subprocess.run(['pandoc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            pandoc_available = True
            print(f"Pandoc检查成功")
        except (subprocess.SubprocessError, FileNotFoundError):
            pandoc_available = False
            print(f"Pandoc命令行工具不可用")
        
        try:
            # 在markdown内容前添加标题和元信息
            enhanced_markdown = f"""# 倍塔狗人工智能课程规划 - {correct_student_name}

## 基本信息

**目标职业:** {career_target}  
**生成时间:** {datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}  
**第一次课时间:**{first_class_time}  
**最后一次课时间:**{last_class_time}  
**课次:**{class_count}  
**课时消耗:**{total_class_time}  
<small>日期:{compact_dates}</small>

---

{markdown_content}
"""
            
            # 使用pypandoc将markdown转换为docx
            # 首先将markdown内容写入临时文件
            temp_md_path = os.path.join(pdf_dir, f"temp_{timestamp}.md")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(temp_md_path), exist_ok=True)
            
            try:
                with open(temp_md_path, 'w', encoding='utf-8') as f:
                    f.write(enhanced_markdown)
                print(f"临时Markdown文件已创建: {temp_md_path}")
            except Exception as file_error:
                raise Exception(f"创建临时Markdown文件失败: {str(file_error)}")
            
            # 使用pypandoc将markdown文件转换为docx
            try:
                # 设置额外的参数以优化转换，包括设置字体为宋体和A4纸张格式
                extra_args = [
                    '--standalone',
                    '--from=markdown+smart',
                    '--to=docx',
                    '--wrap=none',
                    '--variable=mainfont="SimSun"',  # 设置主要字体为宋体
                    '--variable=fontfamily="SimSun"',  # 设置字体家族为宋体
                    '--variable=fontsize=12pt'  # 设置字号
                ]
                
                # 检查是否存在template.docx文件作为参考文档（确保A4纸张格式）
                template_path = os.path.join(os.path.dirname(__file__), 'template.docx')
                if os.path.exists(template_path):
                    extra_args.append(f'--reference-doc={template_path}')
                    print(f"使用参考文档确保A4纸张格式: {template_path}")
                
                pypandoc.convert_file(
                    temp_md_path,
                    'docx',
                    outputfile=docx_path,
                    extra_args=extra_args
                )
                print(f"Word文档转换成功: {docx_path}")
            except Exception as convert_error:
                # 检查是否是因为pandoc本身的问题
                if 'pandoc' in str(convert_error).lower():
                    raise Exception(f"Pandoc转换失败: {str(convert_error)}。请确保已正确安装pandoc。")
                else:
                    raise Exception(f"文档转换过程中出错: {str(convert_error)}")
            finally:
                # 确保无论如何都删除临时文件
                try:
                    if os.path.exists(temp_md_path):
                        os.remove(temp_md_path)
                        print(f"临时文件已删除: {temp_md_path}")
                except Exception as cleanup_error:
                    print(f"清理临时文件时出错: {str(cleanup_error)}")
            
            # 验证生成的文件是否存在且非空
            if os.path.exists(docx_path) and os.path.getsize(docx_path) > 0:
                docx_status = f"Word文档已成功生成并保存至: {docx_path}"
                print(docx_status)
                
                # 生成同名的PDF文件
                pdf_path = os.path.splitext(docx_path)[0] + '.pdf'
                pdf_status = ""
                try:
                    # 使用pypandoc将markdown转换为PDF
                    extra_args_pdf = [
                        '--standalone',
                        '--from=markdown+smart',
                        '--to=pdf',
                        '--wrap=none'
                    ]
                    
                    pypandoc.convert_file(
                        temp_md_path,
                        'pdf',
                        outputfile=pdf_path,
                        extra_args=extra_args_pdf
                    )
                    
                    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                        pdf_status = f"PDF文档已成功生成并保存至: {pdf_path}"
                        print(pdf_status)
                    else:
                        pdf_status = f"警告: 生成的PDF文档可能为空或未正确创建: {pdf_path}"
                        print(pdf_status)
                        
                except Exception as pdf_error:
                    error_msg = str(pdf_error)
                    # 检查是否是pdflatex缺失的错误
                    if "pdflatex not found" in error_msg.lower() or "exitcode \"47\"" in error_msg:
                        pdf_status = f"注意：PDF文档生成失败，因为系统中缺少pdflatex。Word文档已成功生成。要生成PDF，请安装LaTeX或选择不同的PDF引擎。"
                        print(pdf_status)
                    else:
                        pdf_status = f"生成PDF文档时出错: {error_msg}"
                        print(pdf_status)
                    # 如果PDF文件存在但可能不完整，删除它
                    if os.path.exists(pdf_path):
                        try:
                            os.remove(pdf_path)
                            print(f"已删除不完整的PDF文件: {pdf_path}")
                        except:
                            pass
                
                # 返回成功状态和文件路径
                combined_status = f"{docx_status}\n{pdf_status}"
                return {"content": f"{markdown_content}\n\n---\n\n{combined_status}", "file_path": docx_path, "pdf_path": pdf_path if pdf_status.startswith("PDF文档已成功") else None}
            else:
                raise Exception(f"生成的Word文档可能为空或未正确创建: {docx_path}")
                
        except Exception as docx_error:
            error_message = f"生成Word文档时出错: {str(docx_error)}"
            docx_status = error_message
            print(error_message)
            
            # 如果pandoc不可用，提供安装指南
            if not pandoc_available:
                docx_status += "\n\n注意：系统中未检测到pandoc命令行工具。请安装pandoc后重试。"
                docx_status += "\nWindows用户可以从 https://github.com/jgm/pandoc/releases 下载安装程序。"
                docx_status += "\n安装后可能需要重启计算机以更新环境变量。"
        
        # 返回markdown内容和Word文档生成状态（失败情况）
        return {"content": f"{markdown_content}\n\n---\n\n{docx_status}", "file_path": None}
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
                    print(f"调用工具: {tool_obj}, 输入: {tool_input}")
                    
                    # 确保使用正确的方式调用工具
                    if hasattr(tool_callable, 'invoke'):
                        # 如果有invoke方法，使用invoke
                        tool_result = tool_callable.invoke(tool_input)
                    else:
                        # 否则尝试直接调用函数
                        if isinstance(tool_input, dict):
                            tool_result = tool_callable(**tool_input)
                        else:
                            tool_result = tool_callable(tool_input)
                    
                    # 处理返回字典格式的情况（career_planning函数）
                    if isinstance(tool_result, dict) and 'content' in tool_result and 'file_path' in tool_result:
                        content = tool_result['content']
                        file_path = tool_result['file_path']
                        
                        # 如果有文件路径，添加下载链接信息
                        if file_path and os.path.exists(file_path):
                            file_name = os.path.basename(file_path)
                            # 在Gradio中，文件会自动提供下载链接
                            content += f"\n\n---\n\n📄 **文档下载信息**\n"
                            content += f"- Word文件名: {file_name}\n"
                            content += f"- Word文件保存位置: {file_path}\n"
                            content += f"- Word文件大小: {os.path.getsize(file_path) / 1024:.1f} KB\n"
                            
                            # 添加PDF文件信息（如果有）
                            if 'pdf_path' in tool_result and tool_result['pdf_path'] and os.path.exists(tool_result['pdf_path']):
                                pdf_file_name = os.path.basename(tool_result['pdf_path'])
                                content += f"- PDF文件名: {pdf_file_name}\n"
                                content += f"- PDF文件保存位置: {tool_result['pdf_path']}\n"
                                content += f"- PDF文件大小: {os.path.getsize(tool_result['pdf_path']) / 1024:.1f} KB\n"
                            
                            content += f"\n💡 **提示**: 文件已生成在系统的career_plans目录中，您可以直接访问该目录查看和打开文件。"
                        
                        return content
                    else:
                        return tool_result
        else:
            return result.content
    except Exception as e:
        error_msg = str(e)
        if "Service Temporarily Unavailable" in error_msg or "503" in error_msg:
            return "AI服务暂时不可用，请稍后再试。"
        return f"调用AI模型时出错：{error_msg}"

import os
import subprocess

def open_career_plans_folder():
    """打开career_plans文件夹"""
    try:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建career_plans文件夹路径
        folder_path = os.path.join(current_dir, "career_plans")
        
        # 确保文件夹存在
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return f"文件夹已创建：{folder_path}"
        
        # 根据操作系统打开文件夹
        if os.name == 'nt':  # Windows
            os.startfile(folder_path)
        elif os.name == 'posix':  # macOS or Linux
            subprocess.run(['open', folder_path])  # macOS
        
        return f"正在打开文件夹：{folder_path}"
    except Exception as e:
        return f"打开文件夹失败：{str(e)}"

def open_summary_plans_folder():
    """打开summary_plans文件夹"""
    try:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建summary_plans文件夹路径
        folder_path = os.path.join(current_dir, "summary_plans")
        
        # 确保文件夹存在
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return f"文件夹已创建：{folder_path}"
        
        # 根据操作系统打开文件夹
        if os.name == 'nt':  # Windows
            os.startfile(folder_path)
        elif os.name == 'posix':  # macOS or Linux
            subprocess.run(['open', folder_path])  # macOS
        
        return f"正在打开文件夹：{folder_path}"
    except Exception as e:
        return f"打开文件夹失败：{str(e)}"

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

# 使用Blocks API创建更复杂的界面
demo = gr.Blocks(title="学生发展STEM课程规划助手")

with demo:
    gr.Markdown("# 学生职业规划助手")
    gr.Markdown("欢迎使用倍塔狗人工智能职业规划助手，您可以在这里咨询在倍塔狗在什么时间段学习了什么，给出自己的职业目标，根据职业目标规划后续课程。")
    
    # 添加API密钥设置区域
    with gr.Accordion("API密钥设置", open=False):
        gr.Markdown("请输入您的TAVILY_API_KEY以启用网络搜索功能。设置后将保存到本地配置文件中。")
        api_key_input = gr.Textbox(label="TAVILY_API_KEY", placeholder="在此粘贴您的TAVILY_API_KEY", type="password", interactive=True)
        api_submit_btn = gr.Button("保存API密钥")
        api_status = gr.Textbox(label="API密钥状态", interactive=False)
    
    # 创建聊天界面
    chat_interface = gr.ChatInterface(fn=chat_fn)
    
    # 添加打开文件夹按钮
    with gr.Row():
        open_folder_btn = gr.Button("📁 打开career_plans文件夹")
        open_summary_btn = gr.Button("📁 打开summary_plans文件夹")
        status_message = gr.Textbox(label="操作状态", interactive=False)
    
    # 设置按钮点击事件
    open_folder_btn.click(fn=open_career_plans_folder, outputs=status_message)
    open_summary_btn.click(fn=open_summary_plans_folder, outputs=status_message)
    api_submit_btn.click(fn=set_tavily_api_key, inputs=api_key_input, outputs=api_status)

demo.launch()
