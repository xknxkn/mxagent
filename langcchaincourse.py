# -*- coding: utf-8 -*-


import sys
import io
import datetime
import pandas as pd
import unicodedata
import re

from openai import OpenAI
from typing import List
from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama
from pypinyin import lazy_pinyin
from difflib import SequenceMatcher

# 打印显示中文
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

@tool
def validate_user(user_name: str, addresses: List[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_name (str): 用户名.
        addresses (List[str]): 曾经的住址列表.
    """
    print(f"xkn Validating user {user_name} with addresses: {addresses}")
    addressdb={"张三":"汇腾广场303漕溪北","陈知远":"1404望族城","张三丰":"闵行星创广场"}
    if user_name in addressdb:
        addr=addressdb[user_name]
        for address in addresses:
            if address in addr:
                print(f"xkn User {user_name} validated successfully with address: {address}")
                return True
        print(f"xkn User {user_name} validation failed. No matching addresses found.")
        return False
    print(f"xkn User {user_name} not found in the database.")  
    return False

@tool
def food_by_city(city: str,food_catalog) -> str:
    """Get famous food by city.

    Args:
        city (str): 城市名称.
        food_catalog: 食物种类，只能从以下字符串中选择. 碳水, 肉类, 海鲜, 水果.
    """
    food_db = {
        "郑州": {
            "碳水": ["烩面", "胡辣汤", "羊肉汤"],
            "肉类": ["驴肉", "烤鸭"],
            "海鲜": ["鲤鱼", "鲫鱼"],
            "水果": ["苹果", "梨" ]
        },
        "西安": {
            "碳水": ["肉夹馍", "羊肉泡馍", "凉皮"],
            "肉类": ["棒棒肉", "腊汁肉"],
            "海鲜": ["鲤鱼", "鲫鱼"],
            "水果": ["石榴", "葡萄" ]
        },
        "上海": {
            "碳水": ["小笼包", "生煎包", "蟹粉小笼"],
            "肉类": ["红烧肉", "油爆虾"],
            "海鲜": ["小黄鱼", "大黄鱼", "青鱼"],
            "水果": ["杨梅", "荔枝" ]
        }
    }

@tool
def find_the_student(name:str) -> str:
    """
    Search for the corresponding student names in the Excel dataset through Chinese, English or pinyin.

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
            return norm_to_orig[input_norm]
        # try case-insensitive contains in original names
        for n in names:
            if input_raw in n:
                return n

    # prepare pinyin query
    py_query = re.sub(r'[^a-zA-Z]', '', input_raw).lower()
    # if query empty after removing punctuation, fall back to normalized exact match
    if not py_query:
        if input_norm in norm_to_orig:
            return norm_to_orig[input_norm]
        # fallback: fuzzy compare normalized names
        best = None
        best_score = 0.0
        for norm, orig in norm_to_orig.items():
            score = SequenceMatcher(None, input_norm.lower(), norm.lower()).ratio()
            if score > best_score:
                best_score = score; best = orig
        if best_score >= 0.6:
            return best
        return f'未能识别名字：{name}. 建议检查拼写或使用中文姓名。'

    # exact pinyin match
    if py_query in pinyin_index:
        # if multiple names share same pinyin, return the first (could be refined)
        return pinyin_index[py_query][0]

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
        return orig_list[0]
    # else return message with top few suggestions
    # build top3 suggestions
    scored = []
    for py_key, orig_list in pinyin_index.items():
        scored.append((SequenceMatcher(None, py_query, py_key).ratio(), orig_list[0]))
    scored.sort(reverse=True)
    suggestions = [s for _, s in scored[:5]]
    return f'未能精确匹配"{name}"。最接近的候选: {"、".join(suggestions)} (相似度: {best_score:.2f})'


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

    # normalize name helper
    def normalize_name(name: str) -> str:
        n = unicodedata.normalize('NFKC', str(name))
        n = ''.join(n.split())
        if n.endswith('-综评熊学科'):
            n = n[:-7]
        return n

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
    target = normalize_name(student_name)
    student_df = df[df['学生姓名_标准'] == target]
    if student_df.empty:
        candidates = list(df['学生姓名'].dropna().astype(str).unique()[:200])
        return f'未找到学生 {student_name} 的记录。数据中可选学生（最多200个显示）: {"、".join(candidates)}'

    signals = parse_time_signals(time)
    mapping = [f"{sig} -> {st.date()} ~ {ed.date()}" for sig,(st,ed) in signals]
    start, end = signals[0][1]
    sel = student_df[(student_df['上课时间'] >= start) & (student_df['上课时间'] <= end)]
    if sel.empty:
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
        client = OpenAI(api_key="sk-rVAAAqB0erLvKgwXDftvEfAfSwTFnruyArnusgjnBnnekfwR", base_url="https://api.moonshot.cn/v1")
        prompt = f"请把下面多次课程记录的内容合并并生成不超过200字的中文学习总结：{joined}"
        completion = client.chat.completions.create(
            model='moonshot-v1-8k',
            messages=[
                {"role": "system", "content": "你是助教，负责把多条上课记录的内容合并为简短摘要（中文，<=200字）。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        ai_summary = completion.choices[0].message.content.strip()
    except Exception as e:
        return f"时间映射：{'; '.join(mapping)}\nAI摘要失败：{e}"

    return f"时间映射：{'; '.join(mapping)}\n摘要：{ai_summary[:200]}"

    


llm = ChatOllama(
    model="qwen3-vl:235b-cloud",
    validate_model_on_init=False,
    temperature=0,
).bind_tools([validate_user,food_by_city,generate_summary,find_the_student])

#根据提示词调用llm invoke 然后处理
def llmtool_invoke_tool(str_input: str):
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
                print(tool_result)

'''
llmtool_invoke_tool("告诉诉我陈知远是否在汇腾广场303漕溪北路或1404望族城学习过")
llmtool_invoke_tool("告诉我郑州盛产什么碳水食物")
llmtool_invoke_tool("告诉我西安的著名肉类食物")
llmtool_invoke_tool("上海的著名海鲜食物有哪些")
llmtool_invoke_tool("请告诉我please tell me倪承的上个季度的学习情况")
llmtool_invoke_tool("今天天气不错黄宝儿最近学习怎么样")
llmtool_invoke_tool("tell me whether nili_hae\yan同学在excel里")
llmtool_invoke_tool("zhang1.tian2.yu3在不在表格里")
llmtool_invoke_tool("zhang+shu+维在不在表格里")
llmtool_invoke_tool("yet.pngy.i在不在表格里")
'''