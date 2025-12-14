from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from langchain.tools import tool
import os
from tavily import TavilyClient
from typing import List
from pypinyin import lazy_pinyin
from difflib import SequenceMatcher

import unicodedata
import gradio as gr
import sys
import io
import datetime
import pandas as pd
import unicodedata
import re

# 打印显示中文
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

def tavily_search(query: str) -> str:
    """Use this tool to search the web for recent information."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "TAVILY_API_KEY 未配置，请先在 UI 中设置后重试。"
    client = TavilyClient(api_key)
    response = client.search(
        query=query,
        max_results=1
    )
    print("response of tavily search is",response)
    return response['results'][0]["content"]

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
            return norm_to_orig[input_norm]+"是倍塔狗人工智能的学生，欢迎"
        # try case-insensitive contains in original names
        for n in names:
            if input_raw in n:
                return n+"是倍塔狗人工智能的学生，欢迎"

    # prepare pinyin query
    py_query = re.sub(r'[^a-zA-Z]', '', input_raw).lower()
    # if query empty after removing punctuation, fall back to normalized exact match
    if not py_query:
        if input_norm in norm_to_orig:
            return norm_to_orig[input_norm]+"是倍塔狗人工智能的学生，欢迎"
        # fallback: fuzzy compare normalized names
        best = None
        best_score = 0.0
        for norm, orig in norm_to_orig.items():
            score = SequenceMatcher(None, input_norm.lower(), norm.lower()).ratio()
            if score > best_score:
                best_score = score; best = orig
        if best_score >= 0.6:
            return best+"是倍塔狗人工智能的学生，欢迎"
        return f'未能识别名字：{name}. 建议检查拼写或使用中文姓名。'

    # exact pinyin match
    if py_query in pinyin_index:
        # if multiple names share same pinyin, return the first (could be refined)
        return pinyin_index[py_query][0]+"是倍塔狗人工智能的学生，欢迎"

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
        return orig_list[0]+"是倍塔狗人工智能的学生，欢迎"
    # else return message with top few suggestions
    # build top3 suggestions
    scored = []
    for py_key, orig_list in pinyin_index.items():
        scored.append((SequenceMatcher(None, py_query, py_key).ratio(), orig_list[0]))
    scored.sort(reverse=True)
    suggestions = [s for _, s in scored[:5]]
    return f'未能精确匹配"{name}"。最接近的候选: {"、".join(suggestions)} (相似度: {best_score:.2f})'

tools_list = [get_weather,get_batago_school_address,find_batago_student]

llm = ChatOllama(model="qwen3-vl:235b-cloud", temperature=0).bind_tools(tools_list)

#根据提示词调用llm invoke 然后处理, 没找到tool的显示回答的问题
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
                return tool_result
    else:
        return result.content

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
