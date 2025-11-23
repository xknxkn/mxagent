# -*- coding: utf-8 -*-
from typing import List

import sys
import io

from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama

# Ensure stdout/stderr use UTF-8 to avoid Chinese garbling on Windows consoles
try:
    # Python 3.7+: TextIOWrapper has reconfigure
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    # Fallback: wrap buffer with TextIOWrapper
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", write_through=True)
    except Exception:
        # If even that fails, ignore and let environment handle it
        pass


# --- Address fuzzy matching utilities (default implementation) ---
import re
import difflib

try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

try:
    import jieba
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False


def _fullwidth_to_halfwidth(s: str) -> str:
    # convert fullwidth chars to halfwidth
    res = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000:
            code = 0x20
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        res.append(chr(code))
    return ''.join(res)


def _normalize(s: str) -> str:
    if s is None:
        return ''
    s = _fullwidth_to_halfwidth(s)
    # basic punctuation removal and lowercasing
    s = s.strip()
    # remove punctuation (keep word chars, whitespace and common CJK range)
    s = re.sub(r"[^\w\s\u4e00-\u9fff]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _contains_chinese(s: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)


def fuzzy_score(a: str, b: str) -> float:
    """Return similarity score 0-100 between two strings.

    Uses `rapidfuzz` when available for better results, falls back to
    `difflib.SequenceMatcher` otherwise. For Chinese text, if `jieba`
    is available we segment before comparing.
    """
    a_n = _normalize(a)
    b_n = _normalize(b)

    if _HAS_RAPIDFUZZ:
        try:
            if _contains_chinese(a_n) or _contains_chinese(b_n):
                if _HAS_JIEBA:
                    a_tok = ' '.join(jieba.lcut(a_n))
                    b_tok = ' '.join(jieba.lcut(b_n))
                    return float(fuzz.token_set_ratio(a_tok, b_tok))
                return float(fuzz.ratio(a_n, b_n))
            # english / token-friendly
            return float(fuzz.WRatio(a_n, b_n))
        except Exception:
            pass

    # fallback
    try:
        return float(int(difflib.SequenceMatcher(None, a_n, b_n).ratio() * 100))
    except Exception:
        return 0.0


def fuzzy_match_address(query: str, address: str, threshold: float = 80.0) -> tuple[float, bool]:
    """Return (score, is_match) comparing `query` and `address`.

    - `score` is in range 0..100
    - `is_match` is True when score >= threshold
    """
    score = fuzzy_score(query or '', address or '')
    return score, (score >= float(threshold))


def find_best_matches(query: str, addresses: list[str], top_n: int = 3) -> list[tuple[str, float]]:
    """Return top_n (address, score) sorted by score desc."""
    scored = [(addr, fuzzy_score(query, addr)) for addr in addresses]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]



@tool
def validate_user(user_name: str, addresses: List[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_name (str): 用户名.
        addresses (List[str]): 曾经的住址列表.
    """
    print(f"xkn Validating user {user_name} with addresses: {addresses}")
    useraddress={"张三":"汇腾广场303漕溪北","陈知远":"1404望族城","张三丰":"闵行星创广场"}
    if user_name in useraddress:
        addr=useraddress[user_name]
        for address in addresses:
            #地址模糊匹配
            print(f"xkn Checking similarity between '{addr}' and '{address}'")
            score, matched = fuzzy_match_address(addr, address, threshold=75)
            print(f"xkn similarity score={score:.1f}, matched={matched}")
            if matched:
                print(f"xkn User {user_name} validated successfully with address: {address}")
                return True

        print(f"xkn User {user_name} validation failed. No matching addresses found.")
        return False
    print(f"xkn User {user_name} not found in the database.")  
    return False


llm = ChatOllama(
    model="qwen3-vl:235b-cloud",
    validate_model_on_init=True,
    temperature=0,
).bind_tools([validate_user])

result = llm.invoke("请告诉我陈知远是否在汇腾广场303漕溪北路或1404望族城学习过"
)   


if isinstance(result, AIMessage) and result.tool_calls:
    print("Tool calls made during the LLM invocation:")
    print("tool call result is",result.tool_calls)

    for call in result.tool_calls:
        print(call)
        # invoke the tool (support both dict-style and object-style tool calls)
        tool_result = None
        if isinstance(call, dict):
            tool_obj = call.get("tool") or call.get("tool_name") or call.get("name")
            tool_input = call.get("tool_input") or call.get("input") or call.get("args") or {}

            # If tool is a string name, try to resolve it from the LLM's bound tools
            if isinstance(tool_obj, str):
                tool_callable = None
                if hasattr(llm, "tools"):
                    tool_callable = llm.tools.get(tool_obj)
                # fallback: try to resolve a global function with that name
                if tool_callable is None:
                    tool_callable = globals().get(tool_obj)
                if tool_callable is None:
                    print(f"Could not resolve tool callable for name: {tool_obj}")
                else:
                    if callable(tool_callable):
                        tool_result = tool_callable(**tool_input)
                    elif hasattr(tool_callable, "invoke"):
                        # Some tool wrappers expect a single 'input' argument rather than **kwargs.
                        try:
                            tool_result = tool_callable.invoke(**tool_input)
                        except TypeError:
                            try:
                                tool_result = tool_callable.invoke(tool_input)
                            except Exception as e:
                                print("Tool invoke failed:", e)
                    else:
                        print(f"Resolved tool is not callable or invokable: {tool_callable}")

            # If tool is already a callable/function
            elif callable(tool_obj):
                tool_result = tool_obj(**(tool_input or {}))

            # If tool is an object with an invoke method
            elif hasattr(tool_obj, "invoke"):
                tool_result = tool_obj.invoke(**(tool_input or {}))

            else:
                print("Unrecognized tool entry in tool call:", tool_obj)

        else:
            # original object-style calls (has attributes)
            tool_result = call.tool.invoke(**call.tool_input)

        print("tool result is", tool_result)
else:
    print("No tool calls were made during the LLM invocation.")
    

