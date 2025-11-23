# -*- coding: utf-8 -*-
from typing import List

import sys
import io

from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama


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


llm = ChatOllama(
    model="qwen3-vl:235b-cloud",
    validate_model_on_init=False,
    temperature=0,
).bind_tools([validate_user])

result = llm.invoke("请告诉我陈知远是否在汇腾广场303漕溪北路或1404望族城学习过"
)   


if isinstance(result, AIMessage) and result.tool_calls:
    for call in result.tool_calls:
        tool_result = None
        if isinstance(call, dict):
            tool_obj = call.get("tool") or call.get("tool_name") or call.get("name")
            tool_input = call.get("tool_input") or call.get("input") or call.get("args") or {}
            tool_callable = globals().get(tool_obj)
            tool_result = tool_callable.invoke(tool_input)
            print(tool_result)
                
           
    

