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

@tool
def shiwuzhonglei(chengshi: str,bbb:str) -> str:
    """aaa"""
    return "bbb"

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

    if city in food_db and food_catalog in food_db[city]:
        famous_foods = food_db[city][food_catalog]
        return f"{city}的著名{food_catalog}有: " + ", ".join(famous_foods)
    else:
        return f"抱歉，未找到{city}的著名{food_catalog}信息。"

llm = ChatOllama(
    model="qwen3:8b",
    validate_model_on_init=False,
    temperature=0,
).bind_tools([validate_user,food_by_city])

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


llmtool_invoke_tool("告诉诉我陈知远是否在汇腾广场303漕溪北路或1404望族城学习过")
llmtool_invoke_tool("告诉我郑州盛产什么碳水食物")
llmtool_invoke_tool("告诉我西安的著名肉类食物")
llmtool_invoke_tool("上海的著名海鲜食物有哪些")



           
    

