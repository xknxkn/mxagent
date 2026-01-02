# -*- coding: utf-8 -*-


import sys
import io
import datetime
import pandas as pd
import unicodedata
import re
import glob

from openai import OpenAI
from typing import List, Dict, Any, TypedDict
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langgraph.prebuilt import tool_node

# 打印显示中文
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

@tool
def validate_user(user_name: str, addresses: List[str]) -> bool:
    """Validate user by checking if their name exists in the student Excel files.

    Args:
        user_name (str): 用户名.
        addresses (List[str]): 曾经的住址列表（此参数现已忽略）.
    """
    print(f"xkn Validating user {user_name}")
    
    try:
        # 调用find_the_student工具获取所有学生姓名
        all_students = find_the_student()
        
        # 检查用户名是否在所有学生姓名中
        if user_name in all_students:
            print(f"xkn User {user_name} validated successfully - found in student list")
            return True
        else:
            print(f"xkn User {user_name} validation failed - not found in student list")
            return False
            
    except Exception as e:
        print(f"xkn User validation error: {e}")
        return False

@tool
def food_by_city(city: str, food_catalog: str) -> str:
    """Get famous food by city.

    Args:
        city (str): 城市名称.
        food_catalog (str): 食物种类，只能从以下字符串中选择. 碳水, 肉类, 海鲜, 水果.
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
        return f"{city}的{food_catalog}有：{','.join(food_db[city][food_catalog])}"
    else:
        return f"未找到{city}的{food_catalog}信息"

@tool
def summarize_my_class(current_user: str = None) -> str:
    """Summarize class content for the current student from Excel files starting with '上课反馈'.
    This tool automatically uses the currently logged-in student's name.
    
    Args:
        current_user (str, optional): 当前用户的姓名，如果不提供，将尝试从上下文中获取.
    """
    try:
        if not current_user:
            return '无法获取当前用户信息，请先登录'
        
        # 获取学生的上课内容
        class_content = get_student_class_content(current_user)
        
        # 如果没有找到内容，直接返回
        if class_content.startswith('未找到'):
            return class_content
        
        # 创建专门用于总结的LLM实例，设置不同的system prompt
        summary_llm = ChatOllama(
            model="qwen3-vl:235b-cloud",
            validate_model_on_init=False,
            temperature=0,
            system_prompt="你是一个专业的课程总结助手，请将用户提供的上课反馈内容总结为1000字以内的markdown格式文字。总结要清晰、有条理，突出重点内容。",
        )
        
        # 调用LLM生成总结
        prompt = f"请将以下学生的上课反馈内容总结为1000字以内的markdown格式文字：\n\n{class_content}"
        summary_result = summary_llm.invoke([HumanMessage(content=prompt)])
        
        return summary_result.content
        
    except Exception as e:
        return f'生成课程总结时出错: {e}'


def find_the_student() -> str:
    """
    Extract all student names from all Excel files starting with '上课反馈' in the current directory.
    """
    
    try:
        # 查找所有以上课反馈开头的xlsx文件
        files = glob.glob('上课反馈*.xlsx')
        
        if not files:
            return '未找到以上课反馈开头的Excel文件'
        
        all_students = set()
        
        # 遍历所有找到的文件
        for file in files:
            # 读取Excel文件
            df = pd.read_excel(file)
            
            # 检查是否有学生姓名字段
            if '学生姓名' in df.columns:
                # 提取学生姓名，去除重复值
                students = df['学生姓名'].dropna().astype(str).unique()
                all_students.update(students)
        
        if not all_students:
            return '未在文件中找到学生姓名'
        
        # 将学生姓名按逗号分隔返回
        return '、'.join(sorted(all_students))
        
    except Exception as e:
        return f'处理文件时出错: {e}'

def get_student_class_content(student_name: str) -> str:
    """
    Get all class content for a specific student from all Excel files starting with '上课反馈' in the current directory.
    
    Args:
        student_name (str): The name of the student.
        
    Returns:
        str: All class content for the student, concatenated into a single string.
    """
    
    try:
        # 查找所有以上课反馈开头的xlsx文件
        files = glob.glob('上课反馈*.xlsx')
        
        if not files:
            return '未找到以上课反馈开头的Excel文件'
        
        all_content = []
        
        # 遍历所有找到的文件
        for file in files:
            # 读取Excel文件
            df = pd.read_excel(file)
            
            # 检查是否有学生姓名和内容字段
            if '学生姓名' in df.columns and '内容' in df.columns:
                # 筛选出该学生的内容
                student_rows = df[df['学生姓名'].astype(str) == student_name]
                # 提取内容并添加到列表中
                for content in student_rows['内容'].dropna().astype(str):
                    if content.strip():
                        all_content.append(content.strip())
        
        if not all_content:
            return f'未找到学生{student_name}的上课反馈内容'
        
        # 将所有内容合并为一个字符串，每个内容之间用换行符分隔
        return '\n'.join(all_content)
        
    except Exception as e:
        return f'处理文件时出错: {e}'






# 定义AgentState，用于跟踪对话历史
class AgentState(TypedDict):
    messages: List[Any]  # 存储对话消息列表
    current_user: str  # 存储当前验证的用户名

# 初始化LLM并绑定工具，添加system_prompt确保工具调用
llm = ChatOllama(
    model="qwen3-vl:235b-cloud",
    validate_model_on_init=False,
    temperature=0,
    system_prompt="你是一个助手，需要根据用户请求调用相应的工具来获取信息。请直接根据用户请求调用合适的工具，不要问任何问题。必须使用提供的工具来完成任务，不能直接回答用户的问题。",
).bind_tools([validate_user, food_by_city, summarize_my_class])

# 创建自定义工具节点，用于传递当前用户信息
class CustomToolNode:
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}
    
    def __call__(self, state):
        messages = state["messages"]
        current_user = state["current_user"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            results = []
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # 如果是summarize_my_class工具，自动添加current_user参数
                if tool_name == "summarize_my_class":
                    tool_args["current_user"] = current_user
                
                # 执行工具
                tool_result = self.tools[tool_name].invoke(tool_args)
                
                # 创建工具结果消息
                from langchain_core.messages import ToolMessage
                tool_message = ToolMessage(
                    content=str(tool_result),
                    name=tool_name,
                    tool_call_id=tool_call["id"]
                )
                results.append(tool_message)
            
            return {
                "messages": messages + results,
                "current_user": current_user
            }
        return {
            "messages": messages,
            "current_user": current_user
        }

# 创建工具节点
tools = [validate_user, food_by_city, summarize_my_class]
tool_executor = CustomToolNode(tools)

# 创建代理节点
def agent(state: AgentState):
    messages = state["messages"]
    current_user = state["current_user"]
    
    # 调用LLM生成响应，无论是否有工具执行结果
    result = llm.invoke(messages)
    return {"messages": messages + [result], "current_user": current_user}

# 定义条件边函数，用于决定流程是继续执行工具调用还是结束
def should_continue(state: AgentState):
    messages = state["messages"]
    
    # 检查最后一条消息是否有工具调用
    last_message = messages[-1] if messages else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool"
    
    # 否则结束
    return "end"

# 创建状态图
graph_builder = StateGraph(AgentState)

# 添加节点
graph_builder.add_node("agent", agent)  # 代理节点

graph_builder.add_node("tool", tool_executor)  # 工具节点

# 设置入口点
graph_builder.set_entry_point("agent")

# 添加边
graph_builder.add_edge("tool", "agent")  # 工具执行完成后返回代理

graph_builder.add_conditional_edges(
    "agent",  # 从代理节点出发
    should_continue,  # 条件函数
    {"tool": "tool", "end": "__end__"}  # 根据条件返回结果决定下一个节点
)

# 编译状态图
graph = graph_builder.compile()

# 实现llmtool_invoke_tool函数，使用graph.invoke()替代直接LLM调用
def llmtool_invoke_tool(str_input: str, current_user: str):
    # 设置初始状态，包含用户输入的HumanMessage和当前用户信息
    initial_state = {
        "messages": [HumanMessage(content=str_input)],
        "current_user": current_user
    }
    # 使用graph.invoke()调用状态图
    result = graph.invoke(initial_state, config={"recursion_limit": 5})
    
    # 从结果中提取最终答案
    final_answer = None
    # 先查找工具执行结果（ToolMessage）
    for msg in reversed(result["messages"]):
        if hasattr(msg, "name") and hasattr(msg, "tool_call_id") and hasattr(msg, "content"):
            final_answer = msg.content
            break
    # 如果没有工具执行结果，查找AI消息
    if not final_answer:
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
                final_answer = msg.content
                break
    return final_answer or "未能生成有效的响应"


# 测试用例
def test_agent_functionality():
    print("=== 测试LangGraph Agent功能 ===")
    
    # 测试1: 查询郑州的碳水食物 - 使用自然语言
    print("\n1. 测试food_by_city工具 - 查询郑州的碳水食物：")
    result1 = llmtool_invoke_tool("查询郑州的碳水食物", "")
    print(f"结果: {result1}")
    
    # 测试2: 验证用户陈知远的住址 - 使用自然语言
    print("\n2. 测试validate_user工具 - 验证用户陈知远的住址：")
    result2 = llmtool_invoke_tool("验证用户陈知远，他的住址是1404望族城", "")
    print(f"结果: {result2}")
    
    # 测试3: 查询西安的著名肉类食物 - 使用自然语言
    print("\n3. 测试food_by_city工具 - 查询西安的著名肉类食物：")
    result3 = llmtool_invoke_tool("查询西安的著名肉类食物", "")
    print(f"结果: {result3}")
    
    # 测试4: 直接调用find_the_student函数 - 查找所有学生姓名
    print("\n4. 直接测试find_the_student函数 - 查找所有学生姓名：")
    result4 = find_the_student()
    print(f"结果: {result4[:100]}...")  # 只显示前100个字符
    
    # 测试5: 测试课程总结功能
    print("\n5. 测试summarize_my_class工具 - 获取陈知远的课程总结：")
    result5 = llmtool_invoke_tool("我的上课内容", "陈知远")
    print(f"结果: {result5[:200]}...")  # 只显示前200个字符
    
    print("\n=== 测试完成 ===")

# 如果直接运行此文件，创建交互式会话
def main():
    print("=== LangGraph Agent 交互式会话 ===")
    print("请输入您的姓名进行验证：")
    
    # 用户验证阶段
    is_validated = False
    current_user = None
    
    while not is_validated:
        user_name = input("姓名: ").strip()
        
        if not user_name:
            print("姓名不能为空，请重新输入")
            continue
        
        # 调用validate_user函数进行验证，使用空地址列表（因为addresses参数已被忽略）
        validation_result = validate_user.invoke({"user_name": user_name, "addresses": []})
        
        if validation_result:
            is_validated = True
            current_user = user_name
            print(f"验证成功！欢迎您，{current_user}。")
            print("\n您可以提问关于城市食物的问题，或者与您相关的问题。")
            print("当您想结束会话时，请输入'再见'。")
        else:
            print("验证失败，请检查您的姓名是否正确。")
            retry = input("是否重试？(y/n): ").strip().lower()
            if retry != 'y':
                print("会话结束，谢谢使用！")
                return
    
    # 会话阶段
    print("\n=== 开始会话 ===")
    
    while True:
        user_input = input(f"{current_user} > ").strip()
        
        # 检查是否结束会话
        if user_input in ['再见', '拜拜', 'exit', 'quit']:
            print("会话结束，谢谢使用！")
            break
        
        if not user_input:
            continue
        
        # 使用llmtool_invoke_tool处理用户请求，传递当前用户信息
        try:
            result = llmtool_invoke_tool(user_input, current_user)
            print(f"AI > {result}")
        except Exception as e:
            print(f"处理请求时发生错误: {e}")
        
        print()  # 输出空行，提高可读性

if __name__ == "__main__":
    main()
