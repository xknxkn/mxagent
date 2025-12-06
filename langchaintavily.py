
from langchain_core.tools import tool
from langchain.messages import AIMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient
import sys, io

# Ensure stdout can output UTF-8 on Windows consoles to avoid 'gbk' encode errors
try:
    if getattr(sys.stdout, "reconfigure", None):
        sys.stdout.reconfigure(encoding="utf-8")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    # fallback: ignore if we cannot reconfigure
    pass

@tool
def tavily_search(query: str) -> str:
    """Use this tool to search the web for recent information."""
    client = TavilyClient("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")
    response = client.search(
        query=query,
        max_results=1
    )
    print("response of tavily search is",response)
    return response['results'][0]["content"]

tools_list = [tavily_search]

llm = ChatOllama(model="qwen3-vl:235b-cloud", temperature=0).bind_tools(tools_list)

#根据提示词调用llm invoke 然后处理
def llmtool_invoke_tool(str_input: str):
    result = llm.invoke(str_input)
    if isinstance(result, AIMessage) and result.tool_calls:
        for call in result.tool_calls:
            tool_result = None
            if isinstance(call, dict):
                tool_obj = call.get("tool") or call.get("tool_name") or call.get("name")
                tool_input = call.get("tool_input") or call.get("input") or call.get("args") or {}
                tool_callable = globals().get(tool_obj)
                tool_result = tool_callable.invoke(tool_input)
                print(tool_result)



result = llm.invoke("Hi, how are you?")
print(result,"\n\n")
llmtool_invoke_tool("请帮我搜索一下上海今天天气如何")







print(result,"\n\n")

