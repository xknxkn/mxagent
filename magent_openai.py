print("hello1")
from openai import OpenAI
import json

client = OpenAI(
    api_key="sk-jUE8vi5id8DmGcnxpF95SEGIf8CGFyH5xpJjiFF2vsVy5cN7",
    base_url="https://api.moonshot.cn/v1",
)

print("hello2")

def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

print("hello3")

def run_agent_with_tools(user_query):
    # First, let the LLM decide if it needs to use tools
    messages = [
        {"role": "system", "content": """You are an AI assistant that can use tools. 
         Available tools: 
         - search: for searching information
         - get_weather: for getting weather information
         
         If you need to use a tool, respond with JSON format: 
         {"tool": "tool_name", "parameters": {"param1": "value1"}}
         Otherwise, respond normally."""},
        {"role": "user", "content": user_query}
    ]
    
    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=messages,
        temperature=0
    )
    
    response_content = response.choices[0].message.content
    
    # Check if the response is a tool call
    try:
        tool_call = json.loads(response_content)
        if "tool" in tool_call:
            tool_name = tool_call["tool"]
            parameters = tool_call["parameters"]
            
            if tool_name == "search":
                result = search(parameters.get("query", ""))
            elif tool_name == "get_weather":
                result = get_weather(parameters.get("location", ""))
            else:
                result = "Unknown tool"
                
            # Send the result back to the LLM
            messages.append({"role": "assistant", "content": response_content})
            messages.append({"role": "user", "content": f"Tool result: {result}"})
            
            final_response = client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=messages,
                temperature=0
            )
            return final_response.choices[0].message.content
    except json.JSONDecodeError:
        # Not a tool call, return the response directly
        pass
    
    return response_content

print("hello4")

# Use the agent
result = run_agent_with_tools("What's the weather in Shanghai?")
print(result)