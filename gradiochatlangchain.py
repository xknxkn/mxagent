from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
import gradio as gr
import sys
import io

# 打印显示中文
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

llm = ChatOllama(model="qwen3-vl:235b-cloud", temperature=0)

history_langchain = []
# Add user input to history
history_langchain.append(SystemMessage(content='You are a helpful assistant.'))
history_langchain.append(HumanMessage(content='Hello!'))
history_langchain.append(HumanMessage(content='I am Chenzhiyuan'))
history_langchain.append(HumanMessage(content='I am a student studying STEM courses in BataGo'))
history_langchain.append(HumanMessage(content='BataGo is a Famous STEM course provider in Shanghai China'))
history_langchain.append(HumanMessage(content='who am i'))

# Get response from the model
response = llm.invoke(history_langchain)
print("round 1 --------------------",response.content)

#round 2
history_langchain.append(HumanMessage(content='who am i'))
response = llm.invoke(history_langchain)
print("round 2 -------------------",response.content)

#利用Gradio的history里面的user部分得到对user的对话话记忆生成user上下文
def chat_fn(message, history):
    # Convert Gradio history to LangChain format
    history_langchain = [
    HumanMessage(content=msg['content']) if msg['role'] == "user" else AIMessage(content=msg['content'])
    for msg in history
    ]
    # Add user input to history
    history_langchain.append(HumanMessage(content=message))
    # Get response from the model
    response = llm.invoke(history_langchain)
    return response.content

demo = gr.ChatInterface(fn=chat_fn, title="Echo Bot")
demo.launch()
