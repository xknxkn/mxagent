import gradio as gr
import sys
import io
import ollama

# 打印显示中文
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

#原理演示
#完整的上下文
response = ollama.chat(model="qwen3-vl:235b-cloud", messages
                       =[{'role': 'system', 'content': 'You are a helpful assistant.'},
                         {'role': 'user', 'content': 'Hello!'},
                         {'role': 'user', 'content': 'I am Chenzhiyuan'},
                         {'role': 'user', 'content': 'I am a student studing STEM courses in BataGo'},
                         {'role': 'user', 'content': 'BataGo is a famouse STEM course provider in Shanghai China'},
                         {'role': 'user', 'content': 'who am I'}
                         ])
print("round 1-----------------",response['message']['content'])

#再次运行后上下文丢失
response = ollama.chat(model="qwen3-vl:235b-cloud", messages
                       =[{'role': 'system', 'content': 'You are a helpful assistant.'},
                         {'role': 'user', 'content': 'who am I'}
                         ])
print("round 2---------------",response['message']['content'])

#利用Gradio的history里面的user部分得到对user的对话话记忆生成user上下文
def chat_fn(message, history):
    messages = []
    msg = {'role': 'system', 'content': 'You are a helpful assistant.'}
    messages.append(msg)
    print("histroy is",history)
    for h in history:
        if h["role"]=="user":
            hcontents=h["content"]
            print(hcontents)
            for hcntent in hcontents:
                if hcntent["type"]=="text":
                    hctnt=hcntent["text"]
                    print("hctnt is",hctnt)
                    messages.append({'role': 'user', 'content': hctnt})  

    # Append user message as dict, not as string
    messages.append({'role': 'user', 'content': message})
    print("Chat messages:", messages)
    response = ollama.chat(model="qwen3-vl:235b-cloud", messages=messages)
    return response['message']['content']

demo = gr.ChatInterface(fn=chat_fn, title="Echo Bot")
demo.launch()
