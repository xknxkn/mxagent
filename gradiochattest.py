import gradio as gr
import sys
import io

# 打印显示中文
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

def echo(message, history):
    print("Chat history:", history)
    print("Received message:", message)
    #remove first two items from history
    if len(history)>=4:
        histroy=histroy[2:]
    return "bot say "+message

demo = gr.ChatInterface(fn=echo,title="Echo Bot")
demo.launch()
