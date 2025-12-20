import gradio as gr

# 模拟数据库：存储不同会话ID的用户名
user_data = {}

def greet(name, request: gr.Request):
    if request.session_hash not in user_data:
        user_data[request.session_hash] = name
    else:
        name = user_data[request.session_hash]
    return f"你好，{name}！这是你的会话。"

with gr.Blocks() as demo:
    name_input = gr.Textbox(label="输入你的名字")
    output = gr.Textbox(label="问候语")

    submit_btn = gr.Button("提交")
    submit_btn.click(fn=greet, inputs=[name_input], outputs=output)

demo.launch(share=True)
