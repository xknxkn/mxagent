import gradio as gr

def greet(studentname, intensity):
    return "Hello, " + studentname + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

#demo.launch()

demo.launch(share=True)
