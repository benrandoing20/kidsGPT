import gradio as gr
def generate(text, grade):
    prompt = f"<s_grade={grade}> {text}"
    # call generate.py logic here
    return generated_output
gr.Interface(generate, inputs=["text", gr.Slider(1,8)], outputs="text").launch()
