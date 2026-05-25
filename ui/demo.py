import gradio as gr
import requests

# 调用 FastAPI
def recommend(query):

    response = requests.post(
        "http://127.0.0.1:8000/search",
        json={"query": query}
    )

    data = response.json()

    output = ""

    for item in data["results"]:

        output += f"""
Product:
{item['title']}

Description:
{item['description']}

--------------------

"""

    return output

# Gradio页面
iface = gr.Interface(
    fn=recommend,
    inputs="text",
    outputs="text",
    title="AI E-commerce Recommendation System",
    description="Input your shopping needs"
)

iface.launch()