import gradio as gr
from app.search import load_model, prepare_embeddings, get_recommendations

model = load_model()
course_embeddings, course_data = prepare_embeddings(model)


def search_courses(user_query):
    return get_recommendations(model, course_data, course_embeddings, user_query)


iface = gr.Interface(fn=search_courses, inputs="text", outputs="json", title="Smart Course Search")
iface.launch()
