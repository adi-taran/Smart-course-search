from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def prepare_embeddings(model, data_file='data/courses_data.csv'):
    df = pd.read_csv(data_file)
    course_descriptions = df['description'].tolist()
    return model.encode(course_descriptions, convert_to_tensor=True), df

def get_recommendations(model, df, course_embeddings, user_query):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, course_embeddings)[0]
    top_results = torch.topk(scores, k=5)

    recommendations = []
    for idx in top_results[1]:
        recommendations.append({
            "Course Title": df.iloc[idx.item()]['course_title'],
            "Description": df.iloc[idx.item()]['description']
        })
    return recommendations
