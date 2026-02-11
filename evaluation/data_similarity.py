import os
import numpy as np
# from openai import OpenAI
import google.generativeai as genai
from prompts import DATA_SIMILARITY, SEPARATE_COLUMNS
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def eval_data(df1, df2, task, query_info, column_match = True, method="llm-as-a-judge"):

    if method == "llm-as-a-judge":
        if task == "verification":
            action = "verifying"
            query = query_info["claim"]
        else:
            action = "answering"
            query = query_info["question"]
        
        if column_match:
            return eval_data_llm_column_match(df1, df2, query, action)
        else:
            return eval_data_llm(df1, df2, query, action)
        
    elif method == "embedding":
        if column_match:
            return eval_data_embedding_column_match(df1, df2)
        else:
            return eval_data_embedding(df1, df2)

def eval_data_llm(df1, df2, query, action):
    messages = [
        {
            "role": "user",
            "parts": [{"text": DATA_SIMILARITY.format(query=query, df1_head=df1.head(), df2_head=df2.head(), action=action)}]
        }
    ]
    load_dotenv()
    # openai_api_key = os.getenv('OPENAI_API_KEY')
    # openai_org = os.getenv('OPENAI_ORG')
    # client = OpenAI(api_key=openai_api_key, organization=openai_org)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    
    response = gemini_model.generate_content(contents=messages).text

    return float(response)

def eval_data_llm_column_match(df1, df2, query, action):
    messages = [
        {
            "role": "user",
            "parts": [{"text": SEPARATE_COLUMNS.format(query=query, df1_head=df1.head(), df2_head=df2.head(), action=action)}]
        }
    ]
    # openai_api_key = os.getenv('OPENAI_API_KEY')
    # openai_org = os.getenv('OPENAI_ORG')
    # client = OpenAI(api_key=openai_api_key, organization=openai_org)
    
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    
    response = gemini_model.generate_content(contents=messages).text

    return float(response)

def eval_data_embedding(df1, df2):

    text1 = f"{df1.head()}"
    text2 = f"{df2.head()}"
    
    embedding_model_name = "models/text-embedding-004"
    
    res = genai.embed_content(model=embedding_model_name,
                              content=[text1, text2],
                              task_type="semantic_similarity")
    embeddings = res["embedding"]
    embedding1 = embeddings[0]
    embedding2 = embeddings[1]    
    return cosine_similarity([embedding1], [embedding2])[0][0]

def eval_data_embedding_column_match(df1, df2):
    model_name = "models/text-embedding-004"

    df1_cols = list(df1.columns.values)
    df2_cols = list(df2.columns.values)

    df1_embeddings = []
    for col in df1_cols:
        result = genai.embed_content(
            model=model_name,
            content=str(df1[col].to_frame()),
            task_type="semantic_similarity"
        )
        df1_embeddings.append(result['embedding'])

    df2_embeddings = []
    for col in df2_cols:
        result = genai.embed_content(
            model=model_name,
            content=str(df2[col].to_frame()),
            task_type="semantic_similarity"
        )
        df2_embeddings.append(result['embedding'])

    best_similarities = []
    for i in range(len(df1_embeddings)):
        similarities = []
        for j in range(len(df2_embeddings)):
            similarity = cosine_similarity([df1_embeddings[i]], [df2_embeddings[j]])[0][0]
            similarities.append(similarity)
        best_similarities.append(max(similarities))

    return np.mean(best_similarities)