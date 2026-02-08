import google.generativeai as genai

def init_gemini_client(api_key):
    genai.configure(api_key=api_key)
    
def get_gemini_model(model_name: str):
    return genai.GenerativeModel(model_name)