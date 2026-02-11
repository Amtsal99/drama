import google.generativeai as genai

def init_gemini_client(api_key):
    genai.configure(api_key=api_key)
    
def configure_gemini_model(model_name: str, system_prompt: str = None):
    if system_prompt is not None:
        return genai.GenerativeModel(model_name, system_instruction=system_prompt)
    return genai.GenerativeModel(model_name)
