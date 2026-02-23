from agent.data_retriever import DataRetriever
from agent.data_analyzer import DataAnalyzer

from dotenv import load_dotenv
from google import genai

import os
import json
import logging

class DramaBot:
    def __init__(self, task, output_path, api_model):
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        output_file = os.path.join(self.output_path, "output.json")
        if not os.path.exists(output_file):
            with open(output_file, "w") as f:
                json.dump({"trace": [], "cost": []}, f, indent=2)

        self.task = task

        load_dotenv()
        # self.openai_api_key = os.getenv('OPENAI_API_KEY')
        # self.openai_org = os.getenv('OPENAI_ORG')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        
        gemini_client = genai.Client(api_key=self.google_api_key)
        
        self.data_retriever = DataRetriever(task = self.task, api_key = self.google_api_key, api_model = api_model, output_path = self.output_path, client=gemini_client)
        self.data_analyzer = DataAnalyzer(task = self.task, api_key = self.google_api_key, api_model = api_model, output_path = self.output_path, client=gemini_client)

    def run(self, query):
        logging.info(f"📂 Data Retriever Starts")
        search_path = self.data_retriever.run(query) 

        logging.info(f"💻 Data Analyzer Starts")
        result, df, pandas_code = self.data_analyzer.run(query)

        return result, df, pandas_code, search_path