from agent.prompts import ANALYZER_CODE_GEN_VERIFICATION_TASK_DESC, ANALYZER_CODE_GEN_QA_TASK_DESC
from agent.utils import COST_DICT

import os
import pandas as pd
import re
import json
import logging

from google import genai
from agent.subagents.gemini_tool import calculate_gemini_cost

class DataAnalyzer:
    def __init__(self, task, api_key, api_model, output_path, client:genai.Client):
        self.client = client
        self.task = task
        # self.client = get_gemini_model(api_model)
        self.api_model = api_model
        self.output_path = output_path
        self.api_key = api_key

    def run(self, query):

        file_path = f"{self.output_path}/data.csv"

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            code = self.code_gen(df, query)
            try:
                return self.code_exec(df, code)
            except:
                return None, df, code
        
        else:
            if self.task == "verification":
                return False, "", ""
            return None, None, None
    
    def code_gen(self, df, query):
        if self.task == "verification":
            prompt = ANALYZER_CODE_GEN_VERIFICATION_TASK_DESC.format(query=query, df_columns=df.columns, df_head=df.head())
        else:
            prompt = ANALYZER_CODE_GEN_QA_TASK_DESC.format(query=query, df_columns=df.columns, df_head=df.head())
            
        # gemini_model = configure_gemini_model(self.api_model, system_prompt="You are a Python code generator specializing in Pandas. Provide only raw Python code without any markdown formatting.")

        contents = [
            {"role": "user",
             "parts": [
                {"text": prompt}
             ]}
        ]
        
        response = self.client.models.generate_content(model = self.api_model,
                                                       contents=contents)
        
        pandas_code = response.text.strip()
        pandas_code = re.sub(r'```python\n|```', '', pandas_code)

        cost = calculate_gemini_cost(response, model_name="gemini-2.5-flash")
        output_file = os.path.join(self.output_path, "output.json")
        with open(output_file, "r") as f:
            data = json.load(f)
        data["trace"].append(pandas_code)
        if len(data["cost"]) == 0:
            data["cost"].append(cost) 
        else:
            data["cost"].append(cost + data["cost"][-1])
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        return pandas_code
    
    def code_exec(self, df, code):

        local_vars = {'pd': pd}
        
        try:
            exec(code, globals(), local_vars)
        except Exception as e:
            logging.info(f"Error executing generated code: {e}")
            logging.info(f"Generated code: {code}")
            return None, df, code
        
        if self.task == "verification":
            if 'validate_statement' in local_vars:
                result = local_vars['validate_statement'](df)
                return result, df, code
            else:
                logging.info(f"Error executing generated code: {e}")
                logging.info(f"Generated code: {code}")
                return None, df, code
        else:
            if 'answer_question' in local_vars:
                result = local_vars['answer_question'](df)
                return result, df, code
            else:
                logging.info(f"Error executing generated code: {e}")
                logging.info(f"Generated code: {code}")
                return None, df, code