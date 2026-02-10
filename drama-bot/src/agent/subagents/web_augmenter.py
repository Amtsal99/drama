from agent.prompts import RETRIEVER_WEBSEARCH_VERIFICATION, RETRIEVER_WEBSEARCH_QA
from agent.utils import COST_DICT

import google.generativeai as genai
from gemini_tool import calculate_gemini_cost

import os
import json

class WebAugmenter:
    def __init__(self, task, client, output_path):
        self.task = task
        self.client = client
        self.output_path = output_path

    def run(self, query):

        if self.task == "verification":
            prompt = RETRIEVER_WEBSEARCH_VERIFICATION.format(query=query)
        else:
            prompt = RETRIEVER_WEBSEARCH_QA.format(query=query)

        try:
            response = self.model.generate_content(
                prompt,
                tools='google_search_retrieval', 
                generation_config=genai.GenerationConfig(
                    temperature=0.0
                )
            )
            if not response.parts:
                    print("Warning: WebAugmenter response blocked.")
                    response_text = "Error: Search blocked by safety filters."
                    grounding_metadata = None
            else:
                response_text = response.text
                grounding_metadata = response.candidates[0].grounding_metadata
                

        except Exception as e:
            print(f"Error in WebAugmenter: {e}")
            response_text = f"Error executing search: {str(e)}"
            grounding_metadata = None
            
        search_path = []
        if grounding_metadata and grounding_metadata.grounding_chunks:
            for chunk in grounding_metadata.grounding_chunks:
                if chunk.web and chunk.web.uri:
                    search_path.append(chunk.web.uri)
        
        metadata_str = str(grounding_metadata) if grounding_metadata else "No grounding metadata"
        trace = "Annotations:\n" + metadata_str + "\n\nMessage Content:\n" + response_text
        if response.parts:
            cost = calculate_gemini_cost(response, model_name="gemini-2.5-flash")
        
        output_file = os.path.join(self.output_path, "output.json")
        with open(output_file, "r") as f:
            data = json.load(f)
        data["trace"].append(trace)
        if len(data["cost"]) == 0:
            data["cost"].append(cost) 
        else:
            data["cost"].append(cost + data["cost"][-1])
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        # search_path = []
        # for citation in completion.choices[0].message.annotations:
        #     search_path.append(citation["url_citation"]["url"])
        return response_text, search_path