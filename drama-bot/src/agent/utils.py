#### General Utils
from openai import types 
BLACKLIST = ["x.com", "twitter.com", "politifact.com", "factcheck.org", "reuters.com", "instagram.com", "facebook.com", "guardian.com", "usafacts.org", "threads.net"]

COST_DICT = {
    "gpt-3.5-turbo-0125": {
        "max_context": 16_385,
        "cost_per_input_token": 5e-07,
        "cost_per_output_token": 1.5e-06,
    },
    "gpt-3.5-turbo-1106": {
        "max_context": 16_385,
        "cost_per_input_token": 1.5e-06,
        "cost_per_output_token": 2e-06,
    },
    "gpt-3.5-turbo-16k-0613": {
        "max_context": 16_385,
        "cost_per_input_token": 1.5e-06,
        "cost_per_output_token": 2e-06,
    },
    "gpt-4-32k-0613": {
        "max_context": 32_768,
        "cost_per_input_token": 6e-05,
        "cost_per_output_token": 0.00012,
    },
    "gpt-4-0613": {
        "max_context": 8_192,
        "cost_per_input_token": 3e-05,
        "cost_per_output_token": 6e-05,
    },
    "gpt-4-1106-preview": {
        "max_context": 128_000,
        "cost_per_input_token": 1e-05,
        "cost_per_output_token": 3e-05,
    },
    "gpt-4-0125-preview": {
        "max_context": 128_000,
        "cost_per_input_token": 1e-05,
        "cost_per_output_token": 3e-05,
    },
    "gpt-4-turbo-2024-04-09": {
        "max_context": 128_000,
        "cost_per_input_token": 1e-05,
        "cost_per_output_token": 3e-05,
    },
    "gpt-4o-2024-05-13": {
        "max_context": 128_000,
        "cost_per_input_token": 5e-06,
        "cost_per_output_token": 15e-06,
    },
    "gpt-4o-2024-08-06": {
        "max_context": 128_000,
        "cost_per_input_token": 2.5e-06,
        "cost_per_output_token": 10e-06,
    },
    "gpt-4o-2024-11-20": {
        "max_context": 128_000,
        "cost_per_input_token": 2.5e-06,
        "cost_per_output_token": 10e-06,
    },
    "gpt-4o-mini-2024-07-18": {
        "max_context": 128_000,
        "cost_per_input_token": 1.5e-07,
        "cost_per_output_token": 6e-07,
    },
    "gpt-4o-mini": {
        "max_context": 128_000,
        "cost_per_input_token": 1.5e-07,
        "cost_per_output_token": 6e-07,
    },
    "gpt-4o-mini-search-preview": {
        "max_context": 128_000,
        "cost_per_input_token": 1.5e-07,
        "cost_per_output_token": 6e-07,
    },
    "gpt-4o-search-preview": {
        "max_context": 128_000,
        "cost_per_input_token": 2.5e-06,
        "cost_per_output_token": 10e-06,
    },
}

def calculate_gpt_cost(response: types.Completion, model_name: str = "gpt-4o-mini"):
    return (response.usage.prompt_tokens * COST_DICT[model_name]["cost_per_input_token"]) + (response.usage.completion_tokens * COST_DICT[model_name]["cost_per_output_token"])

def calculate_gpt_cost_with_tokens(input_tokens, output_tokens, model_name: str = "gpt-4o-mini"):
    return (input_tokens * COST_DICT[model_name]["cost_per_input_token"]) + (output_tokens * COST_DICT[model_name]["cost_per_output_token"])

def get_input_tokens(response: types.Completion):
    return response.usage.prompt_tokens

def get_output_tokens(response: types.Completion):
    return response.usage.completion_tokens
