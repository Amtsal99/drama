from agent.utils import COST_DICT

# def parse_gemini_response(response) -> list:
#     """Parses the Gemini response to extract relevant information."""
#     parsed_parts = []
#     for part in response.candidates[0].content.parts:
#         parsed_parts.append(part)
#     return parsed_parts

def calculate_gemini_cost(response, model_name: str = "gemini-2.5-flash") -> float:
    """Calculates the cost of the Gemini API call based on token usage."""
    cost = (response.usage_metadata.prompt_token_count * COST_DICT[model_name]["cost_per_input_token"] +
            response.usage_metadata.candidates_token_count * COST_DICT[model_name]["cost_per_output_token"])
    return cost

def get_input_tokens(response) -> int:
    """Retrieves the number of input tokens used in the Gemini API call."""
    return response.usage_metadata.prompt_token_count

def get_output_tokens(response) -> int:
    """Retrieves the number of output tokens used in the Gemini API call."""
    return response.usage_metadata.candidates_token_count