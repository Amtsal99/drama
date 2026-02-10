from agent.utils import COST_DICT

def parse_gemini_response(response) -> list:
    """Parses the Gemini response to extract relevant information."""
    parsed_parts = []
    for part in response.candidates[0].content.parts:
        parsed_parts.append(part)
    return parsed_parts

def calculate_gemini_cost(response, model_name: str = "gemini-2.5-flash") -> float:
    """Calculates the cost of the Gemini API call based on token usage."""
    cost = (response.usage.input_tokens * COST_DICT[model_name]["cost_per_input_token"] +
            response.usage.completion_tokens * COST_DICT[model_name]["cost_per_output_token"])
    return cost