import argparse
import os
import json
import time
import re
import base64

# from openai import OpenAI
from google import genai
from google.genai import types, errors

SYSTEM_PROMPT = """As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'NOT SUCCESS'."""
USER_PROMPT = """TASK: <task>
Result Response: <answer>
<num> screenshots at the end: """

COST_PER_PROMPT_TOKEN = 3e-07
COST_PER_COMPLETION_TOKEN = 2.5e-06


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def auto_eval_by_gemini(process_dir, client:genai.Client, img_num, api_model="gemini-2.5-flash"):
    print(f'--------------------- {process_dir} ---------------------')
    res_files = sorted(os.listdir(process_dir))
    with open(os.path.join(process_dir, 'interact_messages.json')) as fr:
        it_messages = json.load(fr)

    if len(it_messages) == 1:
        print('Not find answer for ' + process_dir + ' only system messages')
        print()
        return 0

    task_info = it_messages[1]["content"]
    if type(task_info) == list:
        task_info = task_info[0]["text"]
    assert 'Now given a task' in task_info
    pattern = r"Now given a task:(.+?)Please interact with"
    matches = re.search(pattern, task_info)
    task_content = matches.group(1).strip()

    ans_info = it_messages[-1]["content"]
    if 'Action: ANSWER' not in ans_info:
        print('Not find answer for ' + process_dir)
        print()
        return 0
    pattern_ans = r"ANSWER[; ]+\[?(.[^\]]*)\]?"
    matches_ans = re.search(pattern_ans, ans_info)
    answer_content = matches_ans.group(1).strip()

    pattern_png = r'screenshot(\d+)\.png'
    matches = [(filename, int(re.search(pattern_png, filename).group(1))) for filename in res_files if re.search(pattern_png, filename)]
    matches.sort(key=lambda x: x[1])
    end_files = matches[-img_num:]

    user_prompt_tmp = USER_PROMPT.replace('<task>', task_content)
    user_prompt_tmp = user_prompt_tmp.replace('<answer>', answer_content)
    user_prompt_tmp = user_prompt_tmp.replace('<num>', str(img_num))
    
    message_parts = []
    message_parts.append(types.Part.from_text(text=user_prompt_tmp))
    
    for png_file in end_files:
        file_path = os.path.join(process_dir, png_file[0])
        
        with open(file_path, 'rb') as f:
            image_bytes = f.read()
            
        message_parts.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/png'
            )
        )
        
    message_parts.append(types.Part.from_text(text="Your verdict:\n"))
    
    while True:
        try:
            print('Calling gemini API to get the auto evaluation......')
            gen_conf = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=1000,
                seed=42
            )
            response:types.GenerateContentResponse = client.models.generate_content(
                model=api_model,
                contents=types.Content(role="user",
                                       parts=message_parts), 
                generation_config=gen_conf
            )
            usage = response.usage_metadata
            prompt_tokens = usage.prompt_token_count or 0
            completion_tokens = usage.candidates_token_count or 0
            
            print('Prompt Tokens:', prompt_tokens, ';',
                'Completion Tokens:', completion_tokens)
            print('Prompt Tokens:', response.usage_metadata.prompt_token_count, ';',
                  'Completion Tokens:', response.usage_metadata.candidates_token_count)
            
            total_cost = (prompt_tokens * COST_PER_PROMPT_TOKEN) + (completion_tokens * COST_PER_COMPLETION_TOKEN)
            print("Cost: ", total_cost)

            print('API call complete...')
            gemini_res = response.text
            break
        
        except errors.APIError as e:
            print(f"API Error ({e.code}): {e.message}")
            
            if e.code in [429, 503]:
                time.sleep(10)
            elif e.code == 400:
                print("Fatal Invalid Request")
                exit(0)
            else:
                time.sleep(10)

        except Exception as e:
            print(f"Unknown Error: {e}")
            time.sleep(10)
    # print_message = messages[1]
    # for idx in range(len(print_message['content'])):
    #     if print_message['content'][idx]['type'] == 'image_url':
    #         print_message['content'][idx]['image_url'] = {"url": "data:image/png;base64, b64_img"}

    # print(print_message)
    # print(gemini_res)

    auto_eval_res = 0 if 'NOT SUCCESS' in gemini_res else 1
    if 'SUCCESS' not in gemini_res:
        auto_eval_res = None
    print('Auto_eval_res:', auto_eval_res)
    print()
    return auto_eval_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_dir', type=str, default='results')
    parser.add_argument('--lesson_dir', type=str, default='results')
    parser.add_argument("--api_key", default="key", type=str, help="YOUR_OPENAI_API_KEY")
    parser.add_argument("--api_model", default="gemini-2.5-flash", type=str, help="api model name")
    parser.add_argument("--max_attached_imgs", type=int, default=1)
    args = parser.parse_args()

    client = genai.Client(api_key=args.api_key)
    webs = ['Allrecipes', 'Amazon', 'Apple', 'ArXiv', 'BBC News', 'Booking', 'Cambridge Dictionary',
            'Coursera', 'ESPN', 'GitHub', 'Google Flights', 'Google Map', 'Google Search', 'Huggingface', 'Wolfram Alpha']

    for web in webs:
        web_task_res = []
        for idx in range(0, 46):
            file_dir = os.path.join(args.process_dir, 'task'+web+'--'+str(idx))
            if os.path.exists(file_dir):
                response = auto_eval_by_gemini(process_dir=file_dir, client=client, img_num=args.max_attached_imgs, api_model=args.api_model)
                web_task_res.append(response)
            else:
                pass
        if web_task_res:
            print(web_task_res)
if __name__ == '__main__':
    main()
