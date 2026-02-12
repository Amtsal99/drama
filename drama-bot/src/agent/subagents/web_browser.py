import platform
import time
import json
import re
import os
import logging
import subprocess
import zipfile
import shutil
import pandas as pd
from io import StringIO

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException

from google import genai
# from google.api_core import exceptions
from google.genai import types, errors as exceptions
# from google.genai import exceptions

from .utils_webbrowser import get_web_element_rect, encode_image, extract_information, get_webarena_accessibility_tree, clip_message_and_obs
from .gemini_tool import calculate_gemini_cost

from agent.prompts import RETRIEVER_FIND_WEBSITE_TASK_DESC, RETRIEVER_SEARCH_TERM_DESC, RETRIEVER_BROWSE_SYSTEM_PROMPT
from agent.utils import BLACKLIST, COST_DICT

class WebBrowser:
    def __init__(self, api_model, client: genai.Client, output_dir, task):
        # self.client = OpenAI(api_key=api_key, organization=org)
        self.client = client
        self.max_iter = 10
        self.api_model = api_model
        self.output_dir = output_dir
        self.window_width = 1024
        self.window_height = 768
        self.max_attached_imgs = 3
        self.fix_box_color = True
        self.seed = 42
        
        self.number_of_downloaded_files = len(os.listdir(self.output_dir))

        # for driver_config
        self.save_accessibility_tree = False
        self.force_device_scale = False
        self.headless = True

        self.task = task
    
    def run(self, query, website = None):

        if website is None:
            search_term = self.plan_search_term(query)
            # start browsing
            res, search_path = self.browse(query, search_term)
        else:
            res, search_path = self.browse(query, website)

        if res is None:
            return search_path

        # Case 1: data is returned in raw contents
        match = re.search(r"```csv\n(.*?)\n```", res, re.DOTALL)
        if match:
            csv_content = match.group(1) 
            df = pd.read_csv(StringIO(csv_content))

            df = df.drop(columns=["Source"])

            # same dir n name with data transformer
            output_csv_path = f"{self.output_dir}/data.csv"

            df.to_csv(output_csv_path, index=False)

            print(f"CSV file saved successfully at: {output_csv_path}")

        else:
            # Case 2: data is downloaded
            if os.path.exists(f"{self.output_dir}/download") and os.path.isdir(f"{self.output_dir}/download"):
                if len(os.listdir(f"{self.output_dir}/download")) > 0:
                    download_path = f"{self.output_dir}/download"
                    for file_name in os.listdir(download_path):
                        full_file_path = os.path.join(download_path, file_name)
                        if os.path.isfile(full_file_path):
                            # same dir but diff name with data transformer
                            shutil.copy(full_file_path, self.output_dir)
                    return search_path

            # Case 3: link to data is returned
            try:
                command = [
                    "curl",
                    "-L",  # Follow redirects
                    "-f",  # Fail on HTTP errors instead of saving an error page
                    "-OJ",  # Preserve the original filename
                    "--output-dir", self.output_dir,
                    "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",  # Spoof browser
                    res  # File link
                ]

                subprocess.run(command, check=True)
                print(f"Downloaded: {res}")
                for file_name in os.listdir(self.output_dir):
                        file_path = os.path.join(self.output_dir, file_name)

                        # same dir but diff name with data transformer
                        # Check if the file is a zip file
                        if zipfile.is_zipfile(file_path):
                            print(f"Unzipping: {file_name}")
                            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                                zip_ref.extractall(self.output_dir)
                                print(f"Extracted: {file_name}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to download {res}: {e}")
                return search_path

        return search_path
    
    # plan for search term
    def plan_search_term(self, query):
        if self.task == "verification":
            action = "verify"
        else:
            action = "answer"
        
        # convert to gemini standard json format 
        contents = [
            {
                "role": "user",
                "parts": [
                    {"text" : RETRIEVER_SEARCH_TERM_DESC.format(action=action, query=query)}
                ]
            }
        ]

        # time.sleep(12)
        response = self.client.models.generate_content(
            model=self.api_model,
            contents=contents,
        )

        cost = calculate_gemini_cost(response, model_name=self.api_model)
        
        # parts_response = parse_gemini_response(response)

        logging.info(f"Search Term: {response.text}")

        output_file = os.path.join(self.output_dir, "output.json")
        with open(output_file, "r") as f:
            data = json.load(f)
        data["trace"].append(f"search term: {response.text}")
        if len(data["cost"]) == 0:
            data["cost"].append(cost) 
        else:
            data["cost"].append(cost + data["cost"][-1])
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        return response.text
    
    # The browsing behavior
    def browse(self, query, search_term):

        options = driver_config(self.save_accessibility_tree, self.force_device_scale, self.headless, self.output_dir)
        options.add_argument("--headless")
        options.add_argument("--disable-blink-features=AutomationControlled")

        # Save Result file
        current_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
        result_dir = os.path.join(self.output_dir, current_time)
        os.makedirs(result_dir, exist_ok=True)

        driver_task = webdriver.Chrome(options=options)

        # About window size, 765 tokens
        # You can resize to height = 512 by yourself (255 tokens, Maybe bad performance)
        driver_task.set_window_size(self.window_width, self.window_height)  # larger height may contain more web information
        if search_term.startswith("http"):
            driver_task.get(search_term)
        else:
            driver_task.get('https://www.google.com/')
        try:
            driver_task.find_element(By.TAG_NAME, 'body').click()
        except:
            pass
        # sometimes enter SPACE, the page will sroll down
        driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")
        time.sleep(5)

        fail_obs = ""  # When error execute the action
        pdf_obs = ""  # When download PDF file
        warn_obs = ""  # Type warning
        pattern = r'Thought:|Action:|Observation:'

        if self.task == "verification":
            action = "verify"
        else:
            action = "answer"
            
        # gemini_model = configure_gemini_model(self.api_model, system_prompt=RETRIEVER_BROWSE_SYSTEM_PROMPT.format(action=action, blacklist=BLACKLIST))
        
        messages = []
                    
        obs_prompt = "Observation: please analyze the attached screenshot and give the Thought and Action. "

        init_msg = RETRIEVER_FIND_WEBSITE_TASK_DESC.format(action=action, query=query, search_term=search_term) + obs_prompt

        it = 0
        accumulate_prompt_token = 0
        accumulate_completion_token = 0

        website_link = None

        def count_files_in_directory(directory):
            return len(os.listdir(directory))

        visited_urls = []
        res = None
        trace = ""

        while it < self.max_iter:
            it += 1
            if any(domain in driver_task.current_url for domain in BLACKLIST):
                driver_task.back()
            visited_urls.append(driver_task.current_url)
            if not fail_obs:
                try:
                    rects, web_eles, web_eles_text = get_web_element_rect(driver_task, fix_color=self.fix_box_color)

                except Exception as e:
                    logging.error('Driver error when adding set-of-mark.')
                    logging.error(e)
                    break

                img_path = os.path.join(self.output_dir, 'screenshot{}.png'.format(it))
                driver_task.save_screenshot(img_path)

                # accessibility tree
                if self.save_accessibility_tree:
                    accessibility_tree_path = os.path.join(self.output_dir, 'accessibility_tree{}'.format(it))
                    get_webarena_accessibility_tree(driver_task, accessibility_tree_path)

                # format msg
                curr_msg = format_msg(it, init_msg, pdf_obs, warn_obs, img_path, web_eles_text)
                messages.append(curr_msg)
            else:
                curr_msg = {
                    'role': 'user',
                    'parts': [{'text': fail_obs}]
                }
                messages.append(curr_msg)

            messages = clip_message_and_obs(messages, self.max_attached_imgs)

            # Call GPT-4v API
            # prompt_tokens, completion_tokens, gpt_call_error, openai_response = call_gpt4v_api(self.client, messages, self.api_model, self.seed)
            
            # to avoid reaching rate limit
            time.sleep(12)
            
            # call gemini-2.5-flash API
            prompt_tokens, completion_tokens, gemini_call_error, gemini_response = call_gemini_api(client=self.client, api_model=self.api_model, messages=messages, seed=self.seed, action=action)

            if gemini_call_error:
                break
            else:
                accumulate_prompt_token += prompt_tokens
                accumulate_completion_token += completion_tokens
                logging.info(f'Accumulate Prompt Tokens: {accumulate_prompt_token}; Accumulate Completion Tokens: {accumulate_completion_token}')
                logging.info('API call complete...')
            # parts_gemini_response = parse_gemini_response(gemini_response)
            gemini_response_text = gemini_response.text
            messages.append({'role': 'assistant', 'content': gemini_response_text})

            # remove the rects on the website
            if rects:
                for rect_ele in rects:
                    driver_task.execute_script("arguments[0].remove()", rect_ele)
                rects = []
                # driver_task.save_screenshot(os.path.join(task_dir, 'screenshot{}_no_box.png'.format(it)))

            # extract action info
            try:
                assert 'Thought:' in gemini_response_text and 'Action:' in gemini_response_text
            except AssertionError as e:
                logging.error(e)
                fail_obs = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
                continue

            # bot_thought = re.split(pattern, gemini_response_text)[1].strip()
            chosen_action = re.split(pattern, gemini_response_text)[2].strip()
            action_key, info = extract_information(chosen_action)
            trace += "\n" + f"Action Key: {action_key}"
            trace += "\n" + f"Info: {info}"

            fail_obs = ""
            pdf_obs = ""
            warn_obs = ""

            window_handle_task = driver_task.current_window_handle
            driver_task.switch_to.window(window_handle_task)

            if action_key == 'check_link':

                click_ele_number = int(info['content'])
                web_ele = web_eles[click_ele_number]

                warn_obs = exec_get_link(web_ele)
                if warn_obs is not None:
                    valid_ext = [".csv", ".tsv", ".zip", ".xlsx", ".xls", ".pdf"]
                    if not any(ext in warn_obs for ext in valid_ext):
                        warn_obs = f"\n Element {click_ele_number} points to an invalid download link! Your next action should NOT be GetLink."
                        allow_get_link = False
                        continue
                    else:
                        warn_obs += f"\n Element {click_ele_number} ponits to a valid download link! Your next action CAN be GetLink."
                else:
                    warn_obs = f"\n Element {click_ele_number} points to an invalid download link! Your next action should NOT be GetLink."
                    allow_get_link = False

            elif action_key == 'get_link':
                if allow_get_link:
                    click_ele_number = int(info['content'])
                    web_ele = web_eles[click_ele_number]

                    website_link = exec_get_link(web_ele)
                    visited_urls.append(website_link)
                    res = website_link
                    break
                else:
                    fail_obs = "You should not choose GetLink because this element does not point to a valid file."

            elif action_key == 'click':
                click_ele_number = int(info[0])
                web_ele = web_eles[click_ele_number]
                exec_action_click(info, web_ele, driver_task)

            elif action_key == "wait_for_downloading":
                    start_time = time.time()
                    downloaded = False
                    while True:
                        time.sleep(30)
                        current_file_count = count_files_in_directory(self.output_dir)

                        if current_file_count != self.number_of_downloaded_files:
                            downloaded = True
                            break

                        if time.time() - start_time > 120:
                            logging.info("2 minutes have passed without any change in the output directory.")
                            warn_obs = "\n This is not a valid download link."
                            break
                    if downloaded:
                        res = ""
                        break

            elif action_key == 'wait':
                time.sleep(5)

            elif action_key == 'type':
                type_ele_number = int(info['number'])
                web_ele = web_eles[type_ele_number]

                warn_obs = exec_action_type(info, web_ele, driver_task)

            elif action_key == 'scroll':
                exec_action_scroll(info, web_eles, driver_task, self.window_height)

            elif action_key == 'goback':
                driver_task.back()
                time.sleep(2)

            elif action_key == 'google':
                driver_task.get('https://www.google.com/')
                time.sleep(2)

            elif action_key == 'get_data':
                res = info['content']
                match = re.search(r"```csv\n(.*?)\n```", res, re.DOTALL)

                if not match:
                    fail_obs = "You should wrap your answer as a relation table."
                    continue
                else:
                    break

            else:
                raise NotImplementedError
            allow_get_link = True
            fail_obs = ""
        driver_task.quit()

        cost = accumulate_prompt_token * COST_DICT[self.api_model]["cost_per_input_token"] + accumulate_completion_token * COST_DICT[self.api_model]["cost_per_output_token"]
        output_file = os.path.join(self.output_dir, "output.json")
        with open(output_file, "r") as f:
            data = json.load(f)
        data["trace"].append(trace)
        if len(data["cost"]) == 0:
            data["cost"].append(cost) 
        else:
            data["cost"].append(cost + data["cost"][-1])
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        return res, visited_urls
    
def driver_config(save_accessibility_tree, force_device_scale, headless, download_dir):
    options = webdriver.ChromeOptions()

    if save_accessibility_tree:
        force_device_scale = True

    if force_device_scale:
        options.add_argument("--force-device-scale-factor=1")
    if headless:
        options.add_argument("--headless")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
    options.add_experimental_option(
        "prefs", {
            "download.default_directory": f"{download_dir}/download/",
            "plugins.always_open_pdf_externally": True
        }
    )
    return options


def format_msg(it, init_msg, pdf_obs, warn_obs, img_path, web_text):
    guidance_text = f"I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"
    
    if it == 1:
        text_content = f"{init_msg}{guidance_text}"
    else:
        if not pdf_obs:
            text_content = f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. {guidance_text}"
        else:
            text_content = f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. {guidance_text}"

    with open(img_path, 'rb') as f:
        image_bytes = f.read()

    curr_msg = types.Content(
        role='user',
        parts=[
            types.Part.from_text(text=text_content),
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/png'
            )
        ]
    )
    
    return curr_msg

def call_gemini_api(client: genai.Client, api_model, messages, seed, action):
            
    # gemini_model = configure_gemini_model(api_model, system_prompt=RETRIEVER_BROWSE_SYSTEM_PROMPT.format(action=action, blacklist=BLACKLIST))
    
    retry_times = 0
    generation_conf = types.GenerateContentConfig(
        system_instructions=RETRIEVER_BROWSE_SYSTEM_PROMPT.format(action=action, blacklist=BLACKLIST),
        max_output_tokens=8192,
        seed=seed,
        http_options=types.HttpOptions(timeout=600)
    )
    
    while True:
        try:
            # to avoid reaching rate limit for free tier API
            # time.sleep(12)
            response:types.GenerateContentResponse = client.models.generate_content(
                model=api_model,
                contents=messages,
                config=generation_conf
            )
            if not response.parts:
                logging.warning(f"Gemini response blocked by safety filters. Feedback: {response.prompt_feedback}")
                return None, None, True, None
            
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count
            
            gemini_call_error = False
            
            return prompt_tokens, completion_tokens, gemini_call_error, response
            
        except exceptions.APIError as e: 
            if e.code == 429:
                logging.info(f'Gemini Quota Exceeded (Rate Limit), retrying... Error: {e.message}')
                time.sleep(10)
            
            elif e.code == 503:
                logging.info(f'Gemini Service Unavailable (Overloaded), retrying... Error: {e.message}')
                time.sleep(15)
                
            elif e.code == 500:
                logging.info(f'Gemini Internal Server Error, retrying... Error: {e.message}')
                time.sleep(15)
                
            elif e.code == 400:
                logging.error(f'Fatal Input Error: {e.message}')
                return None, None, True, None
                
            else:
                logging.error(f'Other API Error ({e.code}): {e.message}')
                retry_times += 1
                time.sleep(5)
                logging.error(f'Unknown Error occurred: {type(e).__name__} - {e}')
                return None, None, True, None
            
        except Exception as e:
            logging.error(f'Unknown Error occurred: {type(e).__name__} - {e}')
            return None, None, True, None

        retry_times += 1
        if retry_times == 10:
            logging.info('Retrying too many times (Gemini)')
            return None, None, True, None


def exec_action_click(info, web_ele, driver_task):
    driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
    link = web_ele.get_attribute('href')
    if not any(domain in link for domain in BLACKLIST):
        web_ele.click()
        time.sleep(3)

def exec_get_link(web_ele):
    link = web_ele.get_attribute('href')
    if link is None:
        try:
            # Try to find a direct <a> inside the element
            try:
                link_element = web_ele.find_element(By.XPATH, ".//a")
            except NoSuchElementException:
                # If there's no direct <a>, try getting the closest parent <a>
                link_element = web_ele.find_element(By.XPATH, "./ancestor::a")

            link = link_element.get_attribute('href')
        except NoSuchElementException:
            return None
    return link

def exec_action_type(info, web_ele, driver_task):
    warn_obs = ""
    type_content = info['content']

    ele_tag_name = web_ele.tag_name.lower()
    ele_type = web_ele.get_attribute("type")
    # outer_html = web_ele.get_attribute("outerHTML")
    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
        warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."
    try:
        # Not always work to delete
        web_ele.clear()
        # Another way to delete
        if platform.system() == 'Darwin':
            web_ele.send_keys(Keys.COMMAND + "a")
        else:
            web_ele.send_keys(Keys.CONTROL + "a")
        web_ele.send_keys(" ")
        web_ele.send_keys(Keys.BACKSPACE)
    except:
        pass

    actions = ActionChains(driver_task)
    actions.click(web_ele).perform()
    actions.pause(1)

    try:
        driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};""")
    except:
        pass

    actions.send_keys(type_content)
    actions.pause(2)

    actions.send_keys(Keys.ENTER)
    actions.perform()
    time.sleep(10)
    return warn_obs


def exec_action_scroll(info, web_eles, driver_task, window_height):
    scroll_ele_number = info['number']
    scroll_content = info['content']
    if scroll_ele_number == "WINDOW":
        if scroll_content == 'down':
            driver_task.execute_script(f"window.scrollBy(0, {window_height*2//3});")
        else:
            driver_task.execute_script(f"window.scrollBy(0, -{window_height*2//3});")
    else:
        # Scroll a specific element (subwindow)
        # scroll_ele_number = int(scroll_ele_number)

        scroll_ele_number = int(scroll_ele_number)
        web_ele = web_eles[scroll_ele_number]

        actions = ActionChains(driver_task)
        driver_task.execute_script("arguments[0].focus();", web_ele)
        if scroll_content == 'down':
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_DOWN).key_up(Keys.ALT).perform()
        else:
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_UP).key_up(Keys.ALT).perform()

    time.sleep(3)