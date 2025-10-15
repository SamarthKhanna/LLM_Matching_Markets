import numpy as np
import random
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import re
from weaklyStable import weaklyStableMatch2
from generate_instance_data import gale_shapley, populate_prefs
from BlockingPairs import blockingPairs


from groq import Groq
from openai import OpenAI
import anthropic
from google import genai
import os
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict
from enum import Enum

class WomenList10(Enum):
    W1 = "W1"
    W2 = "W2"
    W3 = "W3"
    W4 = "W4"
    W5 = "W5"
    W6 = "W6"
    W7 = "W7"
    W8 = "W8"
    W9 = "W9"
    W10 = "W10"

class MenList10(Enum):
    M1 = "M1"
    M2 = "M2"
    M3 = "M3"
    M4 = "M4"
    M5 = "M5"
    M6 = "M6"
    M7 = "M7"
    M8 = "M8"
    M9 = "M9"
    M10 = "M10"


class Matching10(BaseModel):
    model_config = ConfigDict(extra="ignore")
    M1: WomenList10
    M2: WomenList10
    M3: WomenList10
    M4: WomenList10
    M5: WomenList10
    M6: WomenList10
    M7: WomenList10
    M8: WomenList10
    M9: WomenList10
    M10: WomenList10

class Pair(BaseModel):
    model_config = ConfigDict(extra="ignore")
    man: MenList10
    woman: WomenList10

class LLM():
    def __init__(self, auth_file = 'Auth_keys/gemini_auth_key_2.txt', family='gemini', model='gemini-1.0-pro-latest') -> None:
        self.auth_file = auth_file
        self.family = family
        self.model = model
        self.tokens = 0
        if self.family == 'gemini':
            self.init_gemini()
        if self.family == 'mixtral':
            self.init_mixtral()
        if self.family == 'llama':
            self.init_llama()
        if self.family == "chatgpt":
            self.init_gpt()
        if self.family == 'claude':
            self.init_claude()
        if self.family == 'deepseek':
            self.init_deepseek()

    def init_gemini(self):
        with open(self.auth_file, 'r') as f:
            AUTHKEY = f.read()

        # genai.configure(api_key=AUTHKEY)

        # self.llm = genai.GenerativeModel(self.model)
        # self.chat = self.llm.start_chat(history=[])

        self.llm = genai.Client(api_key=AUTHKEY)
            
        # client = AsyncOpenAI(
        #     api_key=AUTHKEY,
        #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        # )
        # self.llm = client.chat.completions

        # config = OpenAIConfig(self.model)
        # self.llm = models.openai(client, config)


    def init_llama(self):
        with open(self.auth_file, 'r') as f:
            AUTHKEY = f.read()

        os.environ["GROQ_API_KEY"]=AUTHKEY

        self.llm = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

    def init_gpt(self):
        with open(self.auth_file, 'r') as f:
            AUTHKEY = f.read()
        os.environ["OPENAI_API_KEY"] = AUTHKEY

        client = OpenAI()
        self.llm = client.chat.completions

    def init_deepseek(self):
        with open(self.auth_file, 'r') as f:
            AUTHKEY = f.read()
        os.environ["OPENAI_API_KEY"] = AUTHKEY

        client = OpenAI(base_url="https://api.deepseek.com")
        self.llm = client.chat.completions

    def init_claude(self):
        with open(self.auth_file, 'r') as f:
            AUTHKEY = f.read()

        self.llm = anthropic.Anthropic(api_key=AUTHKEY)

    def makeLLMRequestFormatting(self, queryText, json_schema):
        print(json_schema)
        response =  self.llm.models.generate_content(
            model = self.model,
            contents = queryText,
            # config={
            #     'resonse_mime_type': 'application/json',
            #     'response_schema': json_schema
            # }
        )
        response = response.text
        return response
    
    def makeLLMRequest(self, queryText):
        usage = None
        if self.family == "gemini":
            # response = self.chat.send_message(
            #     content=queryText,
            #     safety_settings={
            #         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            #         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            #     },
            #     # config={
            #     #     'resonse_mime_type': 'application/json',
            #     #     'response_schema': json_schema
            #     # }
            #     # generation_config=genai.types.GenerationConfig(
            #     #     # Only one candidate for now.
            #     #     # candidate_count=1,
            #     #     # stop_sequences=["x"],
            #     #     # max_output_tokens=20,
            #     #     temperature=0.0,
            #     #     resonse_mime_type = 'application/json',
            #     #     response_schema = json_schema
            #     # ),
            # )
            # response = response.text
            response =  self.llm.models.generate_content(
                model = self.model,
                contents = queryText,
            )
            usage = response.usage_metadata
            response = response.text
            # generator = generate.json(self.llm, Matching10)
            # response = generator(queryText)
            # print(response)

        elif self.family == 'llama':
            response = self.llm.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": queryText,
                    }
                ],
                model=self.model,
                # temperature=0
            )
            usage = response.usage
            response = response.choices[0].message.content

        elif self.family in ("chatgpt", "deepseek"):
            messages = [
                {"role": "user", "content": queryText},
            ]
            response = self.llm.create(messages=messages, model=self.model)
            self.tokens += response.usage.total_tokens
            usage = response.usage
            response = response.choices[0].message.content

        elif self.family == "claude":
            response = self.llm.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    # temperature=0,
                    messages=[
                        {"role": "user", "content": queryText}
                ]
            )
            response = response.content[0].text

        return response, usage
    
    def get_tokens_used(self):
        return self.tokens


class PromptGeneratorBase():
    def __init__(self) -> None:
        pass

    def correct_json_prompt(self, original, response, n, task_worker=False):
        prop = 'W' if task_worker else 'M'
        prompt = 'Previously, I gave you the following task:\n'
        prompt += '---------------------------------------------------------\n'
        prompt += original
        prompt += '---------------------------------------------------------\n'
        prompt += "In your response, you either failed to provide me with a matching or did not adhere to the JSON format I had asked for. Here are the last few lines of your response for reference:"
        prompt += '---------------------------------------------------------\n'
        prompt += response[-3000:]
        prompt += '---------------------------------------------------------\n'
        prompt += 'Please correct your response and provide me with the matching in the following JSON format, enclosed in <answer></answer> tags.'
        json_format = "{\n"
        for m in range(1, n):
            json_format += f"\t\"{prop}{m}\": \"<woman matched with {prop}{m}>\",\n"
        json_format += f"\t\"{prop}{m+1}\": \"<woman matched with {prop}{m+1}>\"\n"
        json_format += "}"
        prompt += f"<answer>\n\
{json_format}\n\
</answer>\n\n"
        if task_worker:
            prompt += "Make sure that each worker is assigned exactly ONE task."
        else:
            prompt += "Make sure that each man/woman is matched with exactly ONE partner."
        return prompt

    def correct_json_obj_prompt(self, original, response, verdict, n, task_worker=False):
        prop = 'W' if task_worker else 'M'
        prompt = 'Previously, I gave you the following task:\n'
        prompt += '---------------------------------------------------------\n'
        prompt += original
        prompt += '---------------------------------------------------------\n'
        prompt += "In your response, you failed adhere to the JSON format I had asked for. Here are the last few lines of your response for reference:"
        prompt += '---------------------------------------------------------\n'
        prompt += response[-3000:]
        prompt += '---------------------------------------------------------\n'
        prompt += verdict
        prompt += 'Please correct your response and provide me with the matching in the following JSON format, enclosed in <answer></answer> tags.'
        json_format = "{\n"
        for m in range(1, n):
            json_format += f"\t\"{prop}{m}\": \"<woman matched with {prop}{m}>\",\n"
        json_format += f"\t\"{prop}{m+1}\": \"<woman matched with {prop}{m+1}>\"\n"
        json_format += "}"
        prompt += f"<answer>\n\
{json_format}\n\
</answer>\n\n"
        if task_worker:
            prompt += "Make sure that each worker is assigned exactly ONE task."
        else:
            prompt += "Make sure that each man/woman is matched with exactly ONE partner."
        return prompt

    def incomplete_matching_prompt(self, original, response, counts, missing, n, task_worker=False):
        prop = 'W' if task_worker else 'M'
        extra = ''
        for woman in counts:
            if len(counts[woman]) > 1 and woman:
                men_list = [f'M{m}' for m in counts[woman]]
                extra = f"For example, W{woman} is matched with {', '.join(men_list[:-1])}, and {men_list[-1]}."
                break
        women_list = [f'W{woman}' for woman in missing]
        if len(women_list) > 1:
            missing_string = f"{', '.join(women_list[:-1])}, and {women_list[-1]} are unmatched."
        else:
            missing_string = f"{women_list[0]} is unmatched."
        prompt = 'Previously, I gave you the following task:'
        prompt += '\n---------------------------------------------------------\n'
        prompt += original
        prompt += '\n---------------------------------------------------------\n'
        if extra != '':
            prompt += f"In your response, the matching you selected involves some women being matched with multiple men, which is not allowed. {extra} Additionally, {missing_string} Here are the last few lines of your response for reference:"
        else:
            f"In your response, {missing_string} Here are the last few lines of your response for reference:"
        prompt += '\n---------------------------------------------------------\n'
        prompt += response[-3000:]
        prompt += '\n---------------------------------------------------------\n'
        prompt += 'Please correct your response and provide me with the matching in the following JSON format, enclosed in <answer></answer> tags.'
        json_format = "{\n"
        for m in range(1, n):
            json_format += f"\t\"{prop}{m}\": \"<woman matched with {prop}{m}>\",\n"
        json_format += f"\t\"{prop}{m+1}\": \"<woman matched with {prop}{m+1}>\"\n"
        json_format += "}"
        prompt += f"<answer>\n\
{json_format}\n\
</answer>\n\n"
        if task_worker:
            prompt += "Make sure that each worker is assigned exactly ONE task."
        else:
            prompt += "Make sure that each man/woman is matched with exactly ONE partner."
        return prompt

    def JSONMatchToList(self, json_match_string):
        json_match_string = json_match_string.replace(" ", "")
        if json_match_string.endswith("],]"):
            pairs = json_match_string[1:-2].split('],[')
        else:
            pairs = json_match_string[1:-1].split('],[')
        
        # print("IN JSON TO STR FUNC")
        # print(pairs)

        # Step 2: Extract the W number from each pair and convert to integer
        result = []
        for pair in pairs:
            # Split the pair and take the second element (Wj), then extract the number
            w_value_str = (pair.split(',')[1][1:])
            if w_value_str[-1] == "]":
                if w_value_str[-2] == "]":
                    w_value = int(w_value_str[:-2])
                else:
                    w_value = int(w_value_str[:-1])
            else:
                w_value = int(w_value_str)
            result.append(w_value)

        # Step 3: Print the resulting list
        return result
        # print(result)  # Output: [7, 2, 9, 5, 4, 1, 6, 8, 10, 3]

class PromptGeneratorPt2c():
    def __init__(self, instance_file, types = ['men_opt']) -> None:
        self.size = int(instance_file.split('_')[0])
        self.instance_file = f"../instances_matchings/{instance_file}"
        self.prompts_list = []
        self.type_index = {}
        self.types = types
        self.instances = pd.read_csv(self.instance_file)

    def set_type_indices(self):
        self.type_index = {typ: list(self.instances.columns).index(typ) for typ in self.types}
        
    def get_bp_prompt(self, instance, type_):
        n = instance[1]
        prompt = f"Consider the following instance of the two-sided matching problem, where {n} men are to be matched with {n} women. \n\
Here are the preference lists for all individuals:\n\n\
<preferences>\n\
{instance[3]}\n\
</preferences>\n\n\
Your task is to dermine how many blocking pairs (pairs that violate stability) are in the following matching.\n\n\
<matching>\n\
{instance[self.type_index[type_]]}\n\
</matching>\n\n\
Please return the number of blocking pairs you find (it is possible that the answer is 0), and enclose your answer in <answer></answer> tags.\
"
        return prompt

    def get_jaccard_prompt(self, instance, type_):
        n = instance[1]
        prompt = f"Consider the following instance of the two-sided matching problem, where {n} men are to be matched with {n} women. \n\
Here are the preference lists for all individuals:\n\n\
<preferences>\n\
{instance[3]}\n\
</preferences>\n\n\
Your task is to dermine the size of the intersection between the following matching and the proposal-optimal matching. Remember that the size of the intersection is computed as |A ∩ B|, where A and B are the set of pairs in each respective matching.\n\n\
<matching>\n\
{instance[self.type_index[type_]]}\n\
</matching>\n\n\
Please return the size you find (it is possible that the answer is 0), and enclose your answer in <answer></answer> tags.\
"
        return prompt
    
    def get_prompts_list(self):
        self.set_type_indices()
        # print(self.instances.values)
        # print("------------------")
        for i, instance in enumerate(self.instances.values):
            for type_ in self.types:
                prompt = self.get_bp_prompt(instance, type_)
                print(instance[self.type_index[type_]+1])
                self.prompts_list.append((i, prompt, type_, instance[self.type_index[type_]+1]))

                prompt = self.get_jaccard_prompt(instance, type_)
                begin = re.search(r"\d", type_)
                idx = -1 if not begin else begin.start()
                intersect_start = 0 if idx==-1 else int(type_[idx:])
                # print(intersect_start)
                self.prompts_list.append((i, prompt, type_, intersect_start))


class PromptGeneratorPt1(PromptGeneratorBase):
    def __init__(self, instance_file, num_instances=50) -> None:
        self.num_instances = num_instances
        self.size = int(instance_file.split('_')[0])
        self.instance_file = f"../instances_matchings/{instance_file}"
        self.prompts_list = []
        self.instances = pd.read_csv(self.instance_file)

    # def set_type_indices(self):
    #     self.type_index = {typ: list(self.instances.columns).index(typ) for typ in self.types}
        
    def format_matching(self, matching_dict):
        if not matching_dict:
            return ''
        matching_list = "["
        for man in range(1, len(matching_dict)+1):
            matching_list += f"[M{man}, W{matching_dict[man]}],"
        matching_list += "]"
        return matching_list

        
    def get_stable_match_generation_prompt(self, instance, include_aglo = False):
        n = instance[1]
        random_list = [i for i in range(1,n+1)]
        random.shuffle(random_list)
        random_matching = {i: random_list[i-1] for i in range(1, n+1)}
        algo_string = """ For this, you can use the Deferred Acceptance algorithm. The steps of this algorithm are described below:

1. Initialize all men and women as unmatched.
2. Create a list to keep track of each man's next proposal (initially set to 0 for all men).
3. While there are unmatched men:
   a. Select an unmatched man (M).
   b. Find the next woman (W) on M's preference list that he hasn't proposed to yet.
   c. If W is unmatched, match M and W.
   d. If W is matched but prefers M to her current partner:
      - Unmatch W from her current partner.
      - Match M and W.
      - Set the unmatched man as W's previous partner.
   e. If W rejects M, move to the next woman on M's preference list.
4. Repeat step 3 until all men are matched.\n\n"""

        json_format = "{\n"
        for m in range(1, n):
            json_format += f"\t\"M{m}\": \"<woman matched with M{m}>\",\n"
        json_format += f"\t\"M{m+1}\": \"<woman matched with M{m+1}>\"\n"
        json_format += "}"
        prompt = f"You are an intelligent assistant who is an expert in algorithms. Consider the following instance of the two-sided matching problem, where {n} men are to be matched with {n} women. \n\
Here are the preference lists for all individuals:\n\n\
<preferences>\n\
{instance[3]}\n\
</preferences>\n\n\
Your task is to find the proposer-optimal stable matching. \
{algo_string if include_aglo else ''}\
You can use XML tags like <scratchpad> to explain your thought process while computing the solution.\n\n\
Once you have found a stable matching, please return your matching in the JSON format given below:\n\n\
<answer>\n\
{json_format}\n\
</answer>\n\n\
Make sure that each man/woman is matched with exactly ONE partner. It is mandatory that you provide a matching as a JSON object enclosed in <answer></answer> tags as described above.\n\
"
        return prompt

    def get_task_worker_prompt(self, instance):
        n = instance[1]
        random_list = [i for i in range(1,n+1)]
        random.shuffle(random_list)
        random_matching = {i: random_list[i-1] for i in range(1, n+1)}

        worker_prefs = "{\nW: {\n"
        task_prefs = "T: {\n"
        worker_profile = instance[4].split('\n')
        task_profile = instance[5].split('\n')
        for i in range(n):
            worker_list = worker_profile[i].split(',')
            wlist = f'W{i+1}: ['
            for j in range(n-1):
                wlist += f'T{worker_list[j]}, '
            wlist += f'T{worker_list[n-1]}]'
            worker_prefs += wlist + '\n'
            task_list = task_profile[i].split(',')
            tlist = f'T{i+1}: ['
            for j in range(n-1):
                tlist += f'W{task_list[j]}, '
            tlist += f'W{task_list[n-1]}]'
            task_prefs += tlist + '\n'

        preference_profile = worker_prefs + '}\n' + task_prefs + '}' + '}'

        json_format = "{\n"
        for m in range(1, n):
            json_format += f"\t\"W{m}\": \"<task assigned to W{m}>\",\n"
        json_format += f"\t\"W{m+1}\": \"<task assigned to W{m+1}>\"\n"
        json_format += "}"
        prompt = f"You are an intelligent assistant who is an expert in algorithms. Consider the following instance of the two-sided matching problem, where {n} workers are to be assigned with {n} tasks, and each worker is assigned exactly one task. \n\
Here are the preference lists for all workers (W) over tasks (T) and the preferences of tasks over workers:\n\n\
<preferences>\n\
{preference_profile}\n\
</preferences>\n\n\
Your task is to find a stable matching of workers and tasks. \
You can use XML tags like <scratchpad> to explain your thought process while computing the solution.\n\n\
Once you have found a stable matching, please return your matching in the JSON format given below:\n\n\
<answer>\n\
{json_format}\n\
</answer>\n\n\
Make sure that each worker is assigned exactly ONE task. It is mandatory that you provide a matching as a JSON object enclosed in <answer></answer> tags as described above.\n\
"
        return prompt
        

    def get_stable_match_generation_prompt_ft(self, instance):
        n = instance[1]

        json_format = "{\n"
        for m in range(1, n):
            json_format += f"\t\"M{m}\": \"<woman matched with M{m}>\",\n"
        json_format += f"\t\"M{m+1}\": \"<woman matched with M{m+1}>\"\n"
        json_format += "}"
        question = f"Consider the following instance of the two-sided matching problem, where {n} men are to be matched with {n} women. \n\
Here are the preference lists for all individuals:\n\n\
<preferences>\n\
{instance[3]}\n\
</preferences>\n\n\
Your task is to find the proposer-optimal stable matching. You can use XML tags like <scratchpad> to explain your thought process while computing the solution.\n\n\
Once you have found a stable matching, please return your matching in the JSON format given below:\n\n\
<answer>\n\
{json_format}\n\
</answer>\n\n\
Make sure that each man/woman is matched with exactly ONE partner. It is mandatory that you provide a matching as a JSON object enclosed in <answer></answer> tags as described above.\n\
"
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are an intelligent assistant who is an expert in algorithms. Your task is to find the proposer-optimal stable matching, for the two-sided matching problem.  

### Question:
{question}

### Response:
<think>"""

        return prompt

    def get_prompts_list(self, include_aglo=False, task_worker = False):
        for i, instance in enumerate(self.instances.values[:self.num_instances]):
            if task_worker:
                prompt = self.get_task_worker_prompt(instance)
            else:
                prompt = self.get_stable_match_generation_prompt(instance, include_aglo)
            self.prompts_list.append((i, prompt, instance[6], int(instance[1]), instance[4], instance[5]))
    
    def get_prompts_list_ft(self):
        for i, instance in enumerate(self.instances.values[:self.num_instances]):
            prompt = self.get_stable_match_generation_prompt_ft(instance)
            self.prompts_list.append((i, prompt, instance[6], int(instance[1]), instance[4], instance[5]))
               

class PromptGeneratorPt2():
    def __init__(self, instance_file, types = ['men_opt', 'women_opt', 'lattice', 'random'], num_instances=50) -> None:
        self.num_instances = num_instances
        self.size = int(instance_file.split('_')[0])
        self.instance_file = f"../instances_matchings/{instance_file}"
        self.prompts_list = []
        self.type_index = {}
        self.types = types
        self.instances = pd.read_csv(self.instance_file)

    def set_type_indices(self):
        print(self.types)
        self.type_index = {typ: list(self.instances.columns).index(typ) for typ in self.types}
        
    def get_stability_prompt(self, instance, type):
        n = instance[1]
        prompt = f"Consider the following instance of the two-sided matching problem, where {n} men are to be matched with {n} women. \n\
Here are the preference lists for all individuals:\n\n\
<preferences>\n\
{instance[3]}\n\
</preferences>\n\n\
Your task is to dermine whether the following matching is stable or not.\n\n\
<matching>\n\
{instance[self.type_index[type]]}\n\
</matching>\n\n\
Please return 'Yes' if you think the provided matching is stable and 'No' if you think it is unstable, and enclose your answer in <answer></answer> tags.\
"
        return prompt
    
    def get_prompts_list(self):
        self.set_type_indices()
        for i, instance in enumerate(self.instances.values[:self.num_instances]):
            for type in self.types:
                prompt = self.get_stability_prompt(instance, type)
                self.prompts_list.append((i, prompt, type))


class PromptGeneratorPt2b(PromptGeneratorBase):
    def __init__(self, instance_file, types = ['random_1', 'random'], num_instances=50) -> None:
        self.num_instances = num_instances
        self.size = int(instance_file.split('_')[0])
        self.instance_file = f"../instances_matchings/{instance_file}"
        self.prompts_list = []
        self.type_index = {}
        self.types = types
        self.instances = pd.read_csv(self.instance_file)

    def set_type_indices(self):
        self.type_index = {typ: list(self.instances.columns).index(typ) for typ in self.types}
        
    def format_matching(self, matching_dict):
        if not matching_dict:
            return ''
        matching_list = "["
        for man in range(1, len(matching_dict)+1):
            matching_list += f"[M{man}, W{matching_dict[man]}],"
        matching_list += "]"
        return matching_list

        
    def get_stable_match_fix_prompt(self, instance, type_, give_bps = False):
        n = instance[1]
        random_list = [i for i in range(1,n+1)]
        random.shuffle(random_list)
        random_matching = {i: random_list[i-1] for i in range(1, n+1)}

        match_string = instance[self.type_index[type_]]
        unstable = self.JSONMatchToList(match_string)
        men_prefs_list = instance[4]
        women_prefs_list = instance[5]
        bp = blockingPairs(instance[1], men_prefs_list, women_prefs_list, np.array(unstable), "weak")
        bp_list = f"{bp['blockingPairs']}"
        bp_string = "["
        for man, woman in bp["blockingPairs"][:-1]:
            bp_string += "{M"+str(man)+", W"+str(woman)+"}, "
        bp_string += "{M"+str(bp["blockingPairs"][-1][0])+", W"+str(bp["blockingPairs"][-1][1])+"}"
        bp_string += "]"
        bp_string = f"\n\nNote that a matching is unstable when it has blocking pairs, i.e. pairs of men and women wh prefer to each to their partners in the matching. Here is the list of blocking pairs in the matching given above, to assist you with converting this into the proposer-optimal stable matching:\n{bp_string}\n\n"
        match_string = match_string.replace(']', '').replace('[', '')
        match_string = match_string.split(',')
        # print(match_string)
        unstable_json = "{\n"
        for m in range(n-1):
            unstable_json += f"\t\"{match_string[2*m].strip()}\": \"{match_string[2*m+1].strip()}\",\n"
        unstable_json += f"\t\"{match_string[2*n-2].strip()}\": \"{match_string[2*n-1].strip()}\"\n"
        unstable_json += "}"

        json_format = "{\n"
        for m in range(1, n):
            json_format += f"\t\"M{m}\": \"<woman matched with M{m}>\",\n"
        json_format += f"\t\"M{m+1}\": \"<woman matched with M{m+1}>\"\n"
        json_format += "}"
        prompt = f"You are an intelligent assistant who is an expert in algorithms. Consider the following instance of the two-sided matching problem and respective unstable matching, where {n} men are to be matched with {n} women. \n\
Here are the preference lists for all individuals:\n\n\
<preferences>\n\
{instance[3]}\n\
</preferences>\n\n\
Here is an unstable matching.\n\
<answer>\n\
{unstable_json}\n\
</answer>\n\n\
Your task is to modify the given unstable matching to make it equivalent to the proposer-optimal stable matching. You can use XML tags like <scratchpad> to explain your thought process while computing the solution.\n\n\{bp_string if give_bps else ''}\
Once you have found a stable matching, please return your matching in the JSON format given below:\n\n\
<answer>\n\
{json_format}\n\
</answer>\n\n\
Make sure that each man/woman is matched with exactly ONE partner. It is mandatory that you provide a matching as a JSON object enclosed in <answer></answer> tags as described above.\n\
"
        return prompt

    # def get_prompts_list(self):
    #     for i, instance in enumerate(self.instances.values):
    #         prompt = self.get_stable_match_generation_prompt(instance)
    #         self.prompts_list.append((i, prompt, instance[6], int(instance[1]), instance[4], instance[5]))

    def get_prompts_list(self, give_bps=False):
        self.set_type_indices()
        for i, instance in enumerate(self.instances.values[:self.num_instances]):
            for type_ in self.types:
                prompt = self.get_stable_match_fix_prompt(instance, type_, give_bps)
                self.prompts_list.append((i, prompt, instance[6], int(instance[1]), instance[4], instance[5], type_))


class PromptGeneratorPt3():
    def __init__(self, instance_file, num_instances = 50) -> None:
        self.num_instances = num_instances
        self.size = int(instance_file.split('_')[0])
        self.instance_file = f"../instances_matchings/{instance_file}"
        self.prompts_list = []
        self.instances = pd.read_csv(self.instance_file)

    # def set_type_indices(self):
    #     self.type_index = {typ: list(self.instances.columns).index(typ) for typ in self.types}
        
    def format_matching(self, matching_dict):
        if not matching_dict:
            return ''
        matching_list = "["
        for man in range(1, len(matching_dict)+1):
            matching_list += f"[M{man}, W{matching_dict[man]}],"
        matching_list += "]"
        return matching_list

        
    def get_lvl_1_prompt(self, instance):
        prompt = """You are an AI assistant tasked with analyzing preference profiles in a two-sided matching problem with one-to-one solutions. 
Your goal is to correctly interpret the given preference lists and answer a specific question about agent preferences.

First, here are the preference lists for all individuals:

<preferences>
{PREFERENCES}
</preferences>

Now, you will be asked a specific question about agent preferences:

<question>
{QUESTION}
</question>

Once you have determined the answer, provide your output in the following format:

1. The solution as a single agent name. For example, "W1"

Present your final answer within <answer> tags.

IMPORTANT: ONLY RETURN THE NAME OF THE SINGLE AGENT THAT IS THE ANSWER TO THE QUESTION. Do not include any explanations or additional information in your final answer.""".format(PREFERENCES=instance[3], QUESTION=instance[13])
        return prompt

    def get_lvl_2_prompt(self, instance):
        prompt = """You are an AI assistant tasked with analyzing preference profiles in a two-sided matching problem with one-to-one solutions. Your goal is to correctly interpret the given preference lists and answer a specific question about agent preferences.

First, here are the preference lists for all individuals:

<preferences>
{PREFERENCES}
</preferences>

Now, you will be asked a specific question about agent preferences:

<question>
{QUESTION}
</question>

Once you have determined the answer, provide your output in the following format:

1. The solution as a YES or a NO. For example, "NO"

Present your final answer within <answer> tags.

IMPORTANT: ONLY RETURN YES OR NO THAT IS THE ANSWER TO THE QUESTION. Do not include any explanations or additional information in your final answer.""".format(PREFERENCES=instance[3], QUESTION=instance[15])
        return prompt
    
    def get_lvl_2n_prompt(self, instance):
        prompt = """You are an AI assistant tasked with analyzing preference profiles in a two-sided matching problem with one-to-one solutions. Your goal is to correctly interpret the given preference lists and answer a specific question about agent preferences.

First, here are the preference lists for all individuals:

<preferences>
{PREFERENCES}
</preferences>

Now, you will be asked a specific question about agent preferences:

<question>
{QUESTION}
</question>

Once you have determined the answer, provide your output in the following format:

1. The solution as a YES or a NO. For example, "NO"

Present your final answer within <answer> tags.

IMPORTANT: ONLY RETURN YES OR NO THAT IS THE ANSWER TO THE QUESTION. Do not include any explanations or additional information in your final answer.""".format(PREFERENCES=instance[3], QUESTION=instance[17])
        return prompt

    def get_prompts_list(self):
        for i, instance in enumerate(self.instances.values[:self.num_instances]):
            # men_prefs_list, women_prefs_list, men_prefs_dict, women_prefs_dict = populate_prefs(instance)
            # print(gale_shapley(men_prefs_list, women_prefs_list, men_prefs_dict, women_prefs_dict, verbose=True))
            prompt = self.get_lvl_1_prompt(instance)
            self.prompts_list.append((i, prompt, 'level_1', instance[14], instance[13]))
            prompt = self.get_lvl_2_prompt(instance)
            self.prompts_list.append((i, prompt, 'level_2', instance[16], instance[15]))
            prompt = self.get_lvl_2n_prompt(instance)
            self.prompts_list.append((i, prompt, 'level_2_noisy', instance[18], instance[17]))


class PromptGeneratorPt4(PromptGeneratorBase):
    def __init__(self, instance_file, num_instances=50) -> None:
        self.num_instances = num_instances
        self.size = int(instance_file.split('_')[0])
        self.instance_file = f"../instances_matchings/{instance_file}"
        if 'womanmaster' in instance_file:
            self.instance_file_ex = "../instances_matchings/5_womanmaster_processed.csv"
        else: 
            self.instance_file_ex = "../instances_matchings/5_ic_processed.csv"
    
        self.prompts_list = []
        self.instances = pd.read_csv(self.instance_file)
        self.examples = pd.read_csv(self.instance_file_ex)
        
    def format_matching(self, matching_dict):
        if not matching_dict:
            return ''
        matching_list = "["
        for man in range(1, len(matching_dict)+1):
            matching_list += f"[M{man}, W{matching_dict[man]}],"
        matching_list += "]"
        return matching_list

        
    def get_cot_vanilla_prompt(self, instance, example_instance):
        n = instance[1]
        m_pref = instance[4]
        w_pref = instance[5]
        m_pref_example = example_instance[4]
        w_pref_example = example_instance[5]

        men_prefs_list, women_prefs_list, men_prefs_dict, women_prefs_dict = populate_prefs(example_instance)
        comments_list = gale_shapley(men_prefs_list, women_prefs_list, men_prefs_dict, women_prefs_dict, verbose=True)
        comments_str = "\n".join(comments_list)

        match_sample = weaklyStableMatch2(example_instance[1], m_pref_example, w_pref_example)

        json_sample_sol = "{\n"
        for m in range(1, example_instance[1]):
            json_sample_sol += f"\t\"M{m}\": \"W{match_sample.findMan(m-1).engagedWith[0]+1}\",\n"
        json_sample_sol += f"\t\"M{m+1}\": \"W{match_sample.findMan(m-1+1).engagedWith[0]+1}\"\n"
        json_sample_sol += "}"

        random_list = [i for i in range(1,n+1)]
        random.shuffle(random_list)
        random_matching = {i: random_list[i-1] for i in range(1, n+1)}

        json_format = "{\n"
        for m in range(1, n):
            json_format += f"\t\"M{m}\": \"<woman matched with M{m}>\",\n"
        json_format += f"\t\"M{m+1}\": \"<woman matched with M{m+1}>\"\n"
        json_format += "}"
        prompt = f"You are an intelligent assistant who is an expert in algorithms. Your task is to find the proposer-optimal stable matching, for the two-sided matching problem. Here is an example to demonstrate how you should proceed:\n\
<example>\n\
<preferences>\n\
{example_instance[3]}\n\
</preferences>\n\n\
{comments_str}\n\
<answer>\n\
{json_sample_sol}\n\
</answer>\n\n\
</example>\n\n\
Consider the following instance of the two-sided matching problem, where {n} men are to be matched with {n} women. \n\
Here are the preference lists for all individuals:\n\n\
<preferences>\n\
{instance[3]}\n\
</preferences>\n\n\
Your task is to find the proposer-optimal stable matching.\n\n\
Once you have found a stable matching, please return your matching in the JSON format given below:\n\n\
<answer>\n\
{json_format}\n\
</answer>\n\n\
Make sure that each man/woman is matched with exactly ONE partner. It is important that you enclose your JSON object in <answer></answer> tags.\
"
        return prompt


    def get_cot_shortlist_prompt(self, instance, example_instance):
        n = instance[1]
        m_pref = instance[4]
        w_pref = instance[5]
        m_pref_example = example_instance[4]
        w_pref_example = example_instance[5]
        comments_list = weaklyStableMatch2(example_instance[1], m_pref_example, w_pref_example, verbose=True)
        comments_str = "\n".join(comments_list)
        match_sample = weaklyStableMatch2(example_instance[1], m_pref_example, w_pref_example)

        json_sample_sol = "{\n"
        for m in range(1, example_instance[1]):
            json_sample_sol += f"\t\"M{m}\": \"W{match_sample.findMan(m-1).engagedWith[0]+1}\",\n"
        json_sample_sol += f"\t\"M{m+1}\": \"W{match_sample.findMan(m-1+1).engagedWith[0]+1}\"\n"
        json_sample_sol += "}"

        # print(json_sample_sol)

        random_list = [i for i in range(1,n+1)]
        random.shuffle(random_list)
        random_matching = {i: random_list[i-1] for i in range(1, n+1)}

        json_format = "{\n"
        for m in range(1, n):
            json_format += f"\t\"M{m}\": \"<woman matched with M{m}>\",\n"
        json_format += f"\t\"M{m+1}\": \"<woman matched with M{m+1}>\"\n"
        json_format += "}"
        prompt = f"You are an intelligent assistant who is an expert in algorithms. Your task is to find the proposer-optimal stable matching, for the two-sided matching problem. Here is an example to demonstrate how you should proceed:\n\
<example>\n\
<preferences>\n\
{example_instance[3]}\n\
</preferences>\n\n\
{comments_str}\n\
<answer>\n\
{json_sample_sol}\n\
</answer>\n\n\
</example>\n\n\
Consider the following instance of the two-sided matching problem, where {n} men are to be matched with {n} women. \n\
Here are the preference lists for all individuals:\n\n\
<preferences>\n\
{instance[3]}\n\
</preferences>\n\n\
Your task is to find the proposer-optimal stable matching.\n\n\
Once you have found a stable matching, please return your matching in the JSON format given below:\n\n\
<answer>\n\
{json_format}\n\
</answer>\n\n\
Make sure that each man/woman is matched with exactly ONE partner. It is important that you enclose your JSON object in <answer></answer> tags.\
"
        return prompt


    def get_fewshot_prompt(self, instance, example_instances):
        n = instance[1]
        m_pref = instance[4]
        w_pref = instance[5]

        examples = ""

        for example_instance in example_instances:
            example_instance = example_instances[0]
            m_pref_example = example_instance[4]
            w_pref_example = example_instance[5]
            match_sample = weaklyStableMatch2(example_instance[1], m_pref_example, w_pref_example)

            json_sample_sol = "{\n"
            for m in range(1, example_instance[1]):
                json_sample_sol += f"\t\"M{m}\": \"W{match_sample.findMan(m-1).engagedWith[0]+1}\",\n"
            json_sample_sol += f"\t\"M{m+1}\": \"W{match_sample.findMan(m-1+1).engagedWith[0]+1}\"\n"
            json_sample_sol += "}"

            examples += f"<example>\n\
<preferences>\n\
{example_instance[3]}\n\
</preferences>\n\n\
<answer>\n\
{json_sample_sol}\n\
</answer>\n\
</example>\n\n"

        # print(examples)

        random_list = [i for i in range(1,n+1)]
        random.shuffle(random_list)
        random_matching = {i: random_list[i-1] for i in range(1, n+1)}

        json_format = "{\n"
        for m in range(1, n):
            json_format += f"\t\"M{m}\": \"<woman matched with M{m}>\",\n"
        json_format += f"\t\"M{m+1}\": \"<woman matched with M{m+1}>\"\n"
        json_format += "}"
        prompt = f"You are an intelligent assistant who is an expert in algorithms. Your task is to find the proposer-optimal stable matching, for the two-sided matching problem. Here are some examples to that might help you understand how to proceed:\n\
{examples}\n\
Consider the following instance of the two-sided matching problem, where {n} men are to be matched with {n} women. \n\
Here are the preference lists for all individuals:\n\n\
<preferences>\n\
{instance[3]}\n\
</preferences>\n\n\
Your task is to find the proposer-optimal stable matching.\n\n\
Once you have found a stable matching, please return your matching in the JSON format given below:\n\n\
<answer>\n\
{json_format}\n\
</answer>\n\n\
Make sure that each man/woman is matched with exactly ONE partner. It is important that you enclose your JSON object in <answer></answer> tags.\
"
        return prompt

    def get_prompts_list(self):
        for i, instance in enumerate(self.instances.values[:self.num_instances]):
            # men_prefs_list, women_prefs_list, men_prefs_dict, women_prefs_dict = populate_prefs(instance)
            # print(gale_shapley(men_prefs_list, women_prefs_list, men_prefs_dict, women_prefs_dict, verbose=True))
            idx = random.randint(0,len(self.examples)-1)
            while idx == i:
                idx = random.randint(0,len(self.examples)-1)
            example_instance = self.examples.values[idx]
            prompt = self.get_cot_shortlist_prompt(instance, example_instance)
            self.prompts_list.append((i, prompt, instance[6], int(instance[1]), instance[4], instance[5], "cot_shortlist"))

            prompt = self.get_cot_vanilla_prompt(instance, example_instance)
            self.prompts_list.append((i, prompt, instance[6], int(instance[1]), instance[4], instance[5], "cot_vanilla"))


            idxs = random.sample(range(len(self.examples)), 3)
            while i in idxs:
                idxs = random.sample(range(len(self.examples)), 3)
            example_instances = [self.examples.values[e] for e in idxs]
            prompt = self.get_fewshot_prompt(instance, example_instances)
            self.prompts_list.append((i, prompt, instance[6], int(instance[1]), instance[4], instance[5], "fewshot_3"))

    
               
       