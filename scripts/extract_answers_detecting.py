from google import genai
import pandas as pd
import enum
from pydantic import BaseModel, ConfigDict
from enum import Enum
import json
import csv
from tqdm import tqdm
import time
from BlockingPairs import blockingPairs
import numpy as np
import os

from components import LLM


def generate_prompt(previous_prompt, previous_response):

    prompt = f"""Previously, I had asked an LLM and the following question:

{previous_prompt}
    
However, the LLM failed to give me a valid solution in the above-mentioned format. Given below are the final few lines of it's response:
----------------------------------------------------------------
{previous_response}
----------------------------------------------------------------
Please extract the answer that the LLM intended to give in its response, and return in the the correct format. Hence, you need to return either <answer>Yes</answer> or <answer>No</answer>, based on the LLM's response. 
"""
    return prompt

client = genai.Client(api_key="API_KEY")
result_dir = '../evaluating_responses/part_2/'
models = [
    # 'gemini20',
    # 'gemini25',
    # 'llama33',
    # 'qwen_qwq',
    # 'deepseek_dist',
    # 'o3-mini',
    'deepseek',
]
for model in models:
    print(F'STARTING {model}')
    corrected = [['Culture', 'Size', 'Instance', 'Type', 'Answer', 'Correctness', 'Response']]
    result_files = os.listdir(result_dir)
    model_files = [filename for filename in result_files if model in filename and 'corrected' not in filename and 'cleaned' not in filename and 'csv' in filename and 'dist' not in filename]
    for model_file in model_files:
        data = pd.read_csv(result_dir+model_file)
        print(f"STARTING {model_file}")
        for i, row in tqdm(enumerate(data.values)):

            culture = row[0]
            size = row[1]
            instance = row[2]
            old_prompt = row[3]
            ptype = row[4] 
            answer = row[5] 
            correct = row[6] 
            old_response = row[7] if 'gemini' in model else row [-1]

            if answer != -1:
                corrected.append([culture, size, instance, ptype, answer, correct, old_response])
                continue

            if '</think>' in old_response: old_response = old_response.split('</think>')[1]
            
            print(old_response[-1000:].lower().count('yes'), old_response[-1000:].lower().count('no'))
            if old_response[-1000:].lower().count('yes') == old_response[-1000:].lower().count('no'):
                print('$$$$$$ EMPTY RESPONSE $$$$$$$')
                print(old_response[-200:])
                corrected.append([culture, size, instance, ptype, -1, 0, old_response])
                continue

            
            prompt = generate_prompt(old_prompt, old_response[-1000:])
            if size == 5: continue
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
            )
            response = response.text
            correct = 0
            ans_extract = response[response.rfind('<answer>'):response.rfind('</answer>')]
            if 'Yes' in ans_extract:
                answer = 1
            elif 'No' in ans_extract:
                answer = 0
            else:
                answer = -1
            if ptype in ['men_opt', 'women_opt', 'lattice'] and answer:
                correct = 1
            elif 'random' in ptype and answer == 0:
                correct = 1
            corrected.append([culture, size, instance, ptype, answer, correct, old_response])
            print(old_response[-1000:], answer, response, ptype)

    with open(result_dir + f"{model}_corrected.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(corrected) 
