import os
os.environ['HF_HOME'] = ""

import torch
from huggingface_hub import login
from transformers import pipeline

from vllm import LLM, SamplingParams

from components import PromptGeneratorPt3
import os
import csv
from BlockingPairs import blockingPairs
import numpy as np
import json
from collections import defaultdict
import math

specifications = {
    'llama32': "meta-llama/Llama-3.2-3B-Instruct",
    'llama33': "meta-llama/Llama-3.3-70B-Instruct",
    'qwen_qwq': "Qwen/QwQ-32B-Preview",
    'qwen_qwqs': "Qwen/QwQ-32B",
    'gemma3': "google/gemma-3-4b-it",
    'deepseek_dist': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'ds_llama_8b': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    'ds_qwen_14b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    'ds_llama_8b_v1': 'omitted to preserve anonymity',
    'ds_llama_8b_full': 'omitted to preserve anonymity',
    'ds_qwen_14b_full': 'omitted to preserve anonymity',
    'qwen_32b_full': "omitted to preserve anonymity",
    'qwen_32b_full_da': "omitted to preserve anonymity",
}

model_name = 'qwen_32b_full_da'

access_token = ""
login(token=access_token, add_to_git_credential=True, new_session=False)
model_id = specifications[model_name]
model = LLM(model=model_id,
            tensor_parallel_size=2,
            # gpu_memory_utilization=0.97,
            # dtype="float16",
            )
if model_name == "qwen_qwq":
    sampling_params = SamplingParams(temperature=0.6,
                                     max_tokens=10000,
                                     presence_penalty=0.2,
                                     top_p=0.95,
                                    )
else:
    sampling_params = SamplingParams(temperature=0.5,
                                     max_tokens=2000,
                                    )    

messages = []
# text = model.apply_chat_template(messages, add_genenration_prompt=True, tokenize=False)
prompt_num = 0
prompts_map = {}

print("LISTING PROMPTS")

data = [['Culture', 'Size', 'Instance', 'Question', 'Question_Type', 'Prompt', 'Correct', 'Correctness', 'Response']]
instance_files = os.listdir('instances_matchings/')
for instance_file in instance_files:
    if '.csv' not in instance_file: 
        continue
    print("READING", instance_file)

    culture = instance_file.split('_')[1]
    size = int(instance_file.split('_')[0])
    pg = PromptGeneratorPt3(instance_file, num_instances=50)

    pg.get_prompts_list()
    prompts_list = pg.prompts_list

    for prompt in prompts_list:
        print(f"INSTANCE: {prompt[0]}, TYPE: {prompt[2]}")
        messages.append([{"role": "user", "content": prompt[1]}])

        prompts_map[prompt_num] = [culture, size, prompt[0], prompt[4], prompt[2], prompt[1], prompt[3]]
        prompt_num += 1


for r in range(2):
    outputs = model.chat(messages, sampling_params=sampling_params,)
    messages = []
    prompt_num = 0

    print(f"GENERATED RESPONSES - ROUND {r+1}")
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        prompt_row = prompts_map[i]
        correct = 0
        if '<answer>' in response and '</answer>' in response:
            answer = response.split('<answer>')[1].split('</answer>')[0].lower().strip()
            correct = -1
            if prompt_row[-1].lower().strip() in answer:
                correct = 1
            else:
                correct = 0
            prompt_row.extend([correct, response])
            data.append(prompt_row)
        else:
            new_prompt = pg.incorrect_qa_format_prompt(prompt_row[-2], response)
            messages.append([{"role": "user", "content": new_prompt}])
            prompts_map[prompt_num] = prompt_row
            prompt_num += 1

outputs = model.chat(messages, sampling_params=sampling_params,)
print("GENERATED RESPONSES - FINAL")
for i, output in enumerate(outputs):
    response = output.outputs[0].text
    prompt_row = prompts_map[i]
    # print(response)
    if '<answer>' in response and '</answer>' in response:
        answer = response.split('<answer>')[1].split('</answer>')[0].lower().strip()
        correct = -1
        if prompt_row[-1].lower().strip() in answer:
            correct = 1
        else:
            correct = 0
    else:
        correct = int(prompt[3].lower().strip() in response.lower().strip())
    print(f"{prompt_row[-1]}, {answer}")
    prompt_row.extend([correct, response])
    data.append(prompt_row)

with open(f'evaluating_responses/part_3/{model_name}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)
