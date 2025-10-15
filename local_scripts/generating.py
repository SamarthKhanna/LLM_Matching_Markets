import os
os.environ['HF_HOME'] = ""

import torch
from huggingface_hub import login
from transformers import pipeline

from vllm import LLM, SamplingParams

from components import PromptGeneratorPt1
import os
import csv
from BlockingPairs import blockingPairs
import numpy as np
import json
from collections import defaultdict

specifications = {
    'llama32': "meta-llama/Llama-3.2-3B-Instruct",
    'llama33': "meta-llama/Llama-3.3-70B-Instruct",
    'qwen_qwq': "Qwen/QwQ-32B-Preview",
    'qwen_qwqs': "Qwen/QwQ-32B",
    'gemma3': "google/gemma-3-4b-it",
    'deepseek_dist': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'ds_llama_8b': 'unsloth/DeepSeek-R1-Distill-Llama-8B',
    'ds_llama_ft': 'omitted to preserve anonymity',
    'ds_llama_ft_mid': "omitted to preserve anonymity",
    'deepseek_qwen_14b': "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    'ds_qwen_14b_ft': "omitted to preserve anonymity",
    'qwen_qwq_16': "omitted to preserve anonymity",
    'qwen_qwq_32': "omitted to preserve anonymity",
    'qwen_qwq_32_lar': "omitted to preserve anonymity",
    'ds_llama_8b_16': "omitted to preserve anonymity",
    'ds_llama_8b_32': "omitted to preserve anonymity",
    'ds_qwen_14b_16': "omitted to preserve anonymity",
    'ds_qwen_14b_32': "omitted to preserve anonymity",
    'ds_qwen_14b_full': "omitted to preserve anonymity",
    'qwen_32b_full': "omitted to preserve anonymity",
    'qwen_comp': 'omitted to preserve anonymity'
}

model_name = 'qwen_comp'
access_token = ""
login(token=access_token, add_to_git_credential=True, new_session=False)


model_id = specifications[model_name]
model = LLM(model=model_id,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.97,
            # dtype="float16",
            )
if model_name == "qwen_qwq":
    sampling_params = SamplingParams(temperature=0.6,
                                    max_tokens=100000,
                                    presence_penalty=0.2,
                                    top_p=0.95,
                                    )
else:
    sampling_params = SamplingParams(temperature=0.5,
                                    max_tokens=30000,
                                    )    

def JSONMatchToList(json_match_string):
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

def JSONobjToList(json_obj, n):
    if not json_obj:
        print("EMPTY JSON OBJ")
        return [], 'empty'
    matching = []
    for m in range(1, n+1):
        mstring = f"M{m}"
        if mstring not in json_obj or not json_obj[mstring] or (type(json_obj[mstring]) == str and json_obj[mstring].lower().strip() == 'none'):
            matching.append(0)
        else:
            try:
                woman = int(json_obj[mstring][1:])
                matching.append(woman)
            except:
                print(f"WRONG VALUE FORMAT: {json_obj[mstring]}")
                return matching, f"For example, {mstring}'s match is given the incorrect format - {json_obj[mstring]}. "
    return matching, "okay"

def jaccard_similarity(list1, list2):
    set1 = set()
    set2 = set()
    for m, w in enumerate(list1):
        set1.add((m+1, w))
    for m, w in enumerate(list2):
        set2.add((m+1, w))

    jaccard_sim = len(set1.intersection(set2))/len(set1.union(set2))

    return jaccard_sim, len(set1.intersection(set2))

messages = []
# text = model.apply_chat_template(messages, add_genenration_prompt=True, tokenize=False)
prompt_num = 0
prompts_map = {}

print("LISTING PROMPTS")

data = [['Culture', 'Size', 'Instance', 'Prompt', 'Men_prefs', 'Women_prefs', 'Men_opt', 'Answer', 'Correctness','Blocking_Pair_Count', 'Blocking_Pair_List', 'Jaccard_Similarity', "Intersection", 'Response', "Num_tries", 'Remarks']]
instance_files = os.listdir('instances_matchings/')
for instance_file in instance_files:
    if '.csv' not in instance_file: 
        continue
    print("READING", instance_file)

    culture = instance_file.split('_')[1]
    size = int(instance_file.split('_')[0])

    # if size in [20,50]: continue
    
    pg = PromptGeneratorPt1(instance_file, num_instances=50)

    if '16' in model_name or '32' in model_name or 'full' in model_name:
        print('USING FT PROMPT')
        pg.get_prompts_list_ft()
    else:
        pg.get_prompts_list()
    prompts_list = pg.prompts_list

    for prompt in prompts_list:
        # print(f"INSTANCE: {prompt[0]}")
        messages.append([{"role": "user", "content": prompt[1]}])

        prompts_map[prompt_num] = [culture, size, prompt[0], prompt[1], prompt[4], prompt[5], prompt[2], prompt[1], {}]
        prompt_num += 1


for r in range(2):
    outputs = model.chat(messages, sampling_params=sampling_params,)
    messages = []
    prompt_num = 0

    print(f"GENERATED RESPONSES - ROUND {r+1}")
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        # print(response)
        correct = 0
        prompt_row = prompts_map[i]
        culture = prompt_row[0]
        size = prompt_row[1]
        instance = prompt_row[2]
        og_prompt = prompt_row[3]
        men_prefs = prompt_row[4]
        women_prefs = prompt_row[5]
        men_opt = prompt_row[6]
        current_prompt = [7]
        incomplete = prompt_row[8]
        prompt_row = prompt_row[:-2]
        
        try:
            answer_ext = response[response.rfind('<answer>')+8:response.rfind('</answer>')]
            answer = answer_ext[answer_ext.index('{'):answer_ext.index('}')+1]
            # print(answer)
            answer = json.loads(answer)
            # print(answer)
        except:
            print("INCORRECT JSON FORMAT")
            # print(response[response.rfind('<answer>')+8:response.rfind('</answer>')])
            new_prompt = pg.correct_json_prompt(og_prompt, response, size)
            # print(new_prompt)
            messages.append([{"role": "user", "content": new_prompt}])
            prompts_map[prompt_num] = [culture, size, instance, og_prompt, men_prefs, women_prefs, men_opt, new_prompt, incomplete]
            prompt_num += 1
            continue

        man_opt_list = JSONMatchToList(men_opt)
        llm_answer_list, verdict = JSONobjToList(answer, size)
            
        if verdict not in ['empty', 'okay']:
            print("FORMATTING ERROR!")
            # print(answer_ext)
            # print(answer)
            # print(verdict)
            new_prompt = pg.correct_json_obj_prompt(og_prompt, response, verdict, size)
            # print(new_prompt)
            messages.append([{"role": "user", "content": new_prompt}])
            prompts_map[prompt_num] = [culture, size, instance, og_prompt, men_prefs, women_prefs, men_opt, new_prompt, incomplete]
            prompt_num += 1
            continue

        try:
            js, inter = jaccard_similarity(man_opt_list, llm_answer_list)
        except:
            js, inter = -1, 0

        if len(set(llm_answer_list)) < size or 0 in llm_answer_list:
            counts = defaultdict(list)
            invalid = False
            for m, woman in enumerate(llm_answer_list):
                counts[woman].append(m+1)
                if woman != 0 and len(counts[woman]) > 1:
                    invalid = True
            new_prompt = pg.incomplete_matching_prompt(og_prompt, response, counts, set([w+1 for w in range(size)]) - set(llm_answer_list), size)
            if invalid:
                print('INVALID MATCHING!')
                # print(answer)
                # print(counts)
            else:
                incomplete = answer
                print(f"INCOMPLETE MATCHING: {set([w+1 for w in range(size)]) - set(llm_answer_list)} missing.")
                # print(answer)
            # print(new_prompt)
            messages.append([{"role": "user", "content": new_prompt}])
            prompts_map[prompt_num] = [culture, size, instance, og_prompt, men_prefs, women_prefs, men_opt, new_prompt, incomplete]
            prompt_num += 1
            continue
        
        bp = blockingPairs(size, men_prefs, women_prefs, np.array(llm_answer_list), "weak")
        bp_count = bp["blockingPairCount"]
        bp_list = f"{bp['blockingPairs']}"
        correct = 1 if bp_count == 0 else 0
        
        prompt_row.extend([answer, correct, bp_count, bp_list, js, inter, response, r+1, "Processed smoothly."])
        
        # print(f"{prompt_row[-2]}, {answer}")
        data.append(prompt_row)

outputs = model.chat(messages, sampling_params=sampling_params,)
print("GENERATED RESPONSES - FINAL")
for i, output in enumerate(outputs):
    response = output.outputs[0].text
    # print(response)
    correct = 0
    prompt_row = prompts_map[i]
    culture = prompt_row[0]
    size = prompt_row[1]
    instance = prompt_row[2]
    og_prompt = prompt_row[3]
    men_prefs = prompt_row[4]
    women_prefs = prompt_row[5]
    men_opt = prompt_row[6]
    current_prompt = prompt_row[7]
    incomplete = prompt_row[8]
    prompt_row = prompt_row[:-2]
    
    try:
        answer = response[response.rfind('<answer>')+8:response.rfind('</answer>')]
        answer = answer[answer.index('{'):answer.index('}')+1]
        # print(answer)
        answer = json.loads(answer)
        # print(answer)
    except:
        print("INCORRECT JSON FORMAT!")
        # print(response[-2000:])
        if incomplete:
            data.append([culture, size, instance, og_prompt, men_prefs, women_prefs, men_opt, incomplete, 0, 0, '', 0, 0, response, 3, "INCOMPLETE MATCHING!"])
        else:
            data.append([culture, size, instance, og_prompt, men_prefs, women_prefs, men_opt, -1, 0, 0, '', 0, 0, response, 3, "INCORRECT JSON FORMAT!"])
        continue

    man_opt_list = JSONMatchToList(men_opt)
    llm_answer_list, verdict = JSONobjToList(answer, size)
    
    if verdict not in ['empty', 'okay']:
        print("FORMATTING ERROR!")
        # print(answer)
        # print(verdict)
        if incomplete:
            data.append([culture, size, instance, og_prompt, men_prefs, women_prefs, men_opt, incomplete, 0, 0, '', 0, 0, response, 3, "INCOMPLETE MATCHING!"])
        else:
            data.append([culture, size, instance, og_prompt, men_prefs, women_prefs, men_opt, -1, 0, 0, '', 0, 0, response, 3, "FORMATTING ERROR!"])
        continue

    try:
        js, inter = jaccard_similarity(man_opt_list, llm_answer_list)
    except:
        js, inter = -1, 0

    if len(set(llm_answer_list)) < size or 0 in llm_answer_list:
        # print(llm_answer_list)
        counts = defaultdict(list)
        invalid = False
        for m, woman in enumerate(llm_answer_list):
            counts[woman].append(m+1)
            if woman != 0 and len(counts[woman]) > 1:
                invalid = True
        if invalid:
            if incomplete:
                data.append([culture, size, instance, og_prompt, men_prefs, women_prefs, men_opt, incomplete, 0, 0, '', 0, 0, response, 3, "INCOMPLETE MATCHING!"])
            print(f"INVALID MATCHING!")
            # print(answer)
            # print(counts)
            data.append([culture, size, instance, og_prompt, men_prefs, women_prefs, men_opt, answer, 0, 0, '', 0, 0, response, 3, "INVALID MATCHING!"])
        else:
            data.append([culture, size, instance, og_prompt, men_prefs, women_prefs, men_opt, answer, 0, 0, '', 0, 0, response, 3, "INCOMPLETE MATCHING!"])
            print(f"INCOMPLETE MATCHING: {set([w+1 for w in range(size)]) - set(llm_answer_list)} missing.")
        continue

    try:
        bp = blockingPairs(size, men_prefs, women_prefs, np.array(llm_answer_list), "weak")
        bp_count = bp["blockingPairCount"]
        bp_list = f"{bp['blockingPairs']}"
        correct = 1 if bp_count == 0 else 0
    except:
        if incomplete:
            data.append([culture, size, instance, og_prompt, men_prefs, women_prefs, men_opt, incomplete, 0, 0, '', 0, 0, response, 3, "INCOMPLETE MATCHING!"])
        else:
            data.append([culture, size, instance, og_prompt, men_prefs, women_prefs, men_opt, -1, 0, 0, '', 0, 0, response, 3, "ERROR COMPUTING BLOCKING PAIRS!!"])
        continue

    data.append([culture, size, instance, og_prompt, men_prefs, women_prefs, men_opt, answer, correct, bp_count, bp_list, js, inter, response, 3, "Processed smoothly."])

with open(f'evaluating_responses/part_1/{model_name}_repeat_30k.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)
