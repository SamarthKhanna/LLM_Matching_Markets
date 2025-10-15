from components import LLM, PromptGeneratorPt1
import os
import csv
from tqdm import tqdm
import math
from BlockingPairs import blockingPairs
from collections import defaultdict
import copy
import json
import numpy as np
import pickle as pkl
import shutil

specifications = {
    'llama33': ('../Auth_keys/llama_auth_key.txt', 'llama', 'llama-3.3-70b-versatile'),
    'llama3': ('../Auth_keys/llama_auth_key.txt', 'llama', 'llama3-70b-8192'),
    'deepseek_dist': ('../Auth_keys/llama_auth_key.txt', 'llama', 'deepseek-r1-distill-llama-70b'),
    'qwen': ('../Auth_keys/llama_auth_key.txt', 'llama', 'qwen-qwq-32b'),
    'gemini': ('../Auth_keys/gemini_auth_key_2.txt', 'gemini', 'gemini-1.5-pro'),
    'gemini25': ('../Auth_keys/gemini_auth_key_2.txt', 'gemini', 'gemini-2.5-pro-preview-03-25'),
    'gemini20': ('../Auth_keys/gemini_auth_key_2.txt', 'gemini', 'gemini-2.0-flash'),
    'o3-mini': ('../Auth_keys/openai_auth_key_fairlab.txt', 'chatgpt', 'o3-mini'),
    'o4-mini': ('../Auth_keys/openai_auth_key_fairlab.txt', 'chatgpt', 'o4-mini'),
    'deepseek': ('../Auth_keys/deepseek_auth_key.txt', 'deepseek', 'deepseek-reasoner')
}

model_type = 'deepseek'
auth_key_path, model_family, model = specifications[model_type]

llm = LLM(auth_file=auth_key_path, family=model_family, model=model)

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
        if mstring not in json_obj or not json_obj[mstring] or json_obj[mstring].lower().strip() == 'none':
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


model_folder = f"../evaluating_responses/part_1b/{model_type}/"
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

instance_files = os.listdir('../instances_matchings/')
for instance_file in instance_files:
    print("GENERATING RESPONSES FOR ", instance_file, model_type)
    data = [['Culture', 'Size', 'Instance', 'Prompt', 'Answer', 'Correctness','Blocking_Pair_Count', 'Blocking_Pair_List', 'Jaccard_Similarity', "Intersection", 'Response', 'Input_Tokens', 'Output_Tokens', "Num_tries", 'Remarks']]
    if '.csv' not in instance_file: 
        continue
    # print(instance_file)
    culture = instance_file.split('_')[1]
    size = int(instance_file.split('_')[0])
    result_file = f'../evaluating_responses/part_1b/{model_type}_{culture}_{size}_repeat.csv'
    if os.path.exists(result_file):
        print(f"File \"{result_file}\" already exists!")
        continue
    intermediate_folder = f'../evaluating_responses/part_1b/{model_type}/{culture}_{size}/'
    if not os.path.exists(intermediate_folder):
        os.mkdir(intermediate_folder)

    if size < 10: continue

    pg = PromptGeneratorPt1(instance_file, num_instances=50)

    pg.get_prompts_list(include_aglo=True)
    prompts_list = pg.prompts_list

    # print(prompts_list)

    for p, prompt in enumerate(tqdm(prompts_list)):
        ip_tokens, op_tokens = 0, 0
        incomplete = None
        # print(prompt[1])
        llm = LLM(auth_file=auth_key_path, family=model_family, model=model)
        prompt_req, og_prompt = prompt[1], copy.deepcopy(prompt[1])
        smooth = False

        if os.path.exists(intermediate_folder+f'row_{p}'):
            with open(intermediate_folder+f'row_{p}', 'rb') as file:
                row = pkl.load(file)
            print(f'Already have row_{p}')
            data.append(row)
            continue
         
        for r in range(2):
            if smooth: continue
            print(f"ROUND {r+1}")

            # for t in range(3):
            #     try:
            response, usage = llm.makeLLMRequest(prompt_req)
                #     break
                # except:
                #     print(f"FAILED AT TRY {t+1}")
                #     continue
            
            if 'gemini' not in model_type: 
                ip_tokens += usage.prompt_tokens
                op_tokens += usage.completion_tokens
            else:
                ip_tokens += usage.prompt_token_count
                op_tokens += usage.candidates_token_count if usage.candidates_token_count else 0
            
            try:
                answer_ext = response[response.rfind('<answer>')+8:response.rfind('</answer>')]
                answer = answer_ext[answer_ext.index('{'):answer_ext.index('}')+1]
                # print(answer)
                answer = json.loads(answer)
                # print(answer)
            except:
                print("INCORRECT JSON FORMAT")
                # print(response[response.rfind('<answer>')+8:response.rfind('</answer>')])
                prompt_req = pg.correct_json_prompt(og_prompt, response, size)
                # print(prompt_req)
                continue


            man_opt_list = JSONMatchToList(prompt[2])
            llm_answer_list, verdict = JSONobjToList(answer, size)
            
            if verdict not in ['empty', 'okay']:
                print("FORMATTING ERROR!")
                print(answer_ext)
                print(answer)
                prompt_req = pg.correct_json_obj_prompt(og_prompt, response, verdict, size)
                # print(prompt_req)
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
                prompt_req = pg.incomplete_matching_prompt(og_prompt, response, counts, set([w+1 for w in range(size)]) - set(llm_answer_list),size)
                if invalid:
                    print('INVALID MATCHING!')
                    print(answer)
                    print(counts)
                else:
                    incomplete = answer
                    print(f"INCOMPLETE MATCHING: {set([w+1 for w in range(size)]) - set(llm_answer_list)} missing.")
                    print(answer)
                # print(prompt_req)
                continue

            bp = blockingPairs(size, prompt[4], prompt[5], np.array(llm_answer_list), "weak")
            bp_count = bp["blockingPairCount"]
            correct = 1 if bp_count == 0 else 0
            bp_list = f"{bp['blockingPairs']}"
            row = [culture, size, prompt[0], og_prompt, answer, correct, bp_count, bp_list, js, inter, response, ip_tokens, op_tokens, r+1, "Processed smoothly."]
            data.append(row)
            with open(intermediate_folder+f'row_{p}', 'wb') as file:
                pkl.dump(row, file)
            smooth = True

        # print(response)
        if smooth:
            print(f"$$$$$$$$$$$$$$$$$$ PROCESSED INSTANCE {prompt[0]} CORRECTLY (CORRECT = {correct}) $$$$$$$$$$$$$$$$$$$$$") 
            continue
        # print(prompt_req)
        for t in range(5):
            try: 
                response, usage = llm.makeLLMRequest(prompt_req)
                break
            except:
                continue
        if 'gemini' not in model_type: 
            ip_tokens += usage.prompt_tokens
            op_tokens += usage.completion_tokens
        else:
            ip_tokens += usage.prompt_token_count
            op_tokens += usage.candidates_token_count if usage.candidates_token_count else 0
        
        try:
            answer = response[response.rfind('<answer>')+8:response.rfind('</answer>')]
            answer = answer[answer.index('{'):answer.index('}')+1]
            # print(answer)
            answer = json.loads(answer)
            # print(answer)
        except:
            print("INCORRECT JSON FORMAT!")
            print(response[-2000:])
            if incomplete:
                data.append([culture, size, prompt[0], og_prompt, incomplete, 0, 0, '', 0, 0, response, ip_tokens, op_tokens, 3, "INCOMPLETE MATCHING!"])
            else:
                data.append([culture, size, prompt[0], og_prompt, -1, 0, 0, '', 0, 0, response, ip_tokens, op_tokens, 3, "INCORRECT JSON FORMAT!"])
            continue


        man_opt_list = JSONMatchToList(prompt[2])
        llm_answer_list, verdict = JSONobjToList(answer, size)
        
        if verdict not in ['empty', 'okay']:
            print("FORMATTING ERROR!")
            print(answer)
            print(verdict)
            if incomplete:
                data.append([culture, size, prompt[0], og_prompt, incomplete, 0, 0, '', 0, 0, response, ip_tokens, op_tokens, 3, "INCOMPLETE MATCHING!"])
            else:
                data.append([culture, size, prompt[0], og_prompt, -1, 0, 0, '', 0, 0, response, ip_tokens, op_tokens, 3, "FORMATTING ERROR!"])
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
                    data.append([culture, size, prompt[0], og_prompt, incomplete, 0, 0, '', 0, 0, response, ip_tokens, op_tokens, 3, "INCOMPLETE MATCHING!"])
                data.append([culture, size, prompt[0], og_prompt, answer, 0, 0, '', 0, 0, response, ip_tokens, op_tokens, 3, "INVALID MATCHING!"])
                print(f"INVALID MATCHING!")
                print(answer)
                print(counts)
            else:
                data.append([culture, size, prompt[0], og_prompt, answer, 0, 0, '', 0, 0, response, ip_tokens, op_tokens, 3, "INCOMPLETE MATCHING!"])
                print(f"INCOMPLETE MATCHING: {set([w+1 for w in range(size)]) - set(llm_answer_list)} missing.")
            continue

        try:
            bp = blockingPairs(size, prompt[4], prompt[5], np.array(llm_answer_list), "weak")
            bp_count = bp["blockingPairCount"]
            correct = 1 if bp_count == 0 else 0
            bp_list = f"{bp['blockingPairs']}"
        except:
            print("ERROR COMPUTING BLOCKING PAIRS!!")
            if incomplete:
                data.append([culture, size, prompt[0], prompt[6], og_prompt, incomplete, 0, 0, '', 0, 0, response, ip_tokens, op_tokens, 3, "INCOMPLETE MATCHING!"])
            else:
                data.append([culture, size, prompt[0], prompt[6], og_prompt, -1, 0, 0, '', 0, 0, response, ip_tokens, op_tokens, 3, "ERROR COMPUTING BLOCKING PAIRS!!"])
            continue

        row = [culture, size, prompt[0], og_prompt, answer, correct, bp_count, bp_list, js, inter, response, ip_tokens, op_tokens, 3, "Processed smoothly."]
        data.append(row)
        with open(intermediate_folder+f'row_{p}', 'wb') as file:
            pkl.dump(row, file)

        print(f"########################### INSTANCE {prompt[0]} #################################")

    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

    # shutil.rmtree(intermediate_folder)



        

    