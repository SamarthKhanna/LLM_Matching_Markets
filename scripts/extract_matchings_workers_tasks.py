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

class TaskList10(Enum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"
    T6 = "T6"
    T7 = "T7"
    T8 = "T8"
    T9 = "T9"
    T10 = "T10"
    No_one = "None"

class WorkerList10(Enum):
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

class Matching10(BaseModel):
    model_config = ConfigDict(extra="ignore")
    W1: TaskList10
    W2: TaskList10
    W3: TaskList10
    W4: TaskList10
    W5: TaskList10
    W6: TaskList10
    W7: TaskList10
    W8: TaskList10
    W9: TaskList10
    W10: TaskList10

class TaskList20(Enum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"
    T6 = "T6"
    T7 = "T7"
    T8 = "T8"
    T9 = "T9"
    T10 = "T10"
    T11 = "T11"
    T12 = "T12"
    T13 = "T13"
    T14 = "T14"
    T15 = "T15"
    T16 = "T16"
    T17 = "T17"
    T18 = "T18"
    T19 = "T19"
    T20 = "T20"
    No_one = "None"

class WorkerList20(Enum):
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
    W11 = "W11"
    W12 = "W12"
    W13 = "W13"
    W14 = "W14"
    W15 = "W15"
    W16 = "W16"
    W17 = "W17"
    W18 = "W18"
    W19 = "W19"
    W20 = "W20"

class Matching20(BaseModel):
    model_config = ConfigDict(extra="ignore")
    W1: TaskList20
    W2: TaskList20
    W3: TaskList20
    W4: TaskList20
    W5: TaskList20
    W6: TaskList20
    W7: TaskList20
    W8: TaskList20
    W9: TaskList20
    W10: TaskList20
    W11: TaskList20
    W12: TaskList20
    W13: TaskList20
    W14: TaskList20
    W15: TaskList20
    W16: TaskList20
    W17: TaskList20
    W18: TaskList20
    W19: TaskList20
    W20: TaskList20

class TaskList50(Enum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"
    T6 = "T6"
    T7 = "T7"
    T8 = "T8"
    T9 = "T9"
    T10 = "T10"
    T11 = "T11"
    T12 = "T12"
    T13 = "T13"
    T14 = "T14"
    T15 = "T15"
    T16 = "T16"
    T17 = "T17"
    T18 = "T18"
    T19 = "T19"
    T20 = "T20"
    T21 = "T21"
    T22 = "T22"
    T23 = "T23"
    T24 = "T24"
    T25 = "T25"
    T26 = "T26"
    T27 = "T27"
    T28 = "T28"
    T29 = "T29"
    T30 = "T30"
    T31 = "T31"
    T32 = "T32"
    T33 = "T33"
    T34 = "T34"
    T35 = "T35"
    T36 = "T36"
    T37 = "T37"
    T38 = "T38"
    T39 = "T39"
    T40 = "T40"
    T41 = "T41"
    T42 = "T42"
    T43 = "T43"
    T44 = "T44"
    T45 = "T45"
    T46 = "T46"
    T47 = "T47"
    T48 = "T48"
    T49 = "T49"
    T50 = "T50"
    No_one = "None"

class WorkerList50(Enum):
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
    W11 = "W11"
    W12 = "W12"
    W13 = "W13"
    W14 = "W14"
    W15 = "W15"
    W16 = "W16"
    W17 = "W17"
    W18 = "W18"
    W19 = "W19"
    W20 = "W20"
    W21 = "W21"
    W22 = "W22"
    W23 = "W23"
    W24 = "W24"
    W25 = "W25"
    W26 = "W26"
    W27 = "W27"
    W28 = "W28"
    W29 = "W29"
    W30 = "W30"
    W31 = "W31"
    W32 = "W32"
    W33 = "W33"
    W34 = "W34"
    W35 = "W35"
    W36 = "W36"
    W37 = "W37"
    W38 = "W38"
    W39 = "W39"
    W40 = "W40"
    W41 = "W41"
    W42 = "W42"
    W43 = "W43"
    W44 = "W44"
    W45 = "W45"
    W46 = "W46"
    W47 = "W47"
    W48 = "W48"
    W49 = "W49"
    W50 = "W10"

class Matching50(BaseModel):
    model_config = ConfigDict(extra="ignore")
    W1: TaskList50
    W2: TaskList50
    W3: TaskList50
    W4: TaskList50
    W5: TaskList50
    W6: TaskList50
    W7: TaskList50
    W8: TaskList50
    W9: TaskList50
    W10: TaskList50
    W11: TaskList50
    W12: TaskList50
    W13: TaskList50
    W14: TaskList50
    W15: TaskList50
    W16: TaskList50
    W17: TaskList50
    W18: TaskList50
    W19: TaskList50
    W20: TaskList50
    W21: TaskList50
    W22: TaskList50
    W23: TaskList50
    W24: TaskList50
    W25: TaskList50
    W26: TaskList50
    W27: TaskList50
    W28: TaskList50
    W29: TaskList50
    W30: TaskList50
    W31: TaskList50
    W32: TaskList50
    W33: TaskList50
    W34: TaskList50
    W35: TaskList50
    W36: TaskList50
    W37: TaskList50
    W38: TaskList50
    W39: TaskList50
    W40: TaskList50
    W41: TaskList50
    W42: TaskList50
    W43: TaskList50
    W44: TaskList50
    W45: TaskList50
    W46: TaskList50
    W47: TaskList50
    W48: TaskList50
    W49: TaskList50
    W50: TaskList50


def generate_prompt(n, previous_response):
    json_format = "{\n"
    for m in range(1, n):
        json_format += f"\t\"W{m}\": \"<task assigned to with W{m}>\",\n"
    json_format += f"\t\"W{m+1}\": \"<task assigned to W{m+1}>\"\n"
    json_format += "}"

    prompt = f"""Previously, I had given an LLM and instance of the two-sided matching problem, where {n} workers were to be
assigned {n} tasks, and asked it to return a solution in the following format:

{json_format}
    
However, the LLM failed to give me a valid solution in the above-mentioned format. Given below are the final few lines of it's response:
----------------------------------------------------------------
{previous_response}
----------------------------------------------------------------
Please parse this response into the JSON format mentioned above. If no matching is provided, or if a worker is assigned no task, return \"None\" for that worker. If the response contains only the algorithm or steps needed to compute the matching, do not attempt to run the algorithm and derive the matching yourself, and return \"None\" for every worker. 
"""
    return prompt

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
        mstring = f"W{m}"
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

schema_classes = {
    10: {
        'womanlist': TaskList10,
        'manlist': WorkerList10,
        'matching': Matching10
    },
    20: {
        'womanlist': TaskList20,
        'manlist': WorkerList20,
        'matching': Matching20
    },
    50: {
        'womanlist': TaskList50,
        'manlist': WorkerList50,
        'matching': Matching50
    },
}

client = genai.Client(api_key="API_KEY")
result_dir = '../evaluating_responses/part_1c/'
models = [
    # 'gemini20',
    # 'gemini25',
    # 'llama33',
    # 'qwen_qwq',
    # 'deepseek_dist',
    'o3-mini',
    # "deepseek",
    # "ds_llama_8b",
    # "ds_llama_ft_2",
    # "ds_llama_ft_mid",
]
for model in models:
    corrected = [['Culture', 'Size', 'Instance', 'Corrected', 'Correctness','Blocking_Pair_Count','Blocking_Pair_List','Jaccard_Similarity','Intersection', 'Remarks']]
    result_files = os.listdir(result_dir)
    model_files = [filename for filename in result_files if model in filename and 'corrected' not in filename and 'csv' in filename and '_' in filename and 'dist' not in filename]
    for model_file in model_files:
        data = pd.read_csv(result_dir+model_file)
        # data = data[data['Remarks'] != "Processed smoothly."]
        print(model_file)

        for i, row in tqdm(enumerate(data.values)):

            culture = row[0]
            size = row[1]
            instance = row[2]
            answer = row[4] if 'gemini' in model or 'o3-mini' in model or ('deepseek' in model and 'dist' not in model) else row [7]
            correct = row[5] if 'gemini' in model or 'o3-mini' in model or ('deepseek' in model and 'dist' not in model)  else row [8]
            bp_count = row[6] if 'gemini' in model or 'o3-mini' in model or ('deepseek' in model and 'dist' not in model)  else row [9]
            bp_list = row[7] if 'gemini' in model or 'o3-mini' in model or ('deepseek' in model and 'dist' not in model)  else row [10]
            js = row[8] if 'gemini' in model or 'o3-mini' in model or ('deepseek' in model and 'dist' not in model)  else row [11]
            inter = row[9] if 'gemini' in model or 'o3-mini' in model else row [12]
            remarks = row[-1] 
            old_response = row[-5] if 'gemini' in model or 'o3-mini' in model or ('deepseek' in model and 'dist' not in model)  else row [-3]
            instances_data = pd.read_csv(f'../instances_matchings/{size}_{culture}_processed.csv').values
            man_opt_list = instances_data[instance][6]
            man_opt_list = JSONMatchToList(man_opt_list)
            men_prefs_string = instances_data[instance][4]
            women_prefs_string = instances_data[instance][5]

            
            # print(row[2])
            if remarks in ["Processed smoothly.", "INVALID MATCHING!"]:
                # print(row[-1])
                if 'gemini' not in model and 'o3' not in model and remarks == "Processed smoothly.":
                    # try:
                    # print("HERE", model_file, answer)
                    llm_answer_list, verdict = JSONobjToList(json.loads(answer.replace('\'', '\"')), size)
                    # except:
                    #     print("JSON LOAD FAILED")

                    js, inter = jaccard_similarity(man_opt_list, llm_answer_list)
                    # print(culture, size, instance, js, inter, man_opt_list, llm_answer_list)
                    bp = blockingPairs(size, men_prefs_string, women_prefs_string, np.array(llm_answer_list), "weak")
                    bp_count = bp["blockingPairCount"]
                    correct = 1 if bp_count == 0 else 0
                    bp_list = f"{bp['blockingPairs']}"
                corrected.append([culture, size, instance, answer, correct, bp_count, bp_list, js, inter, remarks])
                continue
            prompt = generate_prompt(size, old_response[-5000:])
            # print(row[-3][-5000:])
            # response = llm.makeLLMRequestFormatting(queryText=prompt, json_schema=schema_classes[row[1]]['matching'].model_json_schema())
            # print(response)
            if size == 5: continue
            schema = schema_classes[size]['matching'].model_json_schema()
            # print(schema)
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': schema,
                },
            )
            obj = json.loads(response.text)
            llm_answer_list, verdict = JSONobjToList(obj, row[1])
            # print(llm_answer_list)

            try:
                js, inter = jaccard_similarity(man_opt_list, llm_answer_list)
            except:
                js, inter = -1, 0

            if (len(set(obj.values())) < size or 0 in llm_answer_list) and (len(set(obj.values())) > 1):
                counts = {}
                invalid = False
                for man in obj:
                    counts[obj[man]] = counts.get(obj[man], 0) + 1
                    if counts[obj[man]] > 1:
                        invalid = True
                # print(row[-3][-2000:])
                # print(response.text)
                # print(counts)
                # time.sleep(60)
                if invalid:
                    if remarks == 'INCOMPLETE MATCHING!':
                        print(f'INCOMPLETE: {obj}---{llm_answer_list}')
                        corrected.append([culture, size, instance, answer, correct, bp_count, bp_list, js, inter, remarks])
                    else:
                        # answer_json = json.loads(answer.replace('\'', '\"'))
                        # if answer_json != obj: 
                        #     previous_list, prevv = JSONobjToList(answer_json, size)
                        #     # for p in range(size):
                        #     print(old_response[-1000:])
                        #     print(answer_json)
                        #     print(previous_list)
                        #     print(llm_answer_list)
                        #     print(obj)
                        #     # print(f'INVALID: {obj}---{llm_answer_list}')
                        corrected.append([culture, size, instance, obj, 0, 0, 0, 0, 0, 'INVALID MATCHING!'])
                else:
                    print(f'INCOMPLETE: {obj}---{llm_answer_list}')
                    corrected.append([culture, size, instance, obj, 0, 0, 0, 0, 0, 'INCOMPLETE MATCHING!'])
                continue
            elif len(set(obj.values())) == 1 and 0 in llm_answer_list:
                print(old_response[-1000:])
                print(f'EMPTY MATCHING: {obj}---{llm_answer_list}')
                corrected.append([culture, size, instance, obj, 0, 0, 0, 0, 0, 'EMPTY/NO MATCHING!'])   
                continue     

            try:
                bp = blockingPairs(size, men_prefs_string, women_prefs_string, np.array(llm_answer_list), "weak")
                bp_count = bp["blockingPairCount"]
                correct = 1 if bp_count == 0 else 0
                bp_list = f"{bp['blockingPairs']}"
                print(f'SMOOTH: {obj}---{llm_answer_list}')
                corrected.append([culture, size, instance, obj, correct, bp_count, bp_list, js, inter, 'Processed smoothly.'])
                print("SUCCESSFUL RETRIEVAL")
            except:
                print(llm_answer_list)
                break

    with open(result_dir + f"{model}_corrected.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(corrected) 
