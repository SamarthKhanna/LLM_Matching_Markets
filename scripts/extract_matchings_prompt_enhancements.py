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
    No_one = "None"

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

class WomenList20(Enum):
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
    No_one = "None"

class MenList20(Enum):
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
    M11 = "M11"
    M12 = "M12"
    M13 = "M13"
    M14 = "M14"
    M15 = "M15"
    M16 = "M16"
    M17 = "M17"
    M18 = "M18"
    M19 = "M19"
    M20 = "M20"

class Matching20(BaseModel):
    model_config = ConfigDict(extra="ignore")
    M1: WomenList20
    M2: WomenList20
    M3: WomenList20
    M4: WomenList20
    M5: WomenList20
    M6: WomenList20
    M7: WomenList20
    M8: WomenList20
    M9: WomenList20
    M10: WomenList20
    M11: WomenList20
    M12: WomenList20
    M13: WomenList20
    M14: WomenList20
    M15: WomenList20
    M16: WomenList20
    M17: WomenList20
    M18: WomenList20
    M19: WomenList20
    M20: WomenList20

class WomenList50(Enum):
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
    W50 = "W50"
    No_one = "None"

class MenList50(Enum):
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
    M11 = "M11"
    M12 = "M12"
    M13 = "M13"
    M14 = "M14"
    M15 = "M15"
    M16 = "M16"
    M17 = "M17"
    M18 = "M18"
    M19 = "M19"
    M20 = "M20"
    M21 = "M21"
    M22 = "M22"
    M23 = "M23"
    M24 = "M24"
    M25 = "M25"
    M26 = "M26"
    M27 = "M27"
    M28 = "M28"
    M29 = "M29"
    M30 = "M30"
    M31 = "M31"
    M32 = "M32"
    M33 = "M33"
    M34 = "M34"
    M35 = "M35"
    M36 = "M36"
    M37 = "M37"
    M38 = "M38"
    M39 = "M39"
    M40 = "M40"
    M41 = "M41"
    M42 = "M42"
    M43 = "M43"
    M44 = "M44"
    M45 = "M45"
    M46 = "M46"
    M47 = "M47"
    M48 = "M48"
    M49 = "M49"
    M50 = "M10"

class Matching50(BaseModel):
    model_config = ConfigDict(extra="ignore")
    M1: WomenList50
    M2: WomenList50
    M3: WomenList50
    M4: WomenList50
    M5: WomenList50
    M6: WomenList50
    M7: WomenList50
    M8: WomenList50
    M9: WomenList50
    M10: WomenList50
    M11: WomenList50
    M12: WomenList50
    M13: WomenList50
    M14: WomenList50
    M15: WomenList50
    M16: WomenList50
    M17: WomenList50
    M18: WomenList50
    M19: WomenList50
    M20: WomenList50
    M21: WomenList50
    M22: WomenList50
    M23: WomenList50
    M24: WomenList50
    M25: WomenList50
    M26: WomenList50
    M27: WomenList50
    M28: WomenList50
    M29: WomenList50
    M30: WomenList50
    M31: WomenList50
    M32: WomenList50
    M33: WomenList50
    M34: WomenList50
    M35: WomenList50
    M36: WomenList50
    M37: WomenList50
    M38: WomenList50
    M39: WomenList50
    M40: WomenList50
    M41: WomenList50
    M42: WomenList50
    M43: WomenList50
    M44: WomenList50
    M45: WomenList50
    M46: WomenList50
    M47: WomenList50
    M48: WomenList50
    M49: WomenList50
    M50: WomenList50


def generate_prompt(n, previous_response):
    json_format = "{\n"
    for m in range(1, n):
        json_format += f"\t\"M{m}\": \"<woman matched with M{m}>\",\n"
    json_format += f"\t\"M{m+1}\": \"<woman matched with M{m+1}>\"\n"
    json_format += "}"

    prompt = f"""Previously, I had given an LLM and instance of the two-sided matching problem, where {n} men were to be
matched with {n} women, and asked it to return a solution in the following format:

{json_format}
    
However, the LLM failed to give me a valid solution in the above-mentioned format. Given below are the final few lines of it's response:
----------------------------------------------------------------
{previous_response}
----------------------------------------------------------------
Please parse this response into the JSON format mentioned above. If no matching is provided, or if a man is matched with no woman, return \"None\" for that man. If the response contains only the algorithm or steps needed to compute the matching, do not attempt to run the algorithm and derive the matching yourself, and return \"None\" for every man. 
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

schema_classes = {
    10: {
        'womanlist': WomenList10,
        'manlist': MenList10,
        'matching': Matching10
    },
    20: {
        'womanlist': WomenList20,
        'manlist': MenList20,
        'matching': Matching20
    },
    50: {
        'womanlist': WomenList50,
        'manlist': MenList50,
        'matching': Matching50
    },
}

client = genai.Client(api_key="API_KEY")
result_dir = '../evaluating_responses/part_4/'
models = [
    # 'gemini20',
    # 'gemini25',
    # 'llama33',
    # 'qwen_qwq',
    # "deepseek_dist",
    # "deepseek",
    "o3-mini"
]
for model in models:
    print(F'STARTING {model}')
    corrected = [['Culture', 'Size', 'Instance', 'Type', 'Corrected', 'Correctness','Blocking_Pair_Count','Blocking_Pair_List','Jaccard_Similarity','Intersection', 'Remarks']]
    result_files = os.listdir(result_dir)
    model_files = [filename for filename in result_files if model in filename and 'corrected' not in filename and 'csv' in filename and 'dist' not in filename]
    for model_file in model_files:
        data = pd.read_csv(result_dir+model_file)
        # data = data[data['Remarks'] != "Processed smoothly."]

        for i, row in tqdm(enumerate(data.values)):

            culture = row[0]
            size = row[1]
            instance = row[2]
            ptype = row[3] 
            answer = row[5] if 'gemini' in model or 'o3-mini' in model or ('deepseek' in model and 'dist' not in model) else row [8]
            correct = row[6] if 'gemini' in model or 'o3-mini' in model or ('deepseek' in model and 'dist' not in model) else row [9]
            bp_count = row[7] if 'gemini' in model or 'o3-mini' in model or ('deepseek' in model and 'dist' not in model) else row [10]
            bp_list = row[8] if 'gemini' in model or 'o3-mini' in model or ('deepseek' in model and 'dist' not in model) else row [11]
            js = row[9] if 'gemini' in model or 'o3-mini' in model or ('deepseek' in model and 'dist' not in model) else row [12]
            inter = row[10] if 'gemini' in model or 'o3-mini' in model or ('deepseek' in model and 'dist' not in model) else row [13]
            remarks = row[-1] 
            old_response = row[-5] if 'gemini' in model or 'o3-mini' in model or ('deepseek' in model and 'dist' not in model) else row [-3]
            instances_data = pd.read_csv(f'../instances_matchings/{size}_{culture}_processed.csv').values
            man_opt_list = instances_data[instance][6]
            man_opt_list = JSONMatchToList(man_opt_list)
            men_prefs_string = instances_data[instance][4]
            women_prefs_string = instances_data[instance][5]

            
            # print(row[2])
            if remarks in ["Processed smoothly.", "INVALID MATCHING!"]:
                # print(row[-1])
                if 'gemini' not in model and remarks == "Processed smoothly.":
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
                corrected.append([culture, size, instance, ptype, answer, correct, bp_count, bp_list, js, inter, remarks])
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
                        # print(f'INCOMPLETE: {obj}---{llm_answer_list}')
                        corrected.append([culture, size, instance, ptype, answer, correct, bp_count, bp_list, js, inter, remarks])
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
                        corrected.append([culture, size, instance, ptype, obj, 0, 0, 0, 0, 0, 'INVALID MATCHING!'])
                else:
                    # print(f'INCOMPLETE: {obj}---{llm_answer_list}')
                    corrected.append([culture, size, instance, ptype, obj, 0, 0, 0, 0, 0, 'INCOMPLETE MATCHING!'])
                continue
            elif len(set(obj.values())) == 1 and 0 in llm_answer_list:
                # print(old_response[-1000:])
                # print(f'EMPTY MATCHING: {obj}---{llm_answer_list}')
                corrected.append([culture, size, instance, ptype, obj, 0, 0, 0, 0, 0, 'EMPTY/NO MATCHING!'])   
                continue     

            try:
                bp = blockingPairs(size, men_prefs_string, women_prefs_string, np.array(llm_answer_list), "weak")
                bp_count = bp["blockingPairCount"]
                correct = 1 if bp_count == 0 else 0
                bp_list = f"{bp['blockingPairs']}"
                print(f'SMOOTH: {obj}---{llm_answer_list}')
                corrected.append([culture, size, instance, ptype, obj, correct, bp_count, bp_list, js, inter, 'Processed smoothly.'])
                print("SUCCESSFUL RETRIEVAL")
            except:
                print(llm_answer_list)
                print('BP ERROR')
                break

    with open(result_dir + f"{model}_corrected.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(corrected) 
