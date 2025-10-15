from multi_agent_components import LLM, MultiAgentRoundRobinPt5
import os
import csv
from tqdm import tqdm
import math
from BlockingPairs import blockingPairs
import json
import numpy as np
import asyncio
from collections import defaultdict

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

specifications = {
    'llama33': ('../Auth_keys/llama_auth_key.txt', 'llama', 'llama-3.3-70b-versatile'),
    'llama3': ('../Auth_keys/llama_auth_key.txt', 'llama', 'llama3-70b-8192'),
    'deepseek_dist': ('../Auth_keys/llama_auth_key.txt', 'llama', 'deepseek-r1-distill-llama-70b'),
    'qwen': ('../Auth_keys/llama_auth_key.txt', 'llama', 'qwen-qwq-32b'),
    'gemini': ('../Auth_keys/gemini_auth_key_2.txt', 'gemini', 'gemini-1.5-pro'),
    'gemini20': ('../Auth_keys/gemini_auth_key_2.txt', 'gemini', 'gemini-2.0-flash'),
    'o3-mini': ('../Auth_keys/openai_auth_key_fairlab.txt', 'chatgpt', 'o3-mini')
}

model_type = 'gemini20'
auth_key_path, model_family, model = specifications[model_type]

llm = LLM(auth_file=auth_key_path, family=model_family, model=model)
model_client = llm.autogen_client

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

    return jaccard_sim


async def run_main(team):
    prompt_tokens, completion_tokens = 0, 0
    await team.reset()  # Reset the team for a new task.
    chat_history = ""
    # prompt = """
    # Let’s start the Gale-Shapley algorithm with the given preferences. Please proceed step-by-step.
    # """
    async for message in team.run_stream(task="Let’s start the Gale-Shapley algorithm with the given preferences. Please proceed step-by-step."):  # type: ignore
        if isinstance(message, TaskResult):
            continue
            print("Stop Reason:", message.stop_reason)

        else:
            chat_history += f"{message.content}\n\n================================================="
            # print(message.models_usage)
            if message.models_usage != None:
                prompt_tokens += message.models_usage.prompt_tokens
                completion_tokens += message.models_usage.completion_tokens
            # print(prompt_tokens, completion_tokens)
            # if hasattr(message, '_raw_response') and 'usage' in message._raw_response:
            #     print("Completion Tokens:", message._raw_response['usage'].get('completion_tokens'))
            # elif hasattr(message, 'metadata') and 'token_usage' in message.metadata: # Another possibility
            #     print("Completion Tokens:", message.metadata['token_usage'].get('completion_tokens'))
    return chat_history, prompt_tokens, completion_tokens



instance_files = os.listdir('../instances_matchings/')
for instance_file in instance_files:
    print("GENERATING RESPONSES FOR ", instance_file)
    data = [['Culture', 'Size', 'Instance', 'Prompt', 'Answer', 'Correctness','Blocking_Pair_Count', 'Blocking_Pair_List', 'Jaccard_Similarity', 'Intersection', 'Response', 'Input_Tokens', 'Output_Tokens', 'Remarks']]
    if '.csv' not in instance_file: 
        continue
    culture = instance_file.split('_')[1]
    size = int(instance_file.split('_')[0])
    result_file = f'../evaluating_responses/part_5/{model_type}_{culture}_{size}_pt5_test.csv'
    # if os.path.exists(result_file):
    #     print(f"File \"{result_file}\" already exists!")
    #     continue

    if size in [50,20,5]: continue


    ma_rr = MultiAgentRoundRobinPt5(instance_file, model_client)

    ma_rr.get_rrgc_list()
    rrgc_list = ma_rr.rrgc_list

    for rrgc in tqdm(rrgc_list):
        response, ip_tokens, op_tokens = asyncio.run(run_main(rrgc[2]))

        # print(response)
        
        try:
            answer = response[response.rfind('<answer>')+8:response.rfind('</answer>')]
            answer = answer[answer.index('{'):answer.index('}')+1]
            # print(answer)
            answer = json.loads(answer)
            # print(answer)
        except:
            print("INCORRECT JSON FORMAT!")
            print(response[-2000:])
            data.append([culture, size, rrgc[0], rrgc[1], -1, 0, 0, '', 0, 0, response, ip_tokens, op_tokens, "INCORRECT JSON FORMAT!"])
            continue


        man_opt_list = JSONMatchToList(rrgc[6])
        llm_answer_list, verdict = JSONobjToList(answer, size)
        
        if verdict not in ['empty', 'okay']:
            print("FORMATTING ERROR!")
            print(answer)
            print(verdict)
            data.append([culture, size, rrgc[0], rrgc[1], -1, 0, 0, '', 0, 0, response, ip_tokens, op_tokens, "FORMATTING ERROR!"])
            continue

        n = int(rrgc[7])
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
                data.append([culture, size, rrgc[0], rrgc[1], answer, 0, 0, '', 0, 0, response, ip_tokens, op_tokens, "INVALID MATCHING!"])
                print(f"INVALID MATCHING!")
                print(answer)
                print(counts)
            else:
                data.append([culture, size,  rrgc[0], rrgc[1], answer, 0, 0, '', 0, 0, response, ip_tokens, op_tokens, "INCOMPLETE MATCHING!"])
                print(f"INCOMPLETE MATCHING: {set([w+1 for w in range(size)]) - set(llm_answer_list)} missing.")
            continue

        correct = 0
        try:
            bp = blockingPairs(size, rrgc[4], rrgc[5], np.array(llm_answer_list), "weak")
            bp_count = bp["blockingPairCount"]
            correct = 1 if bp_count == 0 else 0
            bp_list = f"{bp['blockingPairs']}"
        except:
            print("ERROR COMPUTING BLOCKING PAIRS!!")
            data.append([culture, size, rrgc[0], rrgc[1], -1, 0, 0, '', 0, 0, response, ip_tokens, op_tokens, "ERROR COMPUTING BLOCKING PAIRS!!"])
            continue


        data.append([culture, size, rrgc[0], rrgc[1], answer, correct, bp_count, bp_list, js, response, ip_tokens, op_tokens, "Processed smoothly."])
        print(f'########################### PROCESSED INSTANCE {rrgc[0]} CORRECTLY (CORRECT = {correct}) ############################')
        # print(response)

    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)



        

    