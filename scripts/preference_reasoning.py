from components import LLM, PromptGeneratorPt3
import os
import csv
from tqdm import tqdm
import math
from BlockingPairs import blockingPairs
import pickle as pkl
import shutil

specifications = {
    'llama33': ('../Auth_keys/llama_auth_key.txt', 'llama', 'llama-3.3-70b-versatile'),
    'llama3': ('../Auth_keys/llama_auth_key.txt', 'llama', 'llama3-70b-8192'),
    'deepseek_dist': ('../Auth_keys/llama_auth_key.txt', 'llama', 'deepseek-r1-distill-llama-70b'),
    'qwen': ('../Auth_keys/llama_auth_key.txt', 'llama', 'qwen-qwq-32b'),
    'gemini': ('../Auth_keys/gemini_auth_key_2.txt', 'gemini', 'gemini-1.5-pro'),
    'gemini20': ('../Auth_keys/gemini_auth_key_2.txt', 'gemini', 'gemini-2.0-flash'),
    'gemini25': ('../Auth_keys/gemini_auth_key_3.txt', 'gemini', 'gemini-2.5-pro-preview-03-25'),
    'o3-mini': ('../Auth_keys/openai_auth_key_fairlab.txt', 'chatgpt', 'o3-mini'),
    'deepseek': ('../Auth_keys/deepseek_auth_key.txt', 'deepseek', 'deepseek-reasoner')
}

model_type = 'deepseek'
auth_key_path, model_family, model = specifications[model_type]

llm = LLM(auth_file=auth_key_path, family=model_family, model=model)

model_folder = f"../evaluating_responses/part_3/{model_type}/"
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

instance_files = os.listdir('../instances_matchings/')
for instance_file in instance_files:
    print("GENERATING RESPONSES FOR ", instance_file)
    data = [['Culture', 'Size', 'Instance', 'Question', 'Question_Type', 'Prompt', 'Answer', 'Correctness', 'Response', 'Input_Tokens', 'Output_Tokens', 'Remarks']]
    if '.csv' not in instance_file: 
        continue
    culture = instance_file.split('_')[1]
    size = int(instance_file.split('_')[0])
    result_file = f'../evaluating_responses/part_3/{model_type}_{culture}_{size}_pt3.csv'
    if os.path.exists(result_file):
        print(f"File \"{result_file}\" already exists!")
        continue
    intermediate_folder = f'../evaluating_responses/part_3/{model_type}/{culture}_{size}/'
    if not os.path.exists(intermediate_folder):
        os.mkdir(intermediate_folder)

    # if size not in [10]: continue

    pg = PromptGeneratorPt3(instance_file, num_instances=50)

    pg.get_prompts_list()
    prompts_list = pg.prompts_list

    # print(prompts_list)

    for r, prompt in enumerate(tqdm(prompts_list)):
        ip_tokens, op_tokens = 0, 0
        # print(prompt[1])
        if os.path.exists(intermediate_folder+f'row_{r}'):
            with open(intermediate_folder+f'row_{r}', 'rb') as file:
                row = pkl.load(file)
            print(f'Already have row_{r}')
            data.append(row)
            continue
        for t in range(3):
            try:
                response, usage = llm.makeLLMRequest(prompt[1])
                if not response: continue
                break
            except:
                print(f"FAILED AT TRY {t+1}")
                continue
        # print(response)
        if 'gemini' not in model_type: 
            ip_tokens, op_tokens = usage.prompt_tokens, usage.completion_tokens
        else:
            ip_tokens += usage.prompt_token_count
            op_tokens += usage.candidates_token_count if usage.candidates_token_count else 0
        
        if '<answer>' in response and '</answer>' in response:
            answer = response.split('<answer>')[1].split('</answer>')[0].lower().strip()
            correct = -1
            if prompt[3].lower().strip() in answer:
                correct = 1
            else:
                correct = 0
        else:
            correct = int(prompt[3].lower().strip() in response[-100:].lower().strip())
            
        row = [culture, size, prompt[0], prompt[4], prompt[2], prompt[1], prompt[3], correct, response, ip_tokens, op_tokens, "Processed smoothly."]
        data.append(row)
        with open(intermediate_folder+f'row_{r}', 'wb') as file:
            pkl.dump(row, file)

    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

    shutil.rmtree(intermediate_folder)



        

    