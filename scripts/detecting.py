from components import LLM, PromptGeneratorPt2
import os
import csv
from tqdm import tqdm
import math
import pickle as pkl
import shutil

specifications = {
    'llama33': ('../Auth_keys/llama_auth_key.txt', 'llama', 'llama-3.3-70b-versatile'),
    'llama3': ('../Auth_keys/llama_auth_key.txt', 'llama', 'llama3-70b-8192'),
    'deepseek_dist': ('../Auth_keys/llama_auth_key.txt', 'llama', 'deepseek-r1-distill-llama-70b'),
    'qwen': ('../Auth_keys/llama_auth_key.txt', 'llama', 'qwen-qwq-32b'),
    'gemini': ('../Auth_keys/gemini_auth_key_2.txt', 'gemini', 'gemini-1.5-pro'),
    'gemini20': ('../Auth_keys/gemini_auth_key_2.txt', 'gemini', 'gemini-2.0-flash'),
    'gemini25': ('../Auth_keys/gemini_auth_key_2.txt', 'gemini', 'gemini-2.5-pro-preview-03-25'),
    'o3-mini': ('../Auth_keys/openai_auth_key_fairlab.txt', 'chatgpt', 'o3-mini'),
    'deepseek': ('../Auth_keys/deepseek_auth_key.txt', 'deepseek', 'deepseek-reasoner')
}

model_type = 'deepseek'
auth_key_path, model_family, model = specifications[model_type]

llm = LLM(auth_file=auth_key_path, family=model_family, model=model)

model_folder = f"../evaluating_responses/part_2/{model_type}/"
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

instance_files = os.listdir('../instances_matchings/')
for instance_file in instance_files:
    print("GENERATING RESPONSES FOR ", instance_file)
    if '.csv' not in instance_file: 
        continue
    data = [['Culture', 'Size', 'Instance', 'Prompt', 'Type', 'Answer', 'Correctness', 'Response', 'Input_tokens', 'Output_Tokens']]
    culture = instance_file.split('_')[1]
    size = int(instance_file.split('_')[0])
    result_file = f'../evaluating_responses/part_2/{model_type}_{culture}_{size}.csv'
    if os.path.exists(result_file):
        print(f"File \"{result_file}\" already exists!")
        continue
    intermediate_folder = f'../evaluating_responses/part_2/{model_type}/{culture}_{size}/'
    if not os.path.exists(intermediate_folder):
        os.mkdir(intermediate_folder)
    if size < 10: continue
    if culture == "womanmaster":
        pg = PromptGeneratorPt2(instance_file, types = ['men_opt', 'random_1', 
                                                        # f'random_{round(math.sqrt(size))}', f'random_{size}', 
                                                        'random'])
    else:
        pg = PromptGeneratorPt2(instance_file, types = ['men_opt', 'women_opt', 
                                                        # 'lattice', 
                                                        'random_1', 
                                                        # f'random_{round(math.sqrt(size))}', f'random_{size}', 
                                                        'random'])

    pg.get_prompts_list()
    prompts_list = pg.prompts_list

    for r, prompt in enumerate(tqdm(prompts_list)):
        # print(prompt)
        if os.path.exists(intermediate_folder+f'row_{r}'):
            with open(intermediate_folder+f'row_{r}', 'rb') as file:
                row = pkl.load(file)
            print(f'Already have row_{r}')
            data.append(row)
            continue
        llm = LLM(auth_file=auth_key_path, family=model_family, model=model)
        done = False
        for t in range(3):
            try:
                response, usage = llm.makeLLMRequest(prompt[1])
                if not response:
                    print(f"FAILED at TRY {t+1}, returned {response}")
                    continue
            except: 
                print(f"FAILED at TRY {t+1}")
                continue
            if 'gemini' not in model_type: 
                ip_tokens, op_tokens = usage.prompt_tokens, usage.completion_tokens
            else:
                ip_tokens = usage.prompt_token_count
                op_tokens = usage.candidates_token_count if usage.candidates_token_count else 0
            correct = 0
            ans_extract = response[response.rfind('<answer>'):response.rfind('</answer>')]
            if 'Yes' in ans_extract:
                answer = 1
            elif 'No' in ans_extract:
                answer = 0
            else:
                answer = -1
            if prompt[2] in ['men_opt', 'women_opt', 'lattice'] and answer:
                correct = 1
            elif 'random' in prompt[2] and answer == 0:
                correct = 1
            # except:
            #     print(f"FAILED AT TRY {t+1}")
            #     continue

            # print(prompt[2], answer)
            row = [culture, size, prompt[0], prompt[1], prompt[2], answer, correct, response, ip_tokens, op_tokens]
            data.append(row)
            # print(data)
            with open(intermediate_folder+f'row_{r}', 'wb') as file:
                pkl.dump(row, file)
            done = True
            break
        if not done:
            data.append(['ERROR']*10)

    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

    # shutil.rmtree(intermediate_folder)


        

    