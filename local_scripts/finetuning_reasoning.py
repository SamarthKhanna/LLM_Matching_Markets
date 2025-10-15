from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_data_formats
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
import torch
from copy import deepcopy
import pickle as pkl
from transformers import TextStreamer
from tqdm.notebook import tqdm
from datasets import load_dataset
import csv
import os


max_seq_length = 10000 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
culture = 'both'
min_size = 5
max_size = 50
num_instances = 9000

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/QwQ-32B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = False,
    load_in_8bit = False,
    full_finetuning = False,
    token = "", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are an intelligent assistant who is an expert in algorithms. You will be given an instance of the two-sided matching problem, and will be asked to answer a question about the preferences of the agents involved.  

### Question:
{}

### Response:
<think>{}"""

question = """First, here are the preference lists for all individuals:

<preferences>
{
M: {
M1: [W5,W3,W4,W2,W1],
M2: [W2,W3,W5,W1,W4],
M3: [W5,W3,W1,W4,W2],
M4: [W1,W3,W2,W5,W4],
M5: [W2,W3,W4,W1,W5],
},
W: {
W1: [M3,M5,M4,M1,M2],
W2: [M1,M3,M4,M5,M2],
W3: [M3,M2,M4,M1,M5],
W4: [M4,M2,M3,M5,M1],
W5: [M2,M4,M5,M1,M3],
}}
</preferences>

Now, you will be asked a specific question about agent preferences:

<question>
Who is agent W2's, 1-most preferred agent?
</question>

Once you have determined the answer, provide your output in the following format:

1. The solution as a single agent name. For example, ""W1""

Present your final answer within <answer> tags.

IMPORTANT: ONLY RETURN THE NAME OF THE SINGLE AGENT THAT IS THE ANSWER TO THE QUESTION. Do not include any explanations or additional information in your final answer."""


FastLanguageModel.for_inference(model) 
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=10000,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Response:")[1])

with open(f'data/stable-matching/train_data_{culture}_{min_size}_{max_size}_{num_instances}_pref_comp.pkl', 'rb') as file:
  train_data = pkl.load(file)

train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are an intelligent assistant who is an expert in algorithms. You will be given an instance of the two-sided matching problem, and will be asked to answer a question about the preferences of the agents involved.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }

dataset = train_data
dataset = dataset.map(formatting_prompts_func, batched = True,)
print(dataset["text"][0])

model = FastLanguageModel.get_peft_model(
    model,
    r=32,  
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,  
    bias="none",  
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  
    loftq_config=None,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs = 1,
        warmup_steps=5,
        # max_steps=120,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()

print('TOKENIZING')

question = """First, here are the preference lists for all individuals:
        
<preferences>
{
M: {
M1: [W6,W7,W10,W4,W2,W3,W9,W5,W1,W8],
M2: [W1,W3,W8,W2,W6,W10,W7,W4,W5,W9],
M3: [W4,W1,W7,W6,W2,W10,W3,W8,W9,W5],
M4: [W8,W6,W1,W7,W9,W5,W3,W2,W4,W10],
M5: [W9,W6,W2,W10,W5,W4,W8,W7,W3,W1],
M6: [W4,W1,W6,W9,W2,W3,W8,W10,W5,W7],
M7: [W1,W7,W4,W8,W5,W10,W3,W6,W9,W2],
M8: [W4,W7,W9,W3,W8,W2,W6,W5,W1,W10],
M9: [W3,W7,W5,W2,W9,W8,W4,W1,W10,W6],
M10: [W1,W8,W5,W6,W3,W2,W7,W9,W4,W10],
},
W: {
W1: [M7,M6,M9,M2,M1,M5,M3,M10,M8,M4],
W2: [M9,M4,M6,M1,M3,M7,M10,M8,M2,M5],
W3: [M10,M8,M5,M4,M6,M7,M3,M2,M1,M9],
W4: [M7,M3,M9,M6,M8,M2,M5,M1,M4,M10],
W5: [M6,M4,M9,M3,M5,M8,M7,M1,M2,M10],
W6: [M7,M8,M6,M3,M9,M10,M5,M2,M4,M1],
W7: [M6,M7,M5,M3,M10,M2,M4,M8,M9,M1],
W8: [M7,M4,M10,M5,M3,M9,M8,M2,M6,M1],
W9: [M5,M7,M1,M10,M9,M6,M3,M4,M8,M2],
W10: [M10,M4,M5,M9,M3,M8,M1,M6,M2,M7],
}}
</preferences>

Now, you will be asked a specific question about agent preferences:

<question>
Would agent W4, prefer M1 and M2 over M4?
</question>

Once you have determined the answer, provide your output in the following format:

1. The solution as a YES or a NO. For example, ""NO""

Present your final answer within <answer> tags.

IMPORTANT: ONLY RETURN YES OR NO THAT IS THE ANSWER TO THE QUESTION. Do not include any explanations or additional information in your final answer."""

FastLanguageModel.for_inference(model)  # Unsloth has 2x faster inference!
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

print("BEGINNING INFERENCE")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=10000,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Response:")[1])

print("SAVING MODEL")

new_model_local = "FT_models/Qwen-32B-Prefs-r32-full"
model.save_pretrained(new_model_local) 
tokenizer.save_pretrained(new_model_local)

model.save_pretrained_merged(new_model_local, tokenizer, save_method = "merged_16bit",)

new_model_online = "Samarth2511/Qwen-32B-Prefs-r32-full"
model.push_to_hub(new_model_online)
tokenizer.push_to_hub(new_model_online)

model.push_to_hub_merged(new_model_online, tokenizer, save_method = "merged_16bit")