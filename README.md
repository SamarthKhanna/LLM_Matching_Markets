# LLM_Matching_Markets
Benchmarking algorithmic reasoning with ranked preferences

This repository provides the code and data used our NeurIPS 2025 paper ["Matching Markets Meet LLMs: Algorithmic Reasoning with Ranked Preferences"](https://neurips.cc/virtual/2025/poster/120182). Below we how this code can be used to perform each task discussed in the paper, as well as the fine-tuning pipeline. 

### Generating Evaluation Data

1. Run "create_instances.py" with the correct specification of the preference type (a.k.a "culture")
2. Run "generate_instance_data.py"

Note: The actual data used for evaluation is already present in the "instances_matchings" folder.

### Evaluation

1. Run "<task_name>.py" , where <task_name> is the name of the task for which a given model needs to be evaluated (e.g., "generating.py"). Remember to specify the correct model in the script before running it.
2. Run "extract_<task_specific_extension>.py" to obtain the cleaned data for a given model (specified through the model string in the script). The <task_specific_extension> for different tasks is as follows:
- Generation: matchings_generating
- Generation (Worker-Task setting): matchings_workers_tasks
- Detection: answers_detecting
- Resolving: matchings_resolving
- Prompt Enhancements: matchings_prompt_enhancements 


Note: The scripts for the API-based models can be found in the "scripts/" folder whereas those for the models used with local inference can be found in the "local_scripts/" folder. 

### Fine-tuning

1. Data Generation: The scripts "finetuning_data_generating.py" and "finetuning_data_preference_reasoning.py" are used to create datasets for the Generating and Preference Reasoning tasks respectively.
2. Training: The "finetuning_generation.py" script in the "local_scripts/" folder is an example of how we perform fine-tuning for the Generating Stable Solutions task, and the "finetuning_reasoning.py" script is an example of how we perform fine-tuning for the Preference Reasoning task.
3. Evaluation: Fine-tuned models are pushed to HuggingFace Hub and are evaluated in the same way as base models. The details have been temporarily omitted to preserve anonymity.
