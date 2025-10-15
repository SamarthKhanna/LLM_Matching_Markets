from components import PromptGeneratorPt2b
from collections import defaultdict

pg = PromptGeneratorPt2b('5_ic_processed.csv', num_instances=1)
pg.get_prompts_list()

llm_answer_list = [2,2,1,5,2]
counts = defaultdict(list)
invalid = False
for m, woman in enumerate(llm_answer_list):
    counts[woman].append(m+1)
    if woman != 0 and len(counts[woman]) > 1:
        invalid = True

round2prompt = pg.incomplete_matching_prompt(pg.prompts_list[0][1], "xx", counts, set([w+1 for w in range(5)]) - set(llm_answer_list), 5)

prompt = round2prompt.replace('<', '$<$').replace('>', '$>$').replace('{', '\{').replace('}', '\}').replace('\n', '\n\n')

print(prompt)
# cleaned = []
# for row in pg.prompts_list:
#     prompt = row[1].replace('<', '$<$').replace('>', '$>$').replace('{', '\{').replace('}', '\}').replace('\n', '\n\n')
#     # prompt = row[1]
#     cleaned.append(prompt)
#     # print(row[1])
#     print(prompt)
# # print(cleaned[2])

# # print(cleaned)