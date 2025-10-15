import random 
import os
import GeneratePreference
import csv
from weaklyStable import weaklyStableMatch2
from tqdm import tqdm as tqdm
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset
import pandas as pd
import pickle as pkl

class ListNode:
    def __init__(self, val=None, next=None, prev=None) -> None:
        self.val = val
        self.next = next


class DoubleListNode:
    def __init__(self, val=None, next=None, prev=None) -> None:
        self.val = val
        self.next = next
        self.prev = prev

def cleanPrefs(prefArr):
    malePref = ""
    femalePref = ""
    for i in prefArr[0]:
        malePref = malePref + ",".join(map(str, i)) + "\n"
    for j in prefArr[1]:
        femalePref = femalePref + ",".join(map(str, j)) + "\n"
    return malePref, femalePref

def cleanPrefsMaster(prefArr):
    malePref = ""
    femalePref = ""
    n = len(prefArr[1])
    femalePrefMaster = [*range(1,n+1)]
    random.shuffle(femalePrefMaster)
    prefArrW = [femalePrefMaster]*n
    # print(prefArrW)
    for i in prefArr[0]:
        malePref = malePref + ",".join(map(str, i)) + "\n"
    for j in prefArrW:
        femalePref = femalePref + ",".join(map(str, j)) + "\n"
    return malePref, femalePref

def cleanJSON_M(val):
    return f"W{val}"

def cleanJSON_W(val):
    return f"M{val}"

def cleanPrefsJSON(prefArr):
    pref = ["{\n", "M: {\n"]
    # print(prefArr)
    for index, i in enumerate(prefArr[0]):
        # malePref = malePref + ",".join(map(str, i)) + "\n"
        pref.append(f"M{index+1}: [{','.join(map(cleanJSON_M, i))}],\n")

    pref.append("},\n")
    pref.append("W: {\n")
    for index, j in enumerate(prefArr[1]):
        # femalePref = femalePref + ",".join(map(str, j)) + "\n"
        pref.append(f"W{index+1}: [{','.join(map(cleanJSON_W, j))}],\n")
    pref.append("}}")
    return pref

def matchToJSON(data):
    matchJSONArr = ["["]
    for m in data.maleSet:
        matchJSONArr.append(f"[M{m.index+1}, W{m.engagedWith[0]+1}],")
    matchJSONArr.append(f"]")
    return matchJSONArr

def matchListToJSON(match_list):
    matchJSONArr = ["["]
    for m, w in enumerate(match_list):
        matchJSONArr.append(f"[M{m+1}, W{w}],")
    matchJSONArr.append(f"]")
    return "".join(matchJSONArr)

def matchToList(data):
    matchList = []
    for m in data.maleSet:
        matchList.append(m.engagedWith[0]+1)
    return matchList

def generate_impartial_culture_pref(n_min, n_max, trials):
    rows = []
    for i in tqdm(range(trials)):
        n = random.randint(n_min, n_max)
        rawInitPrefs = GeneratePreference.generate_random(n)
        mPref, fPref = cleanPrefs(rawInitPrefs)
        allPrefJSON = cleanPrefsJSON(rawInitPrefs)
        rows.append(["impartial culture", f"{n}", "".join(allPrefJSON), mPref[:-1], fPref[:-1]])

    return rows

def generate_woman_masterlist_pref(n_min, n_max, trials):
    rows = []
    for i in tqdm(range(trials)):
        n = random.randint(n_min, n_max)
        rawInitPrefs = GeneratePreference.generate_random(n)
        rawInitPrefs = list(rawInitPrefs)
        femalePrefMaster = [*range(1,n+1)]
        random.shuffle(femalePrefMaster)
        prefArrW = [femalePrefMaster]*n
        rawInitPrefs[1] = prefArrW
        mPref, fPref = cleanPrefs(rawInitPrefs)
        allPrefJSON = cleanPrefsJSON(rawInitPrefs)
        rows.append(["woman masterlist", f"{n}", "".join(allPrefJSON), mPref[:-1], fPref[:-1]])

    return rows


def swap_partners(match_list, swap_count):
    n = len(match_list)
    mod_list = match_list.copy()
    swap_agents = random.sample(range(n), swap_count)
    swap_partners = [match_list[i] for i in swap_agents]

    print(swap_agents)

    for idx, agent in enumerate(swap_agents):
        mod_list[agent] = swap_partners[(idx+1) % swap_count]

    return mod_list

def jaccardSimilarity(list1, list2):
    set1 = set()
    set2 = set()
    for m, w in enumerate(list1):
        set1.add((m+1, w))
    for m, w in enumerate(list2):
        set2.add((m+1, w))

    jaccard_sim = len(set1.intersection(set2))/len(set1.union(set2))

    return len(set1.intersection(set2)), jaccard_sim


def populate_prefs(instance):
    men_prefs_list = []
    women_prefs_list = []
    men_prefs_dict = {}
    women_prefs_dict = {}
    men_prefs = instance[3].split('\n')
    women_prefs = instance[4].split('\n')
    n = int(instance[1])
    for i in range(n):
        row, prev = None, None
        men_prefs_dict[i] = {}
        man_list = men_prefs[i].split(',')
        for j, ind in enumerate(man_list):
            partner = int(ind)-1
            men_prefs_dict[i][partner] = j
            if not prev:
                row = prev = ListNode(partner)
            else:
                node = ListNode(partner)
                prev.next = node
                prev = prev.next
        men_prefs_list.append(row)
        row, prev = None, None
        women_prefs_dict[i] = {}
        woman_list = women_prefs[i].split(',')
        for j, ind in enumerate(woman_list):
            partner = int(ind)-1
            women_prefs_dict[i][partner] = j
            if not prev:
                row = prev = ListNode(partner)
            else:
                node = ListNode(partner)
                prev.next = node
                prev = prev.next
        women_prefs_list.append(row)

    return men_prefs_list, women_prefs_list, men_prefs_dict, women_prefs_dict


# Vanilla Deferred Acceptance
def gale_shapley(
    men_prefs_list,
    women_prefs_list,
    men_prefs_dict,
    women_prefs_dict,
    direction_change=False,
    verbose=False
):
    if direction_change:
        man_text = "Woman"
        woman_text = "Man"
        man_letter = "W"
        woman_letter = "M"
    else:
        man_text = "Man"
        woman_text = "Woman"
        man_letter = "M"
        woman_letter = "W"

    comments = []

    male_side = {i: None for i in range(len(men_prefs_list))}
    female_side = {i: None for i in range(len(women_prefs_list))}
    proposer = 0
    while proposer < len(men_prefs_list):
        if male_side[proposer] is not None or not men_prefs_list[proposer]:
            proposer += 1
            continue
        candidate = men_prefs_list[proposer].val
        # print(f"{man_text} {proposer} proposes to {woman_text} {candidate}.")
        comments.append(f"M{proposer+1} is free. M{proposer+1} proposes to W{candidate+1}")
        men_prefs_list[proposer] = men_prefs_list[proposer].next
        if female_side[candidate] is None:
            male_side[proposer] = candidate
            female_side[candidate] = proposer
            # print(f"{man_text} {proposer} gets engaged to {woman_text} {candidate} who was free before this.")
            comments.append(f"Since W{candidate+1} is free, W{candidate+1} accepts the proposal. Now M{proposer+1} and W{candidate+1} are matched.")
            # print(f"{man_letter}{proposer} --> {woman_letter}{candidate} (free) | Accepted")
            proposer += 1
        else:
            other = female_side[candidate]
            # print(f"{man_text} {proposer} proposes to {woman_text} {candidate} who is currently paired with {man_text} {other}.")
            if (
                women_prefs_dict[candidate][proposer]
                > women_prefs_dict[candidate][other]
            ):
                comments.append(f"Since W{candidate+1} prefers their current partner M{other+1} to M{proposer+1}, W{candidate+1} rejects the proposal. M{other+1} and W{candidate+1} are still matched, and M{proposer+1} is still free.")
                # print(f"{woman_text} {candidate} prefers their old partner, {man_text} {other} to {man_text} {proposer}.")
                # print(f"{man_letter}{proposer} --> {woman_letter}{candidate} ({man_letter}{other}) | Rejected")
                # print("", end="")
            else:
                female_side[candidate] = proposer
                male_side[proposer] = candidate
                male_side[other] = None
                comments.append(f"Since W{candidate+1} prefers M{proposer+1} to their current partner M{other+1}, W{candidate+1} accepts the proposal. Now M{proposer+1} and W{candidate+1} are matched, and M{other+1} is free.")
                # print(f"{woman_text} {candidate} prefers {man_text} {proposer} to their old partner, {man_text} {other}. The switch is made.")
                # print(f"{man_letter}{proposer} --> {woman_letter}{candidate} ({man_letter}{other}) | Accepted")
                proposer = other
    if verbose:
        return comments
    return male_side, female_side, men_prefs_list, women_prefs_list

def get_cot_shortlist_prompt(instance):
    n = int(instance[1])
    m_pref = instance[3]
    w_pref = instance[4]

    comments_list = weaklyStableMatch2(n, m_pref, w_pref, verbose=True)
    comments_str = "\n".join(comments_list)

    men_prefs_list, women_prefs_list, men_prefs_dict, women_prefs_dict = populate_prefs(instance)
    comments_list = gale_shapley(men_prefs_list, women_prefs_list, men_prefs_dict, women_prefs_dict, verbose=True)
    comments_str = "\n".join(comments_list)

    match_sample = weaklyStableMatch2(n, m_pref, w_pref)

    json_sample_sol = "{\n"
    for m in range(1, n):
        json_sample_sol += f"\t\"M{m}\": \"W{match_sample.findMan(m-1).engagedWith[0]+1}\",\n"
    json_sample_sol += f"\t\"M{m+1}\": \"W{match_sample.findMan(m-1+1).engagedWith[0]+1}\"\n"
    json_sample_sol += "}"

    vanilla_algo = """
1. Initialize all men and women as unmatched.
2. Create a list to keep track of each man's next proposal (initially set to 0 for all men).
3. While there are unmatched men:
   a. Select an unmatched man (M).
   b. Find the next woman (W) on M's preference list that he hasn't proposed to yet.
   c. If W is unmatched, match M and W.
   d. If W is matched but prefers M to her current partner:
      - Unmatch W from her current partner.
      - Match M and W.
      - Set the unmatched man as W's previous partner.
   e. If W rejects M, move to the next woman on M's preference list.
4. Repeat step 3 until all men are matched.
"""

    json_format = "{\n"
    for m in range(1, n):
        json_format += f"\t\"M{m}\": \"<woman matched with M{m}>\",\n"
    json_format += f"\t\"M{m+1}\": \"<woman matched with M{m+1}>\"\n"
    json_format += "}"
    prompt = f"Consider the following instance of the two-sided matching problem, where {n} men are to be matched with {n} women. \n\
Here are the preference lists for all individuals:\n\n\
<preferences>\n\
{instance[2]}\n\
</preferences>\n\n\
Your task is to find the proposer-optimal stable matching.\n\n\
Once you have found a stable matching, please return your matching in the JSON format given below:\n\n\
<answer>\n\
{json_format}\n\
</answer>\n\n\
Make sure that each man/woman is matched with exactly ONE partner. It is important that you enclose your JSON object in <answer></answer> tags.\
"
    cot = f"Okay, so I need to find a stable matching for {n} men and {n} women. For this, I can use the Gale-Shapley \
algorithm. Before I compute the solution for this instance, I will list the steps of the Gale-Shapley algorithm. \n\
{vanilla_algo}\n\
Next, I will exectute the above algorithm for the given instance. The steps are as follows:\n\n\
{comments_str}"
    
    response = f"<answer>\n{json_sample_sol}\n</answer>"
    return prompt, cot, response

def run_new_examples(culture, sizes=[5,20], num_instances = 1000, train_frac = 1):
    output_file_name = f"../finetuning_data/{culture}_{sizes[0]}_{sizes[1]}_{num_examples}.csv"
    # if os.path.exists(output_file_name): 
    #     print('Already exists!')
    #     return f"{output_file_name} already exists!"
    
    fields = ["pref_type", "n", "combined_pref_json", "man_pref_string", "woman_pref_string", "men_opt", "Question", "Complex_CoT", "Response"]

    if culture == 'ic':
        instances = generate_impartial_culture_pref(sizes[0], sizes[1], num_instances)
    elif culture == 'wm':
        instances = generate_woman_masterlist_pref(sizes[0], sizes[1], num_instances)
    else:
        instances = generate_impartial_culture_pref(sizes[0], sizes[1], num_instances//2)
        instances += generate_woman_masterlist_pref(sizes[0], sizes[1], num_instances//2)
        random.shuffle(instances)

    rows = []
    print("STARTING")
    for i, row in enumerate(tqdm(instances)):
        curr_row = [row[0], row[1], row[2], row[3], row[4]]
        n = int(row[1])
        man_pref_str = row[3]
        woman_pref_str = row[4]
        man_opt_sol = weaklyStableMatch2(n, man_pref_str, woman_pref_str)
        man_opt_sol_list = matchToList(man_opt_sol)

        curr_row.append(matchListToJSON(man_opt_sol_list))

        prompt, cot, response = get_cot_shortlist_prompt(curr_row)

        if i < 1:
            print(prompt)
            print(cot)
            print(response)

        curr_row.extend([prompt, cot, response])

        rows.append(curr_row)
        
        if len(rows) > num_instances - 1:
            break

    writeCSV(fields=fields, rows=rows, csv_name=output_file_name)

    train_rows = rows[:round(train_frac*num_instances)]
    test_rows = rows[round(train_frac*num_instances):]

    train = pd.DataFrame(train_rows, columns = fields)
    test = pd.DataFrame(test_rows, columns = fields)

    train_data = Dataset(pa.Table.from_pandas(train))
    test_data = Dataset(pa.Table.from_pandas(test))

    with open(f'../finetuning_data/train_data_{culture}_{sizes[0]}_{sizes[1]}_{num_instances}.pkl', 'wb') as file:
        pkl.dump(train_data, file)
    with open(f'../finetuning_data/test_data_{culture}_{sizes[0]}_{sizes[1]}_{num_instances}.pkl', 'wb') as file:
        pkl.dump(test_data, file)

    return "Successfully created .txt and .csv files"

def writeCSV(fields, rows, csv_name=None):
    file_name = csv_name
    with open(f"{file_name}.csv", 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(rows)

max_size = 50
min_size = 5
num_examples = 50000

for culture in ['both']:
    print("HERE")
    run_new_examples(culture, [min_size, max_size], num_examples)
