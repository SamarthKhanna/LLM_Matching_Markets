import random
import GeneratePreference
from pathlib import Path
# from weaklyStable import weaklyStableMatch2
import csv

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

def generate_impartial_culture_pref(n, trials, csv_name=None):
    fields = ["pref_type", "n_man", "n_woman", "combined_pref_json", "man_pref_string", "woman_pref_string"]

    # fields = ["numPeopleEachSide", "preferenceProfile", "desiredMatching", "mPref", "fPref"]
    rows = []
    for i in range(trials):
        # f = open(f"LLMPrefs/{n}agents_profile{i+1}.txt", "w")
        # print(f"\nTRIAL {i}")
        rawInitPrefs = GeneratePreference.generate_random(n)
        # print(rawInitPrefs)
        mPref, fPref = cleanPrefs(rawInitPrefs)
        # print(mPref)
        # initialMatch = weaklyStableMatch2(n, mPref, fPref)
        # matchJSON = matchToJSON(initialMatch)
        allPrefJSON = cleanPrefsJSON(rawInitPrefs)
        # f.writelines(allPrefJSON)

        rows.append(["impartial culture", f"{n}", f"{n}", "".join(allPrefJSON), mPref[:-1], fPref[:-1]])
    # print(rows)
    writeCSV(fields=fields, rows=rows, n=n, trials=trials, pref_type="ic", csv_name=csv_name)
    return "Successfully created .txt and .csv files"

def generate_woman_masterlist_pref(n, trials, csv_name=None):
    # fields = ["numPeopleEachSide", "preferenceProfile", "desiredMatching", "mPref", "fPref"]
    fields = ["pref_type", "n_man", "n_woman", "combined_pref_json", "man_pref_string", "woman_pref_string"]
    rows = []
    for i in range(trials):
        # f = open(f"LLMPrefs/{n}agents_profile{i+1}.txt", "w")
        # print(f"\nTRIAL {i}")
        rawInitPrefs = GeneratePreference.generate_random(n)
        rawInitPrefs = list(rawInitPrefs)
        # print(rawInitPrefs)
        femalePrefMaster = [*range(1,n+1)]
        random.shuffle(femalePrefMaster)
        prefArrW = [femalePrefMaster]*n
        rawInitPrefs[1] = prefArrW
        mPref, fPref = cleanPrefs(rawInitPrefs)
        # print(mPref)
        # initialMatch = weaklyStableMatch2(n, mPref, fPref)
        # matchJSON = matchToJSON(initialMatch)
        allPrefJSON = cleanPrefsJSON(rawInitPrefs)
        # f.writelines(allPrefJSON)

        rows.append(["woman masterlist", f"{n}", f"{n}", "".join(allPrefJSON), mPref[:-1], fPref[:-1]])

        # rows.append([f"{n}", "".join(allPrefJSON), "".join(matchJSON), mPref[:-1], fPref[:-1]])
    # print(rows)
    writeCSV(fields=fields, rows=rows, n=n, trials=trials, pref_type="womanmaster", csv_name=csv_name)
    return "Successfully created .txt and .csv files"

def writeCSV(fields, rows, n, trials, pref_type, csv_name=None):
    file_name = csv_name if csv_name != None else f"n{n}_{pref_type}_pref_{trials}"
    with open(f"../instance_files/{file_name}.csv", 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(rows)

print(generate_impartial_culture_pref(5,10))

print(generate_woman_masterlist_pref(5,10))

# print(uniformPref_menRandomPerm(50,50))
# print(uniformPref_womenMasterPriority(25,50))
# print(uniformPref_womenMasterMod(10, 50))
# print(uniformPref_menRandomPerm(10,10))
# print(uniformPref_menRandomPerm(15,0))