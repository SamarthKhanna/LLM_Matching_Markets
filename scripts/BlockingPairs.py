

import numpy as np


# malePref = "\
# 3,4,{1,2}\n\
# 1,4,{2,3}\n\
# {3,4},{1,2}\n\
# 3,2,{1,4}"

# femalePref = "\
# {1,3},{2,4}\n\
# {3,4},{1,2}\n\
# 2,{1,3},4\n\
# 3,{2,4},1"

# malePref = "\
# {1,4},2,{3,5}\n\
# {1,3},5,4,2\n\
# 5,{1,3},4,2\n\
# 3,{1,2,5},4\n\
# {1,2,4},5,3"

# femalePref = "\
# 5,4,2,{1,3}\n\
# 3,2,4,{1,5}\n\
# {2,5},1,3,4\n\
# {2,4,5},{1,3}\n\
# 5,1,{2,3},4"

# resPref = "\
# {2,3},1\n\
# 3\n\
# 3,{1,2}\n\
# {2,1},3\n\
# 1,2"

# hosPref = "\
# 2,3,1,4\n\
# 2,3,5\n\
# 2,1,4"

# malePref = "\
# 2,1,3,4\n\
# 3,1,2,4\n\
# 1,2,3,4\n\
# 1,4,2,3"


# femalePref = "\
# 1,2,3,4\n\
# 3,1,2,4\n\
# 1,2,3,4\n\
# 1,2,3,4"

# malePref = '1,5,6,3,2,4,7,{8,9,10,11},12\n{7,2,10,1,3},4,5,6,8,9,11,12\n3,2,1,4,5,6,{7,8,9},10,11,12\n1,8,4,2,3,5,6,7,9,10,11,12\n3,2,1,4,5,6,{7,8,9,10},11,12\n1,8,{4,2,3,5,6,7},9,10,11,12\n3,2,1,4,5,6,7,8,9,10,11,12\n3,2,1,4,5,{6,7,8,9,10},11,12\n1,{8,4,2},3,5,6,7,9,10,11,12\n2,3,4,7,8,{1,12,5,6,9},10,11\n9,5,6,2,3,4,{7,8,1},10,{11,12}\n4,2,1,3,5,6,7,8,9,10,11,12'
# femalePref = '1,5,6,3,{2,4,7,8,9,10},11,12\n{7,2,10,1},3,4,5,6,{8,9,11},12\n{3,2,1,4,5,6,7,8,9,10,11,12}\n1,8,4,{2,3,5,6,7},9,10,11,12\n3,2,1,4,5,6,7,{8,9,10},11,12\n1,8,4,2,3,5,6,7,9,10,11,12\n3,2,1,{4,5},6,7,8,9,10,11,12\n2,3,4,7,8,1,12,5,6,9,10,11\n9,{5,6,2,3,4,7},8,1,10,11,12\n1,8,4,2,3,5,6,7,9,10,11,12\n1,8,4,2,3,5,6,{7,9,10,11,12}\n4,2,1,3,5,6,7,8,9,{10,11},12'

# matchArr = np.array([1, 2, 3, 8, 5, 6, 7, 4, 9, 12, 11, 10])

# malePref="{1,2,3}\n{1,3,2}\n{3,1}\n"
# femalePref="{1,2,3}\n{1,3,2}\n{2,1,3}"

malePref="1,3,4,2,5\n4,3,5,2,1\n2,3,5,1,4\n4,1,3,5,2\n4,5,2,3,1"
femalePref="4,1,5,3,2\n4,1,2,3,5\n2,4,1,3,5\n5,4,1,2,3\n5,4,2,1,3"

matchArr = np.array([1,3,2,4,5])


#matchArr = np.array([4, 3, 5, 2, 1])

class IndividualGenerator:
    def __init__(self, n, malePref, femalePref):
        '''if not (len(malePref.split()) == len(femalePref.split()) == n):
            print("The number of agents and the number of preferences given are inconsistent")
            return'''
        self.maleSet = []
        self.femaleSet = []
        for i in range(n):
            self.maleSet.append(Individual(i, "male", n, malePref))
            self.femaleSet.append(Individual(i, "female", n, femalePref))

    # good
    def allEngaged(self):
        for m in self.maleSet:
            if len(m.engagedWith) == 0:
                return False
        for f in self.femaleSet:
            if len(f.engagedWith) == 0:
                return False
        return True

    # good
    def propose(self, man, woman):
        man.engagedWith.append(woman.index)
        woman.engagedWith.append(man.index)

    # good
    def findMan(self, index):
        return self.maleSet[index]

    # good
    def findWoman(self, index):
        return self.femaleSet[index]

    # good look AT THIS
    def deletePair(self, man, woman):
        #rint(resident.index, hospital.index)
        womanRank = man.preference.reverseRank[woman.index]
        manRank = woman.preference.reverseRank[man.index]
        womanRankList = np.copy(man.preference.ranking[womanRank])
        manRankList = np.copy(woman.preference.ranking[manRank])
        #print(womanRankList)
        #print(manRankList)
        womanRankList = np.delete(womanRankList, np.argwhere(womanRankList == woman.index))
        manRankList = np.delete(manRankList, np.argwhere(manRankList == man.index))
        womanRankList = np.append(womanRankList, -1)
        manRankList = np.append(manRankList, -1)
        man.preference.ranking[womanRank] = womanRankList
        woman.preference.ranking[manRank] = manRankList

    # good CHECK
    def removeEngagement(self, man, woman):
        man.engagedWith.remove(woman.index)
        woman.engagedWith.remove(man.index)
        return

    # CHECK
    def emptyMaleList(self):
        for male in self.maleSet:
            if np.all(male.preference.ranking == -1):
                return True
        return False

    def __str__(self):
        outputString = ''
        for male in self.maleSet:
            outputString += f'\nmale: {male.index}, {male.engagedWith}'
        outputString += f'\n'
        for female in self.femaleSet:
            outputString += f'\nfemale: {female.index}, {female.engagedWith}'
        return outputString

    '''def __str__(self):
        outputString = ''
        for male in self.residentSet:
            outputString += f'\nmale: {male.index}, {list(male.preference.ranking)}, {male.engagedWith}'
        outputString += f'\n'
        for female in self.hospitalSet:
            outputString += f'\nfemale: {female.index}, {list(female.preference.ranking)}, {female.engagedWith}'
        return outputString'''


class Individual:
    def __init__(self, index, sex, n, allPrefs):
        self.index = index  # index of male or female individual
        self.sex = sex  # sex of individual (male/female)
        self.engagedWith = []
        #self.preference = RandomRanking(n, 1)
        self.preference = Ranking(index, allPrefs)
        self.originalPreferenceText = "Preference:\n"
        '''for rank in self.preference.ranking:
            self.originalPreferenceText += "\n\t(Rank " + str(rank+1) + ":"
            for index in self.preference.ranking[rank]:
                if self.sex == "male":
                    self.originalPreferenceText += " w" + str(index+1) + ","
                else:
                    self.originalPreferenceText += " m" + str(index+1) + ","
            self.originalPreferenceText = self.originalPreferenceText[:-1]
            self.originalPreferenceText += ")"'''
        self.originalPreference = self.preference.ranking.copy()

                    # good
    def isFree(self):
        if len(self.engagedWith) == 0:
            return True
        return False

    # good NOT GOOD
    def getHead(self):
        index = 0
        while self.preference.getRankOccupants(index)[0] == -1:
            index += 1
        return self.preference.getRankOccupants(index)

    def getRank(self, index):
        return self.preference.getRank(index)

    def __str__(self):
        return f"{self.sex}{self.index}"


# class Ranking:
#     def __init__(self, index, allPref):
#         initRank = makeRanking(allPref.split()[index])
#         self.reverseRank = np.full(len(allPref.split()), -1, dtype=int)

#         row = np.zeros(0)
#         arr = np.zeros(0, dtype=int)
#         for i in initRank[0].split(","):
#             #print(initRank[0])
#             row = np.concatenate([row, [int(i)-1]])
#             self.reverseRank[int(i)-1] = 0
#         arr = np.concatenate([np.zeros(0), row])
#         for i in range(1, len(initRank)):
#             row = np.zeros(0)
#             for j in initRank[i].split(","):
#                 row = np.append(row, [int(j)-1])
#                 self.reverseRank[int(j)-1] = i
#             #print(row)
#             if arr.ndim > 1:
#                 #print(np.shape(arr)[1] < row.size)
#                 if np.shape(arr)[1] < row.size:
#                     arr = np.column_stack([arr, np.full((np.shape(arr)[0], row.size - np.shape(arr)[1]), -1)])
#                 if np.shape(arr)[1] > row.size:
#                     row = np.concatenate([row, np.full(np.shape(arr)[1] - row.size, -1)])
#                 arr = np.vstack([arr, row])
#             else:
#                 if np.shape(arr)[0] < row.size:
#                     arr = np.column_stack([arr, np.full((np.shape(arr)[0], row.size - np.shape(arr)[0]), -1)])
#                 if np.shape(arr)[0] > row.size:
#                     row = np.concatenate([row, np.full(np.shape(arr)[0] - row.size, -1)])
#                 arr = np.vstack([arr, row])
#         arr = arr.astype(int)
#         self.ranking = arr

#     def getRankOccupants(self, rank):
#         return self.ranking[rank]

#     def getRank(self, index):
#         return self.reverseRank[index]

class Ranking:
    def __init__(self, index, allPref):
        initRank = makeRanking(allPref.split()[index])
        self.reverseRank = np.full(len(allPref.split()), -1, dtype=int)

        row = np.zeros(0)
        arr = np.zeros(0, dtype=int)
        # print(initRank)
        for i in initRank[0].split(","):
            #print(initRank[0])
            row = np.concatenate([row, [int(i)-1]])
            self.reverseRank[int(i)-1] = 0
        arr = np.concatenate([np.zeros(0), row])
        if len(initRank) == 1:
            arr = arr.astype(int)
            self.ranking = np.array([arr])
        else:
            for i in range(1, len(initRank)):
                row = np.zeros(0)
                for j in initRank[i].split(","):
                    row = np.append(row, [int(j)-1])
                    self.reverseRank[int(j)-1] = i
                #print(row)
                # print("ARR", arr)
                if arr.ndim > 1:
                    #print(np.shape(arr)[1] < row.size)
                    if np.shape(arr)[1] < row.size:
                        arr = np.column_stack([arr, np.full((np.shape(arr)[0], row.size - np.shape(arr)[1]), -1)])
                    if np.shape(arr)[1] > row.size:
                        row = np.concatenate([row, np.full(np.shape(arr)[1] - row.size, -1)])
                    arr = np.vstack([arr, row])
                else:
                    # print("HERE")
                    if np.shape(arr)[0] < row.size:
                        arr = np.column_stack([arr, np.full((np.shape(arr)[0], row.size - np.shape(arr)[0]), -1)])
                    if np.shape(arr)[0] > row.size:
                        row = np.concatenate([row, np.full(np.shape(arr)[0] - row.size, -1)])
                    arr = np.vstack([arr, row])
            arr = arr.astype(int)
            # print(arr)
            self.ranking = arr

    def getRankOccupants(self, rank):
        # print(rank)
        # print(self.ranking)
        # print(self.ranking[rank])
        return self.ranking[rank]

    def getRank(self, index):
        # print(index)
        # print(self.ranking)
        return self.reverseRank[index]


def makeRanking(prefStr):
    if prefStr.find("{") != -1:
        beginIndex = prefStr.find("{")
        endIndex = prefStr.find("}")
        outputArr = prefStr[:beginIndex].split(",")
        outputArr.remove("")
        outputArr.append(prefStr[beginIndex+1:endIndex])
        outputArr.extend(makeRanking(prefStr[endIndex+1:]))
        outputArr.remove("")
        return outputArr
    else:
        return prefStr.split(",")


def blockingPairs(n, malePref, femalePref, matching, type):
    individuals = IndividualGenerator(n, malePref, femalePref)
    bpCount = 0
    agentCount = 0
    bpArr = np.zeros(2, dtype=int)
    bpClassArr = np.zeros(2, dtype=str)
    agentSet = set()
    # print(matching)
    if type == "weak":
        for m in range(len(individuals.maleSet)):
            foundBP = False
            man = individuals.findMan(m)
            iteration = range(len(man.preference.ranking))
            if matching[m] != None:
                iteration = range(man.getRank(matching[m]-1))
            for j in iteration:
                # for j in range(man.getRank(matching[m]-1)):
                for w in man.preference.getRankOccupants(j):
                    if w == -1:  # FUTURE WARNING -> Addressed
                        continue
                    woman = individuals.findWoman(w)
                    # print()
                    # print(w+1, np.where(matching == w+1)[0])
                    if woman.getRank(m) < woman.getRank(np.where(matching == w+1)[0][0]):
                        bpArr = np.vstack((bpArr, [m+1, w+1]))
                        bpClassArr = np.vstack((bpClassArr, ["strict", "strict"]))
                        bpCount += 1
                        agentSet.add(w+n)
                        if foundBP == False:
                            foundBP = True
                            agentCount += 1
                            agentSet.add(m)
    elif type == "strong":
        for m in range(len(individuals.maleSet)):
            foundBP = False
            man = individuals.findMan(m)
            iteration = range(len(man.preference.ranking))
            if matching[m] != None:
                iteration = range(man.getRank(matching[m]-1))
            for j in iteration:
            # for j in range(man.getRank(matching[m]-1)+1):
                for w in man.preference.getRankOccupants(j):
                    if w == -1:  # FUTURE WARNING -> Addressed
                        continue
                    woman = individuals.findWoman(w)
                    if woman.getRank(m) < woman.getRank(np.where(matching == w+1)[0][0]) and man.getRank(w) <= man.getRank(matching[m]-1):
                        bpArr = np.vstack((bpArr, [m+1, w+1]))
                        if j == man.getRank(matching[m]-1):
                            bpClassArr = np.vstack((bpClassArr, ["weak", "strict"]))
                        else:
                            bpClassArr = np.vstack((bpClassArr, ["strict", "strict"]))
                        bpCount += 1
                    elif woman.getRank(m) == woman.getRank(np.where(matching == w+1)[0][0]) and man.getRank(w) < man.getRank(matching[m]-1):
                        bpArr = np.vstack((bpArr, [m+1, w+1]))
                        bpClassArr = np.vstack((bpClassArr, ["strict", "weak"]))
                        bpCount += 1
                        agentSet.add(w+n)
                        if foundBP == False:
                            foundBP = True
                            agentCount += 1
                            agentSet.add(m)
    elif type == "super":
        for m in range(len(individuals.maleSet)):
            foundBP = False
            man = individuals.findMan(m)
            iteration = range(len(man.preference.ranking))
            if matching[m] != None:
                iteration = range(man.getRank(matching[m]-1))
            for j in iteration:
            # for j in range(man.getRank(matching[m]-1)+1):
                for w in man.preference.getRankOccupants(j):
                    if w == -1:  # FUTURE WARNING -> Addressed
                        continue
                    woman = individuals.findWoman(w)
                    if woman.getRank(m) <= woman.getRank(np.where(matching == w+1)[0][0]) and (matching[m] != None and man.getRank(w) <= man.getRank(matching[m]-1)):
                        print(f"TRUE: {m,w}")
                        print(woman.getRank(m), woman.getRank(np.where(matching == w+1)[0][0]))
                        print(man.getRank(w), man.getRank(matching[m]-1))
                        if m == np.where(matching == w+1) or w == matching[m]-1:
                            continue
                        bpArr = np.vstack((bpArr, [m+1, w+1]))
                        if woman.getRank(m) == woman.getRank(np.where(matching == w+1)[0][0]) and man.getRank(w) == man.getRank(matching[m]-1):
                            bpClassArr = np.vstack((bpClassArr, ["weak", "weak"]))
                        elif woman.getRank(m) < woman.getRank(np.where(matching == w+1)[0][0]) and man.getRank(w) == man.getRank(matching[m]-1):
                            bpClassArr = np.vstack((bpClassArr, ["weak", "strict"]))
                        elif woman.getRank(m) == woman.getRank(np.where(matching == w+1)[0][0]) and man.getRank(w) < man.getRank(matching[m]-1):
                            bpClassArr = np.vstack((bpClassArr, ["strict", "weak"]))
                        elif woman.getRank(m) < woman.getRank(np.where(matching == w+1)[0][0]) and man.getRank(w) < man.getRank(matching[m]-1):
                            bpClassArr = np.vstack((bpClassArr, ["strict", "strict"]))
                        else:
                            continue
                        bpCount += 1
                        agentSet.add(w+n)
                        if foundBP == False:
                            foundBP = True
                            agentCount += 1
                            agentSet.add(m)
    # print(bpArr[1:])
    return {"blockingPairCount": bpCount, "blockingPairs": bpArr[1:].tolist() if bpArr.ndim > 1 else [], "blockingPairsClass": bpClassArr[1:].tolist() if bpClassArr.ndim > 1 else [], "agentsInvolved": len(agentSet)}





# f = IndividualGenerator(4)
# bp = blockingPairs(5, malePref, femalePref, matchArr, "weak")
# print(bp)
