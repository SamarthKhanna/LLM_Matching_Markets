import resource
from resource import *
import numpy as np
from random import random, randint
from math import factorial, e

'''n=200

'''

f = 80

malePref = "\
3,1,5,7,4,2,8,6\n\
6,1,3,4,8,7,5,2\n\
7,4,3,6,5,1,2,8\n\
5,3,8,2,6,1,4,7\n\
4,1,2,8,7,3,6,5\n\
6,2,5,7,8,4,3,1\n\
7,8,1,6,2,3,4,5\n\
2,6,7,1,8,3,4,5"

femalePref = "\
4,3,8,1,2,5,7,6\n\
3,7,5,8,6,4,1,2\n\
7,5,8,3,6,2,1,4\n\
6,4,2,7,3,1,5,8\n\
8,7,1,5,6,4,3,2\n\
5,4,7,6,2,8,3,1\n\
1,4,5,6,2,8,3,7\n\
2,5,4,3,7,8,1,6"

'''malePref = "\
1,2,3,4,5\n\
1,2,3,4,5\n\
1,2,3,4,5\n\
1,2,3,4,5\n\
1,2,3,4,5"




femalePref = "\
1,2,3,5,4\n\
5,4,3,2,1\n\
4,3,2,5,1\n\
4,3,2,5,1\n\
4,2,5,3,1"'''


'''
malePref = "\
1,2,3,4,5\n\
1,2,3,4,5\n\
1,2,3,4,5\n\
1,2,3,4,5\n\
1,2,3,4,5"




femalePref = "\
1,2,3,5,4\n\
4,3,5,2,1\n\
4,3,2,5,1\n\
4,3,2,5,1\n\
4,2,3,1,5"
'''

malePref = "\
1,2,3,4\n\
1,3,2,4\n\
4,1,2,3\n\
1,4,2,3"

femalePref = "\
1,4,3,2\n\
4,3,2,1\n\
4,3,1,2\n\
2,3,4,1"

# This class allows you to generate random strict and complete preferences
class RandomPreferenceGenerator:
    def __init__(self, n):
        self.n = n
        self.prefString = ""

    def generate(self):
        for i in range(self.n):
            arr = np.arange(1, self.n+1)
            np.random.shuffle(arr)
            arrStr = [str(integer) for integer in arr]
            self.prefString += ",".join(list(arrStr)) + f"\n"
        return self.prefString[:-1]


# malePref = RandomPreferenceGenerator(f).generate()
# femalePref = RandomPreferenceGenerator(f).generate()

# This class performs all preprocessing and creates all data structures necessary for the algorithm to run
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

        # testing
        '''self.residentSet[0].preference.ranking = {0: [2, 1], 1: [0]}
        self.residentSet[1].preference.ranking = {0: [1], 1: [0], 2: [2]}
        self.residentSet[2].preference.ranking = {0: [0, 1, 2]}

        self.hospitalSet[0].preference.ranking = {0: [0, 1], 1: [2]}
        self.hospitalSet[1].preference.ranking = {0: [1], 1: [0, 2]}
        self.hospitalSet[2].preference.ranking = {0: [2, 1, 0]}'''



    # Check if all agents are engaged
    def allEngaged(self):
        for m in self.maleSet:
            if len(m.engagedWith) == 0:
                return False
        for f in self.femaleSet:
            if len(f.engagedWith) == 0:
                return False
        return True

    # engage a man and a woman
    def propose(self, man, woman):
        man.engagedWith.append(woman.index)
        woman.engagedWith.append(man.index)

    # find a man's Individual object via his index
    def findMan(self, index):
        return self.maleSet[index]

    # find a woman's Individual object via her index
    def findWoman(self, index):
        return self.femaleSet[index]

    # Delete a (m,w) pair from both agents' preference lists
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

        '''print("TESTING")
        print(man.index, woman.index)
        print(womanRankList)
        print(man.preference.ranking)'''

        man.preference.ranking[womanRank] = womanRankList
        woman.preference.ranking[manRank] = manRankList

    # remove an engagement between two agents
    def removeEngagement(self, man, woman):
        man.engagedWith.remove(woman.index)
        woman.engagedWith.remove(man.index)
        return

    # check if any man has an empty preference list (if he has been rejected by all women)
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

# This class is the data structure which holds all the necessary information relating to a certain agent
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

    # Check if an agent is single
    def isFree(self):
        if len(self.engagedWith) == 0:
            return True
        return False

    # Get the head of an agent's preference list (the first potential valid partner)
    def getHead(self):
        index = 0
        while self.preference.getRankOccupants(index)[0] == -1:
            index += 1
        return self.preference.getRankOccupants(index)

    # Get the rank of an agent relative to the current Individual's preferences
    def getRank(self, index):
        return self.preference.getRank(index)

    def __str__(self):
        return f"{self.sex}{self.index}"

# This class can create randomly generated preferences for weak and complete preferences
class RandomRanking:
    def __init__(self, N, K):
        b = self.getBellNumber(N)
        self.ranking = np.zeros(1)
        self.unm = lambda n, m, b: ((m ** n) / (factorial(m) * b)) / e
        self.getRandomSet(N, b)


    def genBellNumbers(self, n):
        B = [0] * (n + 1)  # Bell Numbers
        # Stirling numbers of the second kind
        S = [[0] * (n + 1) for i in range(n + 1)]
        S[0][1] = 1
        B[0] = 1
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                S[i][j] = S[i - 1][j] * j + S[i - 1][j - 1]
                B[i] += S[i][j]
            S[i - 1] = []  # clean Stirling numbers (ram management)
        return B

    def getBellNumber(self, n):
        B = self.genBellNumbers(n)
        return B[n]

    def getM(self, n, b):
        p = random()
        m = 0
        while (p > 0):
            p -= self.unm(n, m, b)
            m += 1
        return m

    # 1 - choose M from unm (getM)
    # 2 - Drop n labelled balls uniformly into M boxes
    # 3 - Form a set partition λ of [n] with i and j in the same block if and
    # only if balls i and j are in the same box

    def getRandomSet(self, n, b):
        m = self.getM(n, b)
        b = []
        maxBucket = 0
        for i in range(n):
            b += [randint(0, m - 1)]
            if b[-1] > maxBucket:
                maxBucket = b[i]

        '''norm = [-1] * m
        normC = 0
        for i in range(n):
            if norm[b[i]] < 0:
                norm[b[i]] = normC
                normC += 1
            b[i] = norm[b[i]]
            if b[i] > maxBucket:
                maxBucket = b[i]'''

        self.ranking = np.full(shape=(maxBucket + 1, 1), fill_value=-1, dtype=int)
        self.reverseRank = np.full(shape=n, fill_value=-1, dtype=int)
        for i in range(len(b)):
            self.reverseRank[i] = b[i]
            if self.ranking[b[i]][-1] == -1:
                self.ranking[b[i]][np.argmin(self.ranking[b[i]])] = i
            else:
                self.ranking = np.hstack((self.ranking, np.full((np.shape(self.ranking)[0], 1), -1)))
                self.ranking[b[i]][np.argmin(self.ranking[b[i]])] = i
        #return self.ranking
        #print()
        #print(self.ranking)
        self.ranking = np.delete(self.ranking, np.where(self.ranking[:, 0] == -1), axis=0)
        for i in range(len(self.ranking)):
            for j in self.ranking[i]:
                if j == -1:
                    continue
                print(j, i)
                self.reverseRank[j] = i
        '''print()
        print(self.ranking)
        print(self.reverseRank)
        print(len(self.ranking))'''

    def getRankOccupants(self, rank):
        return self.ranking[rank]

    def getRank(self, index):
        return self.reverseRank[index]

# This class can turn preferences (given in the PrefLib format) into arrays which can be inputted into the algorithm
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

# This his a helper function for the Ranking class
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


# separate operation for populating and randomizing ranking
# allows for pre-determined or random ranking

# Old function, just for reference
'''def weaklyStableMatch(n):
    data = IndividualGenerator(n)
    print(resource.getrusage(RUSAGE_SELF))

    while not data.emptyMaleList():
        for man in data.maleSet:
            if data.emptyMaleList():
                return None
            if man.isFree():
                for womanIndex in [man.getHead()[0]]:
                    if womanIndex == -1:
                        continue
                    woman = data.findWoman(womanIndex)
                    data.propose(man, woman)
                    for i in woman.engagedWith[:-1]:
                        #if i == -1:
                            #continue
                        data.removeEngagement(data.findMan(i), woman)
                    for i in range(woman.getRank(man.index), len(woman.preference.ranking)):
                        for mpref in woman.preference.getRankOccupants(i):  # MIGHT CAUSE ISSUE
                            if mpref == -1 or mpref == man.index:
                                continue
                            data.deletePair(data.findMan(mpref), woman)
        if data.allEngaged():
            return data'''

# This function runs Irving's weakly-stable stable marriage algorithm
def weaklyStableMatch2(n, malePref, femalePref, verbose=False):
    data = IndividualGenerator(n, malePref, femalePref)

    comments = []

    '''for m in data.maleSet:
        print(m.index)
        print(m.preference.ranking)
        print()
    for m in data.femaleSet:
        print(m.index)
        print(m.preference.ranking)
        print()'''

    while not data.emptyMaleList():
        for man in data.maleSet:
            # print(data)
            if data.emptyMaleList():
                return None
            if man.isFree():
                woman = data.findWoman(man.getHead()[0])
                data.propose(man, woman)
                comments.append(f"M{man.index+1} is free. M{man.index+1} proposes to W{woman.index+1}. W{woman.index+1} accepts the proposal. Now M{man.index+1} and W{woman.index+1} are matched.")
                # print(comments)
                wEngagementCopy = woman.engagedWith[:-1].copy()
                for i in wEngagementCopy:
                    data.removeEngagement(data.findMan(i), woman)
                    comments.append(f"W{woman.index+1} prefers M{man.index+1}, so W{woman.index+1} breaks her engagement with M{i+1}.")
                menDeleted = []
                for i in range(woman.getRank(man.index), len(woman.preference.ranking)):
                    tempPref = woman.preference.getRankOccupants(i).copy()
                    for mpref in tempPref:  # MIGHT CAUSE ISSUE
                        if mpref == -1 or mpref == man.index:
                            continue
                        data.deletePair(data.findMan(mpref), woman)
                        menDeleted.append(mpref)
                if len(menDeleted) != 0:
                    menString = ""
                    for i in menDeleted:
                        menString += f"M{i+1}, "
                    comments.append(f"W{man.index+1} deletes {menString[:-2]} from her list. {menString[:-2]} delete W{woman.index+1} from their list.")
                    
        # print(comments)
        if verbose:
            return comments

        if data.allEngaged():
            '''for m in data.maleSet:
                print(m.index)
                print(m.preference.ranking)
                print()
            for m in data.femaleSet:
                print(m.index)
                print(m.preference.ranking)
                print()'''
            return data


d = weaklyStableMatch2(4,malePref,femalePref)
# print(d)
# print(resource.getrusage(RUSAGE_SELF))
