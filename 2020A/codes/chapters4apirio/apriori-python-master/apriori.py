import sys
import os.path
import csv
import math 
import types
from collections import defaultdict, Iterable
import itertools

class Apriori:
    def __init__(self, data, minSup, minConf):
        self.dataset = data
        self.transList = defaultdict(list)
        self.freqList = defaultdict(int)
        self.itemset = set()
        self.highSupportList = list()
        self.numItems = 0
        self.prepData()             # initialize the above collections

        self.F = defaultdict(list)

        self.minSup = minSup
        self.minConf = minConf

    def genAssociations(self):
        candidate = {}
        count = {}

        self.F[1] = self.firstPass(self.freqList, 1)
        k=2
        while len(self.F[k-1]) != 0:
            candidate[k] = self.candidateGen(self.F[k-1], k)
            for t in self.transList.items():
                for c in candidate[k]:
                    if set(c).issubset(t[1]):
                        self.freqList[c] += 1

            self.F[k] = self.prune(candidate[k], k)
            if k > 2:
                self.removeSkyline(k, k-1)
            k += 1

        return self.F

    def removeSkyline(self, k, kPrev):
        for item in self.F[k]:
            subsets = self.genSubsets(item)
            for subset in subsets:
                if subset in (self.F[kPrev]):
                    self.F[kPrev].remove(subset)
                    

        subsets = self.genSubsets

    def prune(self, items, k):
        f = []
        for item in items:
            count = self.freqList[item]
            support = self.support(count)
            if support >= .95:
                self.highSupportList.append(item)
            elif support >= self.minSup:
                f.append(item)

        return f

    def candidateGen(self, items, k):
        candidate = []

        if k == 2:
            candidate = [tuple(sorted([x, y])) for x in items for y in items if len((x, y)) == k and x != y]
        else:
            candidate = [tuple(set(x).union(y)) for x in items for y in items if len(set(x).union(y)) == k and x != y]
        
        for c in candidate:
            subsets = self.genSubsets(c)
            if any([ x not in items for x in subsets ]):
                candidate.remove(c)

        return set(candidate)

    def genSubsets(self, item):
        subsets = []
        for i in range(1,len(item)):
            subsets.extend(itertools.combinations(item, i))
        return subsets

    def genRules(self, F):
        H = []

        for k, itemset in F.items():
            if k >= 2:
                for item in itemset:
                    subsets = self.genSubsets(item)
                    for subset in subsets:
                        if len(subset) == 1:
                            subCount = self.freqList[subset[0]]
                        else:
                            subCount = self.freqList[subset]
                        itemCount = self.freqList[item]
                        if subCount != 0:
                            confidence = self.confidence(subCount, itemCount)
                            if confidence >= self.minConf:
                                support = self.support(self.freqList[item])
                                rhs = self.difference(item, subset)
                                if len(rhs) == 1:
                                    H.append((subset, rhs, support, confidence))

        return H

    def difference(self, item, subset):
        return tuple(x for x in item if x not in subset)

    def confidence(self, subCount, itemCount):
        return float(itemCount)/subCount

    def support(self, count):
        return float(count)/self.numItems

    def firstPass(self, items, k):
        f = []
        for item, count in items.items():
            support = self.support(count)
            if support == 1:
                self.highSupportList.append(item)
            elif support >= self.minSup:
                f.append(item)

        return f

    """
    Prepare the transaction data into a dictionary
    key: Receipt.id
    val: set(Goods.Id) 

    Also generates the frequent itemlist for itemsets of size 1
    key: Goods.Id
    val: frequency of Goods.Id in self.transList
    """
    def prepData(self):
        key = 0
        for basket in self.dataset:
            self.numItems += 1
            key = basket[0]
            for i, item in enumerate(basket):
                if i != 0:
                    self.transList[key].append(item.strip())
                    self.itemset.add(item.strip())
                    self.freqList[(item.strip())] += 1

def main():
    goods = defaultdict(list)
    num_args = len(sys.argv)
    minSup = minConf = 0
    noRules = False

    # Make sure the right number of input files are specified
    if  num_args < 4 or num_args > 5:
        print('Expected input format: python apriori.py <dataset.csv> <minSup> <minConf>')
        return
    elif num_args == 5 and sys.argv[1] == "--no-rules":
        dataset = csv.reader(open(sys.argv[2], "r"))
        goodsData = csv.reader(open('goods.csv', "r"))
        minSup  = float(sys.argv[3])
        minConf = float(sys.argv[4])
        noRules = True
        print("Dataset: ", sys.argv[2], " MinSup: ", minSup, " MinConf: ", minConf)
    else: 
        dataset = csv.reader(open(sys.argv[1], "r"))
        goodsData = csv.reader(open('goods.csv', "r"))
        minSup  = float(sys.argv[2])
        minConf = float(sys.argv[3])
        print("Dataset: ", sys.argv[1], " MinSup: ", minSup, " MinConf: ", minConf)

    print ("==================================================================")

    for item in goodsData:
        goods[item[0]] = item[1:]

    a = Apriori(dataset, minSup, minConf)

    frequentItemsets = a.genAssociations()

    count = 0
    for k, item in frequentItemsets.items():
        for i in item:
            if k >= 2:
                count += 1
                print (count,":  ",readable(i, goods),"\tsupport=",a.support(a.freqList[i]))

    print ("Skyline Itemsets: ", count)
    if not noRules:
        rules = a.genRules(frequentItemsets)
        for i, rule in enumerate(rules):
            print ("Rule",i+1,":\t ",readable(rule[0], goods),"\t-->",readable(rule[1], goods),"\t [sup=",rule[2]," conf=",rule[3],"]")

    print ("\n")

def readable(item, goods):
    itemStr = ''
    for k, i in enumerate(item):
        itemStr += goods[i][0] + " " + goods[i][1] +" (" + i + ")"
        if len(item) != 0 and k != len(item)-1:
            itemStr += ",\t"

    return itemStr.replace("'", "")

if __name__ == '__main__':
    main()
