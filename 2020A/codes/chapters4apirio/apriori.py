# -*- coding:gb2312 -*-
import sys
import copy

def init_pass(T):
    C = {}  #C为字典
    for t in T:
        for i in t:

            if i in C.keys():
                C[i] += 1
            else:
                C[i] = 1
    return C

def generate(F):
    C = []
    k = len(F[0]) + 1
    for f1 in F:
        for f2 in F:
            if f1[k-2] < f2[k-2]:
                c = copy.copy(f1)
                c.append(f2[k-2])
                flag = True
                for i in range(0,k-1):
                    s = copy.copy(c)
                    s.pop(i)
                    if s not in F:
                        flag = False
                        break
                if flag and c not in C:
                    C.append(c)
    return C

def compareList(A,B):
    if len(A) <= len(B):
        for a in A:
            if a not in B:
                return False
    else:
        for b in B:
            if b not in A:
                return False
    return True

def apriori(T,minSupport):
    D=[]
    C=init_pass(T)

    keys=sorted(C)
    D.append(keys)#加入D集中
    F=[[]]
    for f in D[0]:
        if C[f]>=minSupport:
            F[0].append([f])
    k=1

    while F[k-1]!=[]:
        D.append(generate(F[k-1]))
        F.append([])
        for c in D[k]:
            count = 0;
            for t in T:
                if compareList(c,t):
                    count += 1
            if count>= minSupport:
                F[k].append(c)
        k += 1

    U = []
    for f in F:
        for x in f:
            U.append(x)
    return U


T = [['A','C','D'],['B','C','E'],['A','B','C','E'],['B','E']]

Z= apriori(T,2)
print(Z)