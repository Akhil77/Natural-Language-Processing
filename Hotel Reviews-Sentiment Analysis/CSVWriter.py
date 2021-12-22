from collections import Counter
import math
import re
import numpy as np
import csv

positive_file = open('hotelPosT-train.txt',encoding= "utf8")
negative_file = open('hotelNegT-train.txt',encoding= "utf8")
positive_words = open('positive-words.txt',encoding= "utf8")
negative_words = open('negative-words.txt',encoding= "utf8")


positive_w_set = set()
negative_w_set = set()
positive_lines = []
negative_lines = []
IDs = []

for i in positive_file:
    m = re.search("(ID-[0-9]+)", i)
    IDs.append(m.group(1))
    p = re.split('ID-[0-9]+', i)
    p = p[1].replace("!"," !")
    p=re.sub("[^A-Za-z\+\-'!]"," ",p)
    p = re.split(" ",p)
    p = [c.lower() for c in p if c!='' and c!= ' ']
    positive_lines.append(p)

for i in negative_file:
    m = re.search("(ID-[0-9]+)", i)
    IDs.append(m.group(1))
    p = re.split('ID-[0-9]+', i)
    p = p[1].replace("!"," !")
    p=re.sub("[^A-Za-z\+\-'!]"," ",p)
    p = re.split(" ",p)
    p = [c.lower() for c in p if c!='' and c!= ' ']
    negative_lines.append(p)

for i in positive_words:
    p=re.sub("\n","",i)
    positive_w_set.add(p)

for i in negative_words:
    p=re.sub("\n","",i)
    negative_w_set.add(p)

#Feature Extraction
pronouns = set(["i", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours"])
def featureExtraction(text, classPN):
    for i in text:
        count = 0
        x1,x2,x3,x4,x5,x6,x7 = 0,0,0,0,0,0,1
        for j in i:
            if j in positive_w_set:
                x1+=1
            if j in negative_w_set:
                x2+=1
            if j == 'no':
                x3=1
            if j in pronouns:
                x4 += 1
            if j == '!':
                x5 = 1
            count += 1
        x6 = math.log(count)
        X.append([x1,x2,x3,x4,x5,x6,x7])
        Y.append(classPN)

X = []
Y = []
featureExtraction(positive_lines,1)
featureExtraction(negative_lines,0)
#print(negative_lines)
L=[[] for _ in range(len(X))]
for i in range(len(X)):
    L[i] = [IDs[i],X[i][0], X[i][1], X[i][2], X[i][3], X[i][4], X[i][5], Y[i]]

def write_output(file, L):
    with open(file, "w") as f:
        wr = csv.writer(f)
        wr.writerows(L)
write_output('mallela-srinivasakhil-assgn2-part1.csv', L)