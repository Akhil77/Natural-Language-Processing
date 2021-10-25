from collections import Counter
import math
import re
import numpy as np
import csv


X = []
Y = []
file = open("mallela-srinivasakhil-assgn2-part1.csv")
csvreader = csv.reader(file)
rows = []
positiveLines = []
negativeLines = []
YP = []
YN = []

#Read training set from mallela-srinivasakhil-assgn2-part1.csv
for row in csvreader:
    if row:
        val = int(row[7])
        Y.append(val)
        if val == 1:
            positiveLines.append([int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), float(row[6]), 1])#bias given a value of 1
            YP.append(val)
        else:
            negativeLines.append([int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), float(row[6]), 1])#bias given a value of 1
            YN.append(val)
file.close()

# Full train set with their features
X = positiveLines
Y = YP
X = X + negativeLines
Y = Y + YN

#Extract test set lines from HW2-testset.txt and subsequently perform feature extraction
test_lines = []
Xfinal = []
IDs = []
test_lines_file = open('HW2-testset.txt',encoding= "utf8")

for i in test_lines_file:
    m = re.search("(ID-[0-9]+)", i)
    IDs.append(m.group(1))
    p = re.split('ID-[0-9]+', i)
    p = p[1].replace("!"," !")
    p=re.sub("[^A-Za-z\+\-'!]"," ",p)
    p = re.split(" ",p)
    p = [c.lower() for c in p if c!='' and c!= ' ']
    test_lines.append(p)

#Rest positive and negative words
positive_words = open('positive-words.txt',encoding= "utf8")
negative_words = open('negative-words.txt',encoding= "utf8")

#Populate positive and negative sets for matching later on
positive_w_set = set()
negative_w_set = set()

for i in positive_words:
    p=re.sub("\n","",i)
    positive_w_set.add(p)

for i in negative_words:
    p=re.sub("\n","",i)
    negative_w_set.add(p)

pronouns = set(["i", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours"])
def featureExtraction(text):
    for i in text:
        count = 0
        x1,x2,x3,x4,x5,x6 = 0,0,0,0,0,0
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
        Xfinal.append([x1,x2,x3,x4,x5,x6])

#Perform feature extraction on test set
featureExtraction(test_lines)

W = [0,0,0,0,0,0,0.1] # Initialized all features weights to zero and selected 0.1 for the bias

#Initialize numpy arrays
x, y = np.array(X, dtype="float64"), np.array(Y, dtype="float64")
w= np.array(W, dtype = "float64")

#Includes Sigmoid
def gradient(x,y,w):
    z = np.dot(x, w)
    k = 1.0/(1.0 + np.exp(-z))
    return np.multiply(k-y, x)

#Pure Sigmoid
def sigmoid(x,w,b):
    z = np.dot(x, w)
    z = z + b
    k = 1.0/(1.0 + np.exp(-z))
    return k

#Stochastic gradient descent function. Learn rate 0.01 and iterations 5000 yielded best results
def stocgd(gradient, x, y, start, learn_rate=0.01, iterations=5000, dtype="float64"):
    dtype_ = np.dtype(dtype)

    x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
    n_obs = x.shape[0]
    xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]

    rng = np.random.default_rng(seed=90)

    weights = np.array(start, dtype=dtype_)

    for _ in range(iterations):
        rng.shuffle(xy)

        for start in range(0, n_obs):
            stop = start + 1
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]
            grad = np.array(gradient(x_batch, y_batch, weights), dtype_)
            diff = -learn_rate * grad
            weights += diff[0]

    return weights

weights = stocgd(gradient, x, y, w)

def finalPrediction(Xfinal):
    lengthX = len(Xfinal)
    s = ""
    for i,x in enumerate(Xfinal):
        p = sigmoid(np.array(x[0:6],dtype="float64"), weights[0:6],weights[6])
        if round(p) == 1:
            s+= IDs[i] + "\t" + "POS"
        else:
            s+= IDs[i] + "\t" + "NEG"
        if i < lengthX - 1:
            s+= "\n"
    return s

#Run scoring on the final testing set
finalOutput = finalPrediction(Xfinal)
output = open("mallela-srinivasakhil-assgn2-out.txt","w")
output.write(finalOutput)
output.close()

