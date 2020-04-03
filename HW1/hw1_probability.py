import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
# nltk.download("book")
from nltk import ngrams
from string import punctuation

# List of gutenberg corpus
# print(gutenberg.fileids())

# List of alphabet
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Alice's Adventures in Wonderland by Lewis Carroll 1865
alice_raw = nltk.corpus.gutenberg.raw("carroll-alice.txt")
alice_raw = alice_raw.lower()
print("length: " + str(len(alice_raw)))       # Count all letters in alice data


''' 1. Probability distribution over the 27 outcomes for a randomly selected letter '''
# Probability for each alphabet letter
for letter in alphabet:
    print(letter + ": " + str(alice_raw.count(letter) / len(alice_raw)) + " / " + str(alice_raw.count(letter)))

# Probability for whitespace, enter, and punctuation
# print(punctuation)
punc = ' ' + punctuation + '\n'
punc = '0123456789' + punc

count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
# print(count(alice_raw, punc))
print(' ' + ": " + str(count(alice_raw, punc) / len(alice_raw)) + " / " + str(count(alice_raw, punc)))


''' 2. Probability distribution over the 27x27 possible bigrams xy '''
alice_bigram = list(ngrams(alice_raw, 2))

# Change all punctuations to ' '
alice_bigram_list = []
for ele in alice_bigram:
    ele = list(ele)
    if ele[0] in punc:
        ele[0] = ' '
    if ele[1] in punc:
        ele[1] = ' '
    alice_bigram_list.append(tuple(ele))

# ConditionalFreqDist
cfd = nltk.ConditionalFreqDist(alice_bigram_list)
cfd.tabulate()
# print(cfd.N())

# cfd = nltk.ConditionalFreqDist([(t[0], t[1]) for t in alice_bigram])
# print(alice_bigram.count((alphabet[0], alphabet[3])))


''' 3. Conditional probability '''
#  3-1. P(y|x)
cpd = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
conditions = sorted(cpd.conditions())
cpdTable = np.zeros((27, 27))
# for item in conditions:
#     for item2 in conditions:
#         print("(" + item + ", " + item2 + ") :" + str(cpd[item].prob(item2)))

for x in range(0, 27):
    for y in range(0, 27):
        # print("(" + conditions[x] + ", " + conditions[y] + ") :" + str(cpd[conditions[x]].prob(conditions[y])))
        cpdTable[x][y] = cpd[conditions[x]].prob(conditions[y])

df = pd.DataFrame(cpdTable, index=conditions, columns=conditions)
print(df)


# 3-2. P(x|y)
cfd_yx = nltk.ConditionalFreqDist([(t[1], t[0]) for t in alice_bigram_list])
cfd_yx.tabulate()

cpd_yx = nltk.ConditionalProbDist(cfd_yx, nltk.MLEProbDist)
conditions = sorted(cpd.conditions())
cpdTable = np.zeros((27, 27))
# for item in conditions:
#     for item2 in conditions:
#         print("(" + item + ", " + item2 + ") :" + str(cpd[item].prob(item2)))

for x in range(0, 27):
    for y in range(0, 27):
        # print("(" + conditions[x] + ", " + conditions[y] + ") :" + str(cpd[conditions[x]].prob(conditions[y])))
        cpdTable[x][y] = cpd_yx[conditions[x]].prob(conditions[y])

df = pd.DataFrame(cpdTable, index=conditions, columns=conditions)
print(cpdTable[0][1])
print(df)


''' To Do '''
# 1. Probability distribution over the 27 outcomes for a randomly selected letter
print("####")
# fd = nltk.probability.FreqDist(alice_raw)
# print(fd.N())   # Count all letters in alice data
# print(fd['a'])  # Count 'a' letters in alice data

# 2. Probability distribution over the 27x27 possible bigrams xy

# 3. Conditional probability
#  3-1. P(y|x)
#  3-2. P(x|y)