import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk import ngrams
from string import punctuation


# Draw plot function
def drawPlot(propTable, xTickLabel, yTickLabel, title, size_scale, saveName):
    fig, ax = plt.subplots()
    x_size = np.size(propTable, 0)
    y_size = np.size(propTable, 1)

    x, y = np.meshgrid(np.arange(x_size), np.arange(y_size))

    im = ax.scatter(
        x=y,
        y=x,
        s=propTable.transpose() * size_scale,
        c=propTable.transpose(),
        cmap='RdYlGn',
        marker='s'
    )

    ax.set(xticks=np.arange(y_size), yticks=np.arange(x_size),
           xticklabels=yTickLabel, yticklabels=xTickLabel)
    ax.set_xticks(np.arange(y_size + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(x_size + 1) - 0.5, minor=True)
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.grid(which='minor')
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-0.5, (y_size-1) + 0.5])
    ax.set_ylim([-0.5, (x_size-1) + 0.5])
    ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_title(title, loc='center')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(r".\\result\\" + saveName + '.png', dpi=300, bbox_inches='tight')
    plt.show()


# Count function for probability of punctuation
countFunc = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))


''' Prepare the data'''
# nltk.download("book")

# List of gutenberg corpus
# print(gutenberg.fileids())

# Alice's Adventures in Wonderland by Lewis Carroll 1865
alice_raw = nltk.corpus.gutenberg.raw("carroll-alice.txt")
alice_raw = alice_raw.lower()
print("length of the book 'Alice's Adventures': " + str(len(alice_raw)))       # Count all letters in alice data

# List of alphabet letters
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# String of punctuation and numbers
# 0123456789 !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
punc = ' ' + punctuation + '\n'
punc = '0123456789' + punc

# Conditions (rows and columns for bigram)
conditions = alphabet.copy()
conditions.append('-')


''' 1. Probability distribution over the 27 outcomes for a randomly selected letter '''
print("### Probability distribution over the 27 outcomes for a randomly selected letter ###")

# Compute probability for bigrams (Marginal probability)
# For the letter X in alphabets and punctuations, compute P(X=x)

# Probabilities for each alphabet letter
pd_letter_table = np.zeros((27, 1))
alphabetCount = 0
for letter in alphabet:
    letterProp = round(alice_raw.count(letter) / len(alice_raw), 5)
    pd_letter_table[alphabetCount] = letterProp
    alphabetCount = alphabetCount + 1
    # print(letter + ": " + str(letterPropTable) + " / " + str(alice_raw.count(letter)))

# Probability for numbers, whitespace, enter, and punctuation
punc_prop = round(countFunc(alice_raw, punc) / len(alice_raw), 5)
pd_letter_table[alphabetCount] = punc_prop
# print(' ' + ": " + str(punc_prop) + " / " + str(countFunc(alice_raw, punc)))

# Dataframe of probability distribution P(X=x) for a randomly selected letter
letters_df = pd.DataFrame(pd_letter_table, index=conditions, columns=['Prob'])
print(letters_df)       # print the probability distribution

# Save the probability distribution data
letters_df.to_csv(r'C:\Users\Leki\Desktop\Dooo\GCT561_ScientificThinking\HW1\result\df_letters.csv')

# Draw and save the plot for the probability distribution
drawPlot(pd_letter_table, conditions, ["Prob"],
         "Probability distribution P(X=x)\n for randomly selected letter", 110,
         "fig_letters")


print()


''' 2. Probability distribution over the 27x27 possible bigrams for ordered pair xy '''
print("### Probability distribution over the 27x27 possible bigrams for ordered pair xy ###")

# Compute bigrams of alice data
alice_bigram = list(ngrams(alice_raw, 2))

# Change all punctuations to ' ' in bigram
alice_bigram_list = []
for ele in alice_bigram:
    ele = list(ele)
    if ele[0] in punc:
        ele[0] = '-'
    if ele[1] in punc:
        ele[1] = '-'
    alice_bigram_list.append(tuple(ele))

# Compute frequencies for bigrams
cfd = nltk.ConditionalFreqDist(alice_bigram_list)
# cfd.tabulate()        # Show frequencies

# Compute probability for bigrams
# For ordered pair XY, compute P(X=x, Y=y)
pd_bigram_table = np.zeros((27, 27))
for x in range(0, 27):
    for y in range(0, 27):
        pd_bigram_table[x][y] = cfd[conditions[x]][conditions[y]] / cfd.N()

# Dataframe of probability distribution P(X=x, Y=y) for ordered pair XY
bigram_df = pd.DataFrame(pd_bigram_table, index=conditions, columns=conditions)
print(bigram_df)       # print the probability distribution

# Save the probability distribution data
bigram_df.to_csv(r'C:\Users\Leki\Desktop\Dooo\GCT561_ScientificThinking\HW1\result\df_bigram.csv')

# Draw and save the plot for the probability distribution
drawPlot(pd_bigram_table, conditions, conditions,
         "Probability distribution P(X=x, Y=y) for ordered pair XY", 1000,
         "fig_bigram")


print()


''' 3. Conditional probability distributions over P(y|x) and P(x|y) '''
print("### Conditional probability distributions over P(y|x) and P(x|y) ###")
#  3-1. P(y|x)
# Compute conditional probability distributions for ordered pair XY
# For ordered pair XY, compute P(Y=y|X=x) = P(X=x, Y=y) / P(X=x)
cpd = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)

cpd_yx_table = np.zeros((27, 27))
for x in range(0, 27):
    for y in range(0, 27):
        # print("(" + conditions[x] + ", " + conditions[y] + ") :" + str(cpd[conditions[x]].prob(conditions[y])))
        cpd_yx_table[x][y] = cpd[conditions[x]].prob(conditions[y])

# Dataframe of probability distribution P(Y=y|X=x) for ordered pair XY
cpd_yx_df = pd.DataFrame(cpd_yx_table, index=conditions, columns=conditions)

print("* P(y|x) table")
print(cpd_yx_df)       # print the probability distribution

# Save the conditional probability distribution data
cpd_yx_df.to_csv(r'C:\Users\Leki\Desktop\Dooo\GCT561_ScientificThinking\HW1\result\df_cpd_yx.csv')

# Draw and save the plot for the probability distribution
drawPlot(cpd_yx_table, conditions, conditions,
         "Conditional probability distribution P(Y=y|X=x) for ordered pair XY", 50,
         "fig_cpd_yx")


print()

# 3-2. P(x|y)
# For ordered pair XY, compute P(X=x|Y=y) = P(X=x, Y=y) / P(Y=y)
cpd_xy_table = np.zeros((27, 27))
for x in range(0, 27):
    for y in range(0, 27):
        if (y == 26):        # punctuation
            cpd_xy_table[x][y] = cfd[conditions[x]][conditions[y]] / countFunc(alice_raw, punc)
        else:               # alphabet
            cpd_xy_table[x][y] = cfd[conditions[x]][conditions[y]] / alice_raw.count(alphabet[y])

# Dataframe of probability distribution P(X=x|Y=y) for ordered pair XY
cpd_xy_df = pd.DataFrame(cpd_xy_table, index=conditions, columns=conditions)

print("* P(x|y) table")
print(cpd_xy_df)       # print the probability distribution

# Save the conditional probability distribution data
cpd_xy_df.to_csv(r'C:\Users\Leki\Desktop\Dooo\GCT561_ScientificThinking\HW1\result\df_cpd_xy.csv')

# Draw and save the plot for the probability distribution
drawPlot(cpd_xy_table, conditions, conditions,
         "Conditional probability distribution P(X=x|Y=y) for ordered pair XY", 50,
         "fig_cpd_xy")
