"""
    Reads the data from the txt file and predict sentiment per speaker
"""


import os


from Tasker import tasker
import matplotlib.pyplot as plt
from collections import defaultdict

# build pipeline
t = tasker("sentiment")


def read_transcript():
    speaker_sentence = defaultdict(list)
    with open('output.txt', 'r') as fp:
        for _, line in enumerate(fp):
            arr = line.split("\t")
            sentences = arr[1].strip("\n")
            for sentence in sentences.split("."):
                #_, score = t(sentence)
                pass
            _, score = t(sentences)
            speaker_sentence[arr[0]].append([sentences, score])
    return speaker_sentence


output = read_transcript()
print(output)
for key, values in output.items():
    y = []
    for value in values:
        y.append(value[1])
    print(key, y)


# Testing sentiment on trascription and finding out how sentiment varies from
# start to end with sentiment score for each sentence

# score close to zero is negative  and close to one is positive
