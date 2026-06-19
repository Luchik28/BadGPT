# Very similar to how I did it.
# he finds the probability of each letter coming after each next letter.
# He builds a graph where half of them are starting letters.
# And then he uses PROBABILILTY to get it to actually work.

from BasicMLP import MLP, TrainMakeMore
import math
import random
words = open('names.txt', 'r').read().splitlines()

wordProb = [[0 for _ in range(27)] for _ in range(27)]

inputs = []
outputs = []

for word in words:
    chs = ['.'] + list(word) + ['.'] # . will be our start/end character.
    for ch1, ch2 in zip(chs, chs[1:]):
        inputs.append(ch1)
        outputs.append(ch2)
        #X is input Y is output. The last column/row is our special start/end character (.)
        ind1 = ord(ch1)-97
        ind2 = ord(ch2)-97
        if ind1 < 0:
            ind1 = 26
        if ind2 < 0:
            ind2 = 26
        wordProb[ind1][ind2] += 1 #Add one to our total count!

#print(wordProb)

#Now we have our inputs/outputs, in letters.

# while(1):
#     c1 = input("enter first letter")
#     c2 = input("enter second letter")
#     ind1 = ord(c1)-97
#     ind2 = ord(c2)-97
#     if ind1 < 0:
#         ind1 = 26
#     if ind2 < 0:
#         ind2 = 26
#     print(wordProb[ind1][ind2])

# Now, generate a word using just probabilities.

def wordWithMath():
    word = "."
    dontdie = 0
    while dontdie < 50:
        prevNum = ord(word[-1])-97
        if prevNum < 0:
            prevNum = 26
        letterChoices = [(wordProb[prevNum][i] * random.random()) for i in range(27)]
        letterNum = letterChoices.index(max(letterChoices))+97
        if (letterNum >= 97+26):
            return word[1:]
        else:
            word+=chr(letterNum)
        dontdie +=1
    return word[1:]


for i in range(10):
    print(wordWithMath())











