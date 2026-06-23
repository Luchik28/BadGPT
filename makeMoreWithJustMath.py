from BasicMLP import MLP, TrainMakeMoreTry2
import math
import random
words = open('names.txt', 'r').read().splitlines()

wordProb = [[0 for _ in range(27)] for _ in range(27)]

inputs = []
outputs = []

for word in words[:50]:
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

#Apply squishiness
for x in range(len(wordProb)):
    for y in range(len(wordProb[0])):
        wordProb[x][y] = wordProb[x][y]/100


#Now we have our inputs/outputs, in letters.

def wordWithMath():
    word = "."
    dontdie = 0
    while dontdie < 50:
        prevNum = ord(word[-1])-97

        if prevNum < 0 or prevNum>26:
            prevNum = 26
        
        letterChoices = [(wordProb[prevNum][i]) for i in range(26)]

        letterNum = letterChoices.index(max(letterChoices))+97
        #letterIndex = random.choices(range(27), weights=wordProb[prevNum])[0]
        if (letterNum==26):
            return word[1:]
        else:
            word+=chr(letterNum)
        dontdie +=1
    print("Length exceeded")
    return word[1:]

#print(wordWithMath())

# How to imporove (ideas):
# Well, one way could be to squish the more common letters with a log function.
# ln produces math domain error.
# tanh is too agressive, there's not enough difference between words.
# When we first divide the counts by 100, so that smaller ones are counted less aggressively,
# It gets a little better but soon drops off to being too short.
# random.choices? Doesn't really do anything.
# So I think we should go back to using an MLP like Karpathy

# Now for the not math but still kind of math:
# we have inputs and outputs, both arrays.
# We need to do One-hot encoding.

for input in range(len(inputs)):
    cur = inputs[input]
    inputs[input] = [0 for _ in range(27)]
    curIndex = ord(cur)-97
    if curIndex <0:
        curIndex=26 # Change it if it's a period
    inputs[input][curIndex] = 1

for output in range(len(outputs)):
    cur = outputs[output]
    outputs[output] = [0 for _ in range(27)]
    curIndex = ord(cur)-97
    if curIndex <0:
        curIndex=26 # Change it if it's a period
    outputs[output][curIndex] = 1

model = MLP(27, [27]) #27 inputs, 27 outputs.

print("Training:")
TrainMakeMoreTry2(model, inputs, outputs, 20, .001)
print("Done Training:")

for i in range(10):
    intro = [0 for _ in range(27)]
    intro[26] = 1
    print(model(intro))

    word = "."
    dontdie = 0
    while dontdie < 50:
        prevNum = ord(word[-1])-97

        if prevNum < 0 or prevNum>26:
            prevNum = 26
        
        letterChoices = [(wordProb[prevNum][i]) for i in range(26)]

        letterNum = letterChoices.index(max(letterChoices))+97
        #letterIndex = random.choices(range(27), weights=wordProb[prevNum])[0]
        if (letterNum==26):
            print(word[1:])
            break
        else:
            word+=chr(letterNum)
        dontdie +=1