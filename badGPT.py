from BasicMLP import Value, Nueron, Layer, MLP, Train

model = MLP(3, [4,4,1])

#set of 4 inputs
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

#desired output, for each of the inputs
ys = [1.0, -1,0, 1.0, -1.0]

#train and test the model!
Train(model, xs, ys, 50, .01)