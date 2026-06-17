import math
import random
#From Andrej Karpathy's vid on Micrograd

class Value:
    def __init__(self, data, _children=(), _opp="", label=""):
        self.data = data
        self.prev = set(_children)
        self.grad = 0
        self.opp = _opp
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) #Make it still work if other is an integer and not a Value (a+1, for instance)
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out = Value(self.data+other.data, (self, other), "+")
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out = Value(self.data * other.data, (self, other), "*")
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
        #Add support for integer * Value (since 2*a doesn't work, but a*2 does)
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "not an int or float"
        out = Value(self.data**other, (self, ), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x=self.data
        t = math.tanh(x)

        def _backward():
            self.grad += (1-t**2) * out.grad

        out = Value(t, (self, ), "tanh")
        out._backward=_backward
        return out
    
    def exp(self):
        x=self.data

        def _backward():
            self.grad += (math.exp(x)*out.data)

        out = Value(math.exp(x), (self, ), "exp")
        out._backward=_backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

#Ok now for a Nueral Network!

class Nueron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)] #create weights for every input
        self.b = Value(random.uniform(-1,1)) #create a bias for the whole nueron
    
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) #iterate over and pair each weight with an input, then create a sum of those weighted inputs, then add the bias
        return act.tanh() #introduce non-linearity with a tanh
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.nuerons = [Nueron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.nuerons]
        return outs[0] if len(outs)==1 else outs
    
    def parameters(self):
        params = []
        for n in self.nuerons:
            params.extend(n.parameters())
        return params

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params

x = [2.0, 3.0, -1.0]
n = MLP(3, [4,4,1])
#print(n(x))


#testing!!! (so exciting :))

#set of 4 inputs
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

#desired output, for each of the inputs
ys = [1.0, -1,0, 1.0, -1.0]

#Old training func

# #current predictions
# ypred = [n(x) for x in xs]

# #loss function
# loss = sum([(yout-ygt)**2 for ygt, yout in zip(ys, ypred)])
# print("loss before training: " + str(loss))


# #training
# lowest=Value(300.0)
# best = []
# for i in range(900): #the number of training cycles

#     loss.backward() #get gradient
#     for p in n.parameters():
#         p.data += -.001 * p.grad #move value in direction of gradient

#     ypred = [n(x) for x in xs] #get predictions
#     loss = sum([(yout-ygt)**2 for ygt, yout in zip(ys, ypred)]) #get loss
#     print(f"loss after training (step {i}): " + str(loss.data)) 

#     if (loss.data) <= (lowest.data): #check
#         print(f"new winner {loss}")
#         lowest.data = loss.data
#         best = ypred[:]
#     else:
#         print(f"no winner, {loss.data}")

# print("**********")
# print(f"final prediction: {best} with loss {lowest.data}")


#exportable train function:
def Train(mlp, data, expectedResult, trainingcycles, step):
    ypred = [mlp(x) for x in data]
    loss = sum([(yout-ygt)**2 for ygt, yout in zip(expectedResult, ypred)])
    print("loss before training: " + str(loss.data))

    lowest=Value(1000000.0) #really big number
    best = []
    for i in range(trainingcycles): #the number of training cycles
        loss.backward() #get gradient
        for p in mlp.parameters():
            p.data += -step * p.grad #move value in direction of gradient (it's negative bc we want to move the loss down)
            p.grad = 0 #zero the grad so it doesn't accumilate.

        ypred = [mlp(x) for x in data] #get predictions
        loss = sum([(yout-ygt)**2 for ygt, yout in zip(expectedResult, ypred)]) #get loss
        print(f"loss after training (step {i}): " + str(loss.data)) 

        if (loss.data) <= (lowest.data): #check
            lowest.data = loss.data
            best = ypred[:]

    print("*****************************")
    print(f"final prediction loss is {lowest.data}, prediction is {best}. Model weights saved.")

#Train(n, xs, ys, 50, .01)

def TrainMakeMore(mlp, data, expectedResult, trainingcycles, step):
    ypred = [mlp(x) for x in data]

    #yout is list ([0.1, 1.0, -.05, etc])
    #ygt is list  ([-1.0, -1.0, 1.0, -1.0, etc])
    #We just add the sum of each value subtracted.
    loss = sum([(sum([act - exp for exp, act in zip(ygt, yout)]))**2 for ygt, yout in zip(expectedResult, ypred)])
    print("loss before training: " + str(loss.data))

    lowest=Value(1000000000.0) #really big number
    best = []

    for i in range(trainingcycles): #the number of training cycles
        loss.backward() #get gradient
        for p in mlp.parameters():
            p.data += -step * p.grad #move value in direction of gradient (it's negative bc we want to move the loss down)
            p.grad = 0 #zero the grad so it doesn't accumilate.

        ypred = [mlp(x) for x in data] #get predictions
        loss = sum([(sum([act - exp for exp, act in zip(ygt, yout)]))**2 for ygt, yout in zip(expectedResult, ypred)]) #get loss
        print(f"loss after training (step {i}): " + str(loss.data)) 

        if (loss.data) <= (lowest.data): #check
            lowest.data = loss.data
            best = ypred[:]

    print("*****************************")
    print(f"final prediction loss is {lowest.data}, prediction is {best}. Model weights saved.")