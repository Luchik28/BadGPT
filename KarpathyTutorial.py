import math
import numpy as np
import random

#From Andrej Karpathy's vid on Micrograd

def f(x):
    return 3**x - 4*x + 5

#print(f(3.0))

xs = np.arange(0.5, 5.0, 0.25)
ys = f(xs)
#print(ys)

# Derivative
h = 0.000000001 #h would be 0 if we were actually calculating it, bc the formula is the limit as h aproached 0. But if we go too small we run into errors with floating point arithamatic and whatnot.
x = 3.0
#print((f(x+h) - f(x)) /h)
# outputs 25.6 for some reason.
#It's basically like, if we bump the formula by h, how much does it change? But h needs to be small cuz the function in not straight. If it was linear, it would jst be rise/run


#more complicated derivative with inputs.

#inputs
#a = 2.0
#b = -3.0
#c = 10.0

#output
#d1 = a*b+c

#shift a by h
#d2 = (a+h)*b+c
#derivative of a?
#da = (d2-d1)/h

#looking
#print(da)

#Value class

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
            self.grad = other * (self.data ** (other-1)) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x=self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)

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

#basic neuron

#inputs
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
#weights
w1= Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
#bias
b = Value(6.7, label="b")
#Apply weights to inputs
x1w1 = x1*w1; x1w1.label = "x1w1"
x2w2 = x2*w2; x2w2.label = "x2w2"
#Combine inputs
x1w1x2w2 = x1w1+x2w2; x1w1x2w2.label = "x1w1x2w2"
#Apply bias to combined inputs
n = x1w1x2w2 + b; n.label="n"
#Apply func that scales it from 0 to 1
o=n.tanh(); o.label="o"
#print(o)

#Now for the backpropogation
o.backward()
#sprint(x1.grad, w1.grad, x2.grad, w2.grad)

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

#current predictions
ypred = [n(x) for x in xs]

#loss function
loss = sum([(yout-ygt)**2 for ygt, yout in zip(ys, ypred)])
print("loss before training: " + str(loss))


#training
lowest=Value(300.0)
best = []
for i in range(900): #the number of training cycles

    loss.backward() #get gradient
    for p in n.parameters():
        p.data += -.001 * p.grad #move value in direction of gradient

    ypred = [n(x) for x in xs] #get predictions
    loss = sum([(yout-ygt)**2 for ygt, yout in zip(ys, ypred)]) #get loss
    print(f"loss after training (step {i}): " + str(loss.data)) 

    if (loss.data) <= (lowest.data): #check
        print(f"new winner {loss}")
        lowest.data = loss.data
        best = ypred[:]
    else:
        print(f"no winner, {loss.data}")

print("**********")
print(f"final prediction: {best} with loss {lowest.data}")
