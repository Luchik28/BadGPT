import math
import random
import numpy as np
import matplotlib.pyplot as plt

class Value:
    def __init__(self, data, _children=(), _opp="", label=""):
        self.data = np.array(data)
        self.prev = set(_children)
        self.grad = np.zeros_like(self.data, dtype=float)
        self.opp = _opp
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def _unbroadcast(grad, shape):
        # sum over axes that were broadcast so grad matches `shape`
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) #Make it still work if other is an integer and not a Value (a+1, for instance)

        def _backward():
            self.grad += Value._unbroadcast(out.grad, self.data.shape)
            other.grad += Value._unbroadcast(out.grad, other.data.shape)

        out = Value(self.data+other.data, (self, other), "+")
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        out = Value(self.data @ other.data, (self, other), "@")
        def _backward():
            # works for 2D (n,k)@(k,m) and batched (...,n,k)@(k,m) inputs
            sg = out.grad @ np.swapaxes(other.data, -1, -2)
            og = np.swapaxes(self.data, -1, -2) @ out.grad
            self.grad += Value._unbroadcast(sg, self.data.shape)
            other.grad += Value._unbroadcast(og, other.data.shape)
        
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
            self.grad += Value._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += Value._unbroadcast(self.data * out.grad, other.data.shape)

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
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # embedding-style lookup: idx is an int array of indices into axis 0
        out = Value(self.data[idx], (self, ), "getitem")

        def _backward():
            np.add.at(self.grad, idx, out.grad) # scatter-add grads back to the rows that were selected

        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Value(self.data.reshape(*shape), (self, ), "reshape")

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, dim0=-2, dim1=-1):
        # PyTorch-style: swaps two axes. NOT numpy's transpose, which wants a full permutation
        out = Value(np.swapaxes(self.data, dim0, dim1), (self, ), "transpose")

        def _backward():
            self.grad += np.swapaxes(out.grad, dim0, dim1) # swapping is its own inverse

        out._backward = _backward
        return out

    def masked_fill(self, mask, value):
        out = Value(np.where(mask, value, self.data), (self, ), "masked_fill")

        def _backward():
            self.grad += np.where(mask, 0.0, out.grad) # masked entries contribute nothing

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        e = np.exp(self.data - self.data.max(axis=axis, keepdims=True)) # subtract max for stability
        p = e / e.sum(axis=axis, keepdims=True)
        out = Value(p, (self, ), "softmax")

        def _backward():
            self.grad += p * (out.grad - (out.grad * p).sum(axis=axis, keepdims=True))

        out._backward = _backward
        return out

    @staticmethod
    def cat(values, axis=-1):
        out = Value(np.concatenate([v.data for v in values], axis=axis), tuple(values), "cat")
        sizes = [v.data.shape[axis] for v in values]

        def _backward():
            splits = np.split(out.grad, np.cumsum(sizes)[:-1], axis=axis)
            for v, g in zip(values, splits):
                v.grad += g

        out._backward = _backward
        return out

    def squeeze(self, axis=None):
        out = Value(np.squeeze(self.data, axis=axis), (self, ), "squeeze")

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out
    
    def tanh(self):
        x=self.data
        t = np.tanh(x)

        def _backward():
            self.grad += (1-t**2) * out.grad

        out = Value(t, (self, ), "tanh")
        out._backward=_backward
        return out
    
    def exp(self):
        x=self.data

        def _backward():
            self.grad += (np.exp(x)*out.data)

        out = Value(np.exp(x), (self, ), "exp")
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

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

#Helper function
def oneHot(c):
    out = [0 for _ in range(27)]
    curIndex = ord(c)-97
    if curIndex <0:
        curIndex=26 # Change it if it's a period
    out[curIndex] = 1
    return out
def cross_entropy(logits, targets):
    # logits: (N, vocab) or (B, T, vocab); targets: (N,) or (B, T) - either shape works,
    # attention-based models predict at every position so logits come in 3D
    orig_shape = logits.data.shape
    targets = np.asarray(targets)
    if logits.data.ndim == 3:
        B, T, V = orig_shape
        flat_logits = logits.data.reshape(B * T, V)
        flat_targets = targets.reshape(B * T)
    else:
        flat_logits = logits.data
        flat_targets = targets

    e = np.exp(flat_logits - flat_logits.max(axis=1, keepdims=True)) # subtract max: avoids overflow on large logits
    probs = e / e.sum(axis=1, keepdims=True)

    correct_probs = probs[np.arange(len(flat_targets)), flat_targets]
    loss = -np.log(correct_probs).mean()

    out = Value(loss, (logits,), "cross_entropy")

    def _backward():
        # gradient of cross entropy + softmax combined is just (probs - 1_correct) / batch_size
        dlogits = probs.copy()
        dlogits[np.arange(len(flat_targets)), flat_targets] -= 1
        dlogits /= len(flat_targets)
        logits.grad += dlogits.reshape(orig_shape)

    out._backward = _backward
    return out
def prog(val, total):
    length = 20
    out = "["
    for i in range(length):
        if i < math.ceil((val/total) * length):
            out += "#"
        else:
            out += "-"
    return out + "]"
def trunc(number, digits):
    stepper = 10 ** digits
    return math.trunc(number * stepper) / stepper

#Different "building blocks of model"
class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = Value(np.random.randn(fan_in, fan_out) / fan_in**0.5)
        self.bias = Value(np.zeros(fan_out)) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1D:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained via backprop)
        self.gamma = Value(np.ones(dim))
        self.beta = Value(np.zeros(dim))
        # buffers (updated with a running average, not backprop)
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)

    def __call__(self, x):
        # reduce over the batch dim for 2D (B, C) input and over (batch, time) for 3D (B, T, C)
        dim = 0 if x.data.ndim == 2 else (0, 1)
        # calculate the forward pass
        if self.training:
            xmean = np.mean(x.data, axis=dim, keepdims=True) # batch mean
            xvar = np.var(x.data, axis=dim, keepdims=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / (xvar + self.eps) ** .5 # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class LayerNorm:
    """Normalizes each token's own feature vector, independent of the batch.

    Unlike BatchNorm1D, this needs no running stats and no train/eval switch -
    every token is normalized the same way whether it's alone (batch size 1,
    e.g. during generation) or in a big batch.
    """

    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = Value(np.ones(dim))
        self.beta = Value(np.zeros(dim))

    def __call__(self, x):
        N = x.data.shape[-1]
        mu = x.data.mean(axis=-1, keepdims=True)
        xmu = x.data - mu
        var = (xmu ** 2).mean(axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        xhat = xmu * std_inv

        out = Value(self.gamma.data * xhat + self.beta.data, (x, self.gamma, self.beta), "layernorm")

        def _backward():
            dy = out.grad
            self.gamma.grad += Value._unbroadcast(dy * xhat, self.gamma.data.shape)
            self.beta.grad += Value._unbroadcast(dy, self.beta.data.shape)

            # exact layernorm backward: mu and var both depend on x, unlike a plain (x-mean)/std
            # composed from ops that treat mean/var as constants (that shortcut undercounts the
            # gradient - it's what BatchNorm1D above does, and it doesn't hold up under gradcheck)
            dxhat = dy * self.gamma.data
            dvar = np.sum(dxhat * xmu * -0.5 * std_inv ** 3, axis=-1, keepdims=True)
            dmu = np.sum(dxhat * -std_inv, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * xmu, axis=-1, keepdims=True)
            x.grad += dxhat * std_inv + dvar * 2.0 * xmu / N + dmu / N

        out._backward = _backward
        return out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = x.tanh()
        return self.out
    def parameters(self):
        return []
    
class Embedding:
  
  def __init__(self, num_embeddings, embedding_dim):
    rng = np.random
    self.weight = Value(rng.standard_normal((num_embeddings, embedding_dim)))
    
  def __call__(self, IX):
    self.out = self.weight[IX] # Value.__getitem__ keeps this in the autograd graph
    return self.out
  
  def parameters(self):
    return [self.weight]

class PositionalEmbedding:
  def __init__(self, block_size, embedding_dim):
    rng = np.random
    self.weight = Value(rng.standard_normal((block_size, embedding_dim)))

  def __call__(self, x):
    B, T, C = x.shape
    self.out = x + self.weight[:T] # (T,C) broadcasts over the batch dim
    return self.out

  def parameters(self):
    return [self.weight]

class FlattenConsecutive:
  
  def __init__(self, n):
    self.n = n
    
  def __call__(self, x):
    B, T, C = x.shape
    x = x.reshape(B, T//self.n, C*self.n)
    if x.shape[1] == 1:
      x = x.squeeze(1)
    self.out = x
    return self.out
  
  def parameters(self):
    return []

class Sequential:
  
  def __init__(self, layers):
    self.layers = layers
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    self.out = x
    return self.out
  
  def parameters(self):
    # get parameters of all layers and stretch them out into one list
    return [p for layer in self.layers for p in layer.parameters()]

class BagOfWords:
  """Averages each token's embedding with all the ones before it (causal mean)."""

  def __init__(self, block_size):
    wei = np.tril(np.ones((block_size, block_size)))
    self.wei = Value(wei / wei.sum(1, keepdims=True)) # rows sum to 1

  def __call__(self, x):
    B, T, C = x.shape
    self.out = self.wei[:T, :T] @ x if T != self.wei.shape[0] else self.wei @ x
    return self.out

  def parameters(self):
    return [] # wei is a fixed constant, not learned

class Head:
  """One head of causal self-attention."""

  def __init__(self, n_embd, head_size, block_size):
    self.key   = Linear(n_embd, head_size, bias=False)
    self.query = Linear(n_embd, head_size, bias=False)
    self.value = Linear(n_embd, head_size, bias=False)
    self.head_size = head_size
    self.mask = np.triu(np.ones((block_size, block_size), dtype=bool), k=1) # True = future

  def __call__(self, x):
    B, T, C = x.shape
    k = self.key(x)   # (B,T,hs) what each token offers
    q = self.query(x) # (B,T,hs) what each token is looking for
    v = self.value(x) # (B,T,hs) what each token actually passes along

    wei = (q @ k.transpose(-2, -1)) * self.head_size**-0.5 # (B,T,T) affinities
    wei = wei.masked_fill(self.mask[:T, :T], -np.inf)      # no peeking ahead
    wei = wei.softmax(axis=-1)                             # rows become a distribution

    self.out = wei @ v
    return self.out

  def parameters(self):
    return self.key.parameters() + self.query.parameters() + self.value.parameters()

class MultiHead:
  """Runs several attention heads in parallel, then combines their outputs.

  Each head asks its own version of "what's relevant to me" - one might learn
  to track the previous vowel, another the subject of the sentence. Splitting
  n_embd across heads (rather than giving every head the full width) keeps the
  total compute roughly the same as a single big head.
  """

  def __init__(self, n_embd, num_heads, block_size):
    assert n_embd % num_heads == 0, "n_embd must divide evenly across heads"
    head_size = n_embd // num_heads
    self.heads = [Head(n_embd, head_size, block_size) for _ in range(num_heads)]
    self.proj = Linear(n_embd, n_embd)

  def __call__(self, x):
    self.out = self.proj(Value.cat([h(x) for h in self.heads], axis=-1))
    return self.out

  def parameters(self):
    return [p for h in self.heads for p in h.parameters()] + self.proj.parameters()

class FeedForward:
  """Per-token MLP: attention only moves information between tokens, this is
  where the model actually computes something with what it gathered."""

  def __init__(self, n_embd, expansion=4):
    self.net = Sequential([
      Linear(n_embd, expansion * n_embd),
      Tanh(),
      Linear(expansion * n_embd, n_embd),
    ])

  def __call__(self, x):
    self.out = self.net(x)
    return self.out

  def parameters(self):
    return self.net.parameters()

class Block:
  """One transformer block: communicate (attention), then compute (feedforward),
  each wrapped in a residual connection so gradients have a direct path back
  through however many blocks get stacked."""

  def __init__(self, n_embd, num_heads, block_size):
    self.ln1 = LayerNorm(n_embd)
    self.attn = MultiHead(n_embd, num_heads, block_size)
    self.ln2 = LayerNorm(n_embd)
    self.ff = FeedForward(n_embd)

  def __call__(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.ff(self.ln2(x))
    self.out = x
    return self.out

  def parameters(self):
    return self.ln1.parameters() + self.attn.parameters() + self.ln2.parameters() + self.ff.parameters()
