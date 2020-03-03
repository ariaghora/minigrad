import numpy as np

class Tensor:
    '''
    Wrapper for numpy ndarray that holds gradient information.
    Tensor acts as the 'node' in a computational graph.
    '''
    def __init__(self, X, is_leaf=True, requires_grad=False):
        if not type(X) is np.array:
            self.data = np.array(X).astype(float)
        else:
            self.data = X.astype(float)
                
        self.set_requires_grad(requires_grad)
        self.shape         = (lambda: self.data.shape)() # link to self.data.shape
        self.is_leaf       = is_leaf
        self.prev          = []
        self.backward_func = None
    
    def set_requires_grad(self, val):
        self.requires_grad = val
        if val:
            self.grad = np.zeros_like(self.data)
    
    def backward(self):
        sorted_nodes = (toposort(self))
        self.grad = np.ones_like(self.data)
        for n in sorted_nodes:
            if n.requires_grad and not(n.is_leaf): 
                n.backward_func(n.grad)
    
    def zero_grad(self):
        sorted_nodes = (toposort(self))
        for n in sorted_nodes:
            n.grad = np.zeros_like(n.data)
            
    ''' operator overloading '''
    def __add__(self, other): return add(self, to_tensor(other))
    def __radd__(self, other): return self.__add__(other)
    def __sub__(self, other): return sub(self, to_tensor(other))
    def __rsub__(self, other): return self.__sub__(other)
    def __mul__(self, other):  return mul(self, to_tensor(other))
    def __rmul__(self, other): return self.__mul__(other)
    def __matmul__(self, other): return matmul(self, to_tensor(other))
    def __rmatmul__(self, other): return self.__matmul__(other)
    
    ''' string representation when the Tensor is printed '''
    def __repr__(self):
        return 'Tensor of\n' + self.data.__repr__()
    
def to_tensor(x):
    return Tensor(x) if np.isscalar(x) or isinstance(x, np.ndarray) else x

def toposort(root):
    ''' 
    Topological sort on the computational graph 
    '''
    sorted_nodes = []
    def toposort_helper(root):
        sorted_nodes.append(root)
        for p in root.prev:
            toposort_helper(p)
    toposort_helper(root)
    return sorted_nodes

def parent_requires_graph(arr):
    '''
    A helper to determine whether a parent node requires grad according
    to its children condition. The parent will require grad if at least
    one of its children requires grad
    '''
    return any(child.requires_grad for child in arr)

def unbroadcast_to(a, shape):
    '''
    Unbroadcast `a` into shape `shape`.
    '''
    if a.shape == shape: return a
    
    m = max(len(a.shape), len(shape))
    shape1 = np.zeros(m)
    shape1[:len(a.shape)] = list(reversed(a.shape))
    shape1 = list(reversed(shape1))
    
    shape2 = np.zeros(m)
    shape2[:len(shape)] = list(reversed(shape))
    shape2 = list(reversed(shape2))
    
    dims = []
    for i, s in enumerate(shape1):
        if s != shape2[i]:
            dims.append(i)
    return np.sum(a, axis=tuple(dims)).reshape(shape)

'''*************************************************************************************
Define required operators. Calling an operator will trigger the construction of 
computational graph.
*************************************************************************************'''
def add(a, b):    
    def backward_add(dy):
        if a.requires_grad: a.grad += unbroadcast_to(dy, a.shape)
        if b.requires_grad: b.grad += unbroadcast_to(dy, b.shape)
    res = Tensor(a.data + b.data, is_leaf=False, requires_grad=parent_requires_graph([a, b]))    
    res.backward_func = backward_add
    res.prev.extend([a, b])
    return res

def sub(a, b):
    def backward_sub(dy):
        if a.requires_grad: a.grad += dy
        if b.requires_grad: b.grad -= dy
    res = Tensor(a.data - b.data, is_leaf=False, requires_grad=parent_requires_graph([a, b]))    
    res.backward_func = backward_sub
    res.prev.extend([a, b])
    return res

def mul(a, b):
    def backward_mul(dy):
        if a.requires_grad: a.grad += dy * b.data
        if b.requires_grad: b.grad += dy * a.data
    res = Tensor(a.data * b.data, is_leaf=False, requires_grad=parent_requires_graph([a, b]))    
    res.backward_func = backward_mul
    res.prev.extend([a, b])
    return res

def matmul(a, b):
    def backward_matmul(dy):
        if a.requires_grad: a.grad += np.matmul(dy, b.data.T)
        if b.requires_grad: b.grad += np.matmul(a.data.T, dy)
    res = Tensor(np.matmul(a.data, b.data), is_leaf=False, requires_grad=parent_requires_graph([a, b]))    
    res.backward_func = backward_matmul
    res.prev.extend([a, b])
    return res

def mean(a, axis=None):
    def backward_mean(dy):
        if a.requires_grad:
            sz_a = a.data.size
            sz_dy = dy.size
            a.grad += np.full_like(a.data, 1.0/(sz_a/sz_dy))
    res = Tensor(np.mean(a.data, axis), is_leaf=False, requires_grad=parent_requires_graph([a]))
    res.backward_func = backward_mean
    res.prev.append(a)
    return res

def square(a):
    return mul(a, a)