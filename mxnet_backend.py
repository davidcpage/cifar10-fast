import numpy as np
import mxnet
from mxnet import gluon
from mxnet.gluon import nn
import utils
from dataclasses import asdict

device = mxnet.context.gpu()

#####################
## data loading
#####################


@utils.transfer.register(mxnet.nd.NDArray)
def _(data, device):
    return data.as_in_context(device)

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.set_random_choices = set_random_choices
        self.dataloader = gluon.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, 
            last_batch=('discard' if drop_last else 'keep')
        )
    
    def __iter__(self):
        if self.set_random_choices: self.dataloader._dataset.set_random_choices()
        return ({'input': utils.transfer(x, device).astype(np.float16), 'target': utils.transfer(y, device).astype(np.int32)} for (x,y) in self.dataloader)
    
    def __len__(self): 
        return len(self.dataloader)
    
class Correct(gluon.HybridBlock):
    def hybrid_forward(self, F, classifier, target):
        return F.argmax(classifier, axis=1).astype(np.int32) == target

class Concat(gluon.HybridBlock):
    def __init__(self, dim=1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def hybrid_forward(self, F, *xs):
        return F.concat(*xs, dim=self.dim)

def from_numpy(x, ctx=device):
    return mxnet.nd.array(x, ctx=ctx, dtype=x.dtype)

def remap_args(func, args_map): 
    return lambda **kw: func(**{args_map.get(k, k): v for (k,v) in kw.items()})

class SplitBatchNorm(nn.BatchNorm):
    def __init__(self, num_chunks, **kwargs):
        nn.BatchNorm.__init__(self, **kwargs)
        self.num_chunks = num_chunks
        
    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        ys = [F.BatchNorm(xx, gamma, beta, running_mean, running_var,
                           name='fwd', **self._kwargs) for xx in F.split(x, self.num_chunks, axis=0)]
        return F.concat(*ys, dim=0)  


def bn(momentum, eps, affine, num_features, num_chunks, **kw):
    if num_chunks > 1:
        return SplitBatchNorm(momentum=1-momentum, epsilon=eps, center=affine, scale=affine, 
                              in_channels=num_features, num_chunks=num_chunks)
    else:
        return nn.BatchNorm(
        momentum=1-momentum, epsilon=eps, center=affine, scale=affine, in_channels=num_features)

def convert_to_mxnet(layer):
    x_ent = lambda weight, reduction, **kw: gluon.loss.SoftmaxCrossEntropyLoss(
        weight=weight)
    type_map = {
        utils.Conv2d: remap_args(nn.Conv2D, {'out_channels': 'channels', 'stride': 
                    'strides', 'bias': 'use_bias'}),
        utils.Linear: remap_args(nn.Dense, {'in_features': 'in_units', 'out_features': 'units',
                    'bias': 'use_bias'}),
        utils.BatchNorm2d: bn,
        utils.MaxPool2d: remap_args(nn.MaxPool2D, {'kernel_size': 'pool_size', 'stride': 'strides'}),
        utils.CrossEntropyLoss: x_ent,
        utils.ReLU: lambda **kw: nn.Activation('relu'),
        utils.Flatten: nn.Flatten,
        utils.Correct: Correct,
        utils.Concat: Concat,
    }
    if type(layer) in type_map:
        return type_map[type(layer)](**asdict(layer))
    return layer


class Network(gluon.HybridBlock):  
    def __init__(self, net):
        super().__init__()
        self.graph = utils.build_graph(net)
        self.cache = None
        for n, (v, _) in self.graph.items(): 
            setattr(self, n, convert_to_mxnet(v))

    def hybrid_forward(self, F, x):
        cache = {'input': x}
        for n, (_, i) in self.graph.items():
            cache[n] = getattr(self, n)(*[cache[x] for x in i])
            if n == 'classifier': break
        return cache['classifier']
    
    def __call__(self, inputs):
        output_nodes = ['loss', 'correct']
        cache = inputs
        cache['classifier'] = self.forward(cache['input'])
        for node in output_nodes:
            (_, i) = self.graph[node]
            cache[node] = getattr(self, node)(*[cache[x] for x in i])
        return cache       
    
    def train(self, mode):
        mxnet.autograd.set_training(mode)

    def zero_grad(self):
        pass #pytorch API compat
    
    def load_weights(self, weights, device=device):
        for k, wt in weights.items():
            node_name, param_name = k.split('.')
            node = getattr(self, node_name)
            if isinstance(node, nn.BatchNorm):
                param_name = {'weight': 'gamma', 'bias': 'beta'}.get(param_name, param_name)
            getattr(node, param_name)._load_init(from_numpy(wt), ctx=device)  
        return self
    
    def half(self):
        self.cast('float16')
        return self
    
    def trainable_params(self):
        return [param for (name, param) in self.collect_params().items() if param.grad_req != 'null']

    
@utils.grad_data.register(gluon.parameter.Parameter)
def _(param):
    return param.grad()

@utils.weight_data.register(gluon.parameter.Parameter)
def _(param):
    return param.data()

@utils.forward.register(Network)
def _(model, batch, recording=False):
    with mxnet.autograd.record(recording):
        return model(batch)

@utils.to_numpy.register(mxnet.nd.NDArray)
def _(x): 
    return x.asnumpy() 

@utils.add_.register(mxnet.nd.NDArray)
def _(x , a, y):
    if a is 0: return
    if a is 1: x[:] += y
    else: x[:] += a*y

@utils.mul_.register(mxnet.nd.NDArray)
def _(x, y):
    x[:] *=y

@utils.zeros_like.register(mxnet.nd.NDArray)
def _(x):
    return mxnet.nd.zeros(x.shape, x.context, dtype=x.dtype)
