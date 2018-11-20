import os
import os.path
import pickle
import hashlib
import urllib.request
import tarfile
from tqdm import tqdm
from inspect import signature
from collections import namedtuple
import time
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Any
from functools import singledispatch

#####################
# utils
#####################

class Timer():
    def __init__(self):
        self.times = [time.time()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.times.append(time.time())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t

localtime = lambda: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


class TableLogger():
    def append(self, output):
        if not hasattr(self, 'keys'):
            self.keys = output.keys()
            print(*(f'{k:>12s}' for k in self.keys))
        filtered = [output[k] for k in self.keys]
        print(*(f'{v:12.4f}' if isinstance(v, np.float) else f'{v:12}' for v in filtered))


#####################
## dataset download (no torchvision import)
#####################
def compute_md5(fpath):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            md5.update(chunk)
    return md5.hexdigest()

def download_and_extract(url, root, filename, md5):
    fpath = os.path.join(root, filename)
    if os.path.isfile(fpath) and (md5 == compute_md5(fpath)):
        print(f'Using already downloaded file: {fpath}')
    else:
        print(f'Downloading {url} to {fpath}')
        pbar = tqdm(unit='B', unit_scale=True)
        def bar_updater(count, block_size, total_size):
            if pbar.total is None and total_size: 
                pbar.total = total_size 
            pbar.update(count * block_size - pbar.n)
        urllib.request.urlretrieve(
            url, fpath,
            reporthook=bar_updater
        )
    with tarfile.open(fpath, "r:gz") as tar:
        tar.extractall(path=root)
        
def unpickle(fpath, md5=None):
    assert (os.path.isfile(fpath) and (md5 == compute_md5(fpath)))
    with open(fpath, 'rb') as f:
        return pickle.load(f, encoding='latin1')
    
def get_cifar10(root='./data'):
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    tar_filename = ('cifar-10-python.tar.gz', 'c58f30108f718f92721af3b95e74349a')
    data_path = 'cifar-10-batches-py'
    data_filenames = {
        'train': [
            ('data_batch_1', 'c99cafc152244af753f735de768cd75f'),
            ('data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'),
            ('data_batch_3', '54ebc095f3ab1f0389bbae665268c751'),
            ('data_batch_4', '634d18415352ddfa80567beed471001a'),
            ('data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'),
        ],
        'test': [
            ('test_batch',   '40351d587109b95175f43aff81a1287e'),
        ]
    }
    os.makedirs(root, exist_ok=True)
    download_and_extract(url, root, *tar_filename)
    raw_data = {group: {filename: unpickle(os.path.join(root, data_path, filename), md5) 
                for (filename, md5) in filenames} 
                for (group, filenames) in data_filenames.items()}
    data = {group: np.vstack([d['data'] for d in batches.values()]).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1)) 
                for group,batches in raw_data.items()} 
    targets = {group: sum((d['labels'] for d in batches.values()),[]) 
                for group, batches in raw_data.items()}
    return data, targets

#####################
## data preprocessing
#####################

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

#####################
## data augmentation
#####################

class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:,y0:y0+self.h,x0:x0+self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)}
    
    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)
    
class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x 
        
    def options(self, x_shape):
        return {'choice': [True, False]}

class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:,y0:y0+self.h,x0:x0+self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)} 
    
    
class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None
        
    def __len__(self):
        return len(self.dataset)
           
    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k,v) in choices.items()}
            data = f(data, **args)
        return data, labels
    
    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k:np.random.choice(v, size=N) for (k,v) in options.items()})


#####################
## backend compat
#####################
@singledispatch
def to_numpy(x):
    raise NotImplementedError

#####################
## layer types
#####################
_pair = lambda x: (x,x) if isinstance(x, int) else tuple(x)

@dataclass
class Conv2d():
    in_channels: str
    out_channels: int
    kernel_size: Any
    stride: Any = 1
    padding: Any = 0
    dilation: Any = 1
    groups: int = 1
    bias: bool = True

    def __post_init__(self):
        self.kernel_size = _pair(self.kernel_size)
        self.stride = _pair(self.stride)
        self.padding = _pair(self.padding)
        self.dilation = _pair(self.dilation)
        
@dataclass
class BatchNorm2d():
    num_features: int
    eps: float = 1e-05
    momentum: float = 0.1
    affine: bool = True

@dataclass
class MaxPool2d():
    kernel_size: Any
    stride: Any = None
    padding: Any = 0

    def __post_init__(self):
        self.kernel_size = _pair(self.kernel_size)
        self.stride = self.kernel_size if (self.stride is None) else _pair(self.stride)

@dataclass
class Linear():
    in_features: int
    out_features: int
    bias: bool = True

@dataclass
class CrossEntropyLoss():
    weight: Any = None
    reduction: str ='elementwise_mean'

@dataclass
class Identity():
    def __call__(self, x): return x

@dataclass
class ReLU():
    inplace: bool = True

@dataclass
class Mul():
    weight: float
    def __call__(self, x): return x*self.weight

@dataclass        
class Flatten():
    def __call__(self, x): return x.view(x.size(0), x.size(1))

@dataclass
class Add():
    def __call__(self, x, y): return x + y

@dataclass
class Concat():
    dim: int = 1

@dataclass
class Correct():
    def __call__(self, classifier, target):
        return classifier.max(dim = 1)[1] == target

#####################
## weights init
#####################

def conv_init(layer):
    c_in, c_out, kh, kw = layer.in_channels, layer.out_channels, *layer.kernel_size
    u = 1. / np.sqrt(c_in*kh*kw)
    res = {'weight': np.random.uniform(-u, u, size=(c_out, c_in, kh, kw))}
    if layer.bias:
        res['bias'] = np.full(c_out, 0.0)
    return res

def bn_init(layer):
    n = layer.num_features
    return {'weight': np.full(n, 1.), 'bias': np.full(n, 0.), 'running_mean': np.full(n, 0.), 'running_var': np.full(n, 1.)}   

def linear_init(layer):
    u = 1. / np.sqrt(layer.in_features)
    res = {'weight': np.random.uniform(-u, u, size=(layer.out_features, layer.in_features))}
    if layer.bias:
        res['bias'] = np.random.uniform(-u, u, size=layer.in_features)
    return res

def initial_weights(net, init_funcs=((Conv2d, conv_init), (BatchNorm2d, bn_init), (Linear, linear_init))):
    init_funcs = dict(init_funcs)
    return {k+'.'+p: v.astype(np.float32) for k, (layer, _) in build_graph(net).items() if 
            type(layer) in init_funcs for p,v in init_funcs[type(layer)](layer).items()}

#####################
## dict utils
#####################

union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}

def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict): yield from path_iter(val, (*pfx, name))
        else: yield ((*pfx, name), val)  


#####################
## graph building
#####################

sep='_'
RelativePath = namedtuple('RelativePath', ('parts'))
rel_path = lambda *parts: RelativePath(parts)

def build_graph(net):
    net = dict(path_iter(net)) 
    default_inputs = [[('input',)]]+[[k] for k in net.keys()]
    with_default_inputs = lambda vals: (val if isinstance(val, tuple) else (val, default_inputs[idx]) for 
                                        idx,val in enumerate(vals))
    parts = lambda path, pfx: (tuple(pfx) + path.parts if isinstance(path, RelativePath) else (path,) if 
                                isinstance(path, str) else path)
    return {sep.join((*pfx, name)): (val, [sep.join(parts(x, pfx)) for x in inputs]) for 
            (*pfx, name), (val, inputs) in zip(net.keys(), with_default_inputs(net.values()))}

#####################
## training utils
#####################

class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]

concat = lambda xs: np.array(xs) if xs[0].shape is () else np.concatenate(xs)

def collect(stats, output):
    for k,v in stats.items():
        v.append(to_numpy(output[k]))

@singledispatch
def forward(model, batch, recording=False):
    return model(batch)

@singledispatch
def grad_data(param):
    return p.grad.data

@singledispatch
def weight_data(param):
    return p.data


class StatsLogger():
    def __init__(self, keys):
        self.stats = {k:[] for k in keys}
        self.N = 0

    def append(self, output):
        for k,v in self.stats.items():
            v.append(to_numpy(output[k]))
        self.N += len(output['target'])
    
    def mean(self, key):
        return np.sum(concat(self.stats[key]), dtype=np.float)/self.N

def run_batches(model, batches, training, optimizer_step=None, stats=None):
    stats = stats or StatsLogger(('loss', 'correct'))
    model.train(training)   
    for batch in batches:
        output = forward(model, batch, recording=training)
        stats.append(output) #transfers data to the CPU. don't move from here without benchmarking as it may change overlap of data transfer and computation 
        if training:
            output['loss'].backward()
            optimizer_step()
            model.zero_grad() 
    return stats

@singledispatch
def add_(x, a, y):
    #x += a*y
    raise NotImplementedError

@singledispatch
def mul_(x, y):
    #x *= y
    raise NotImplementedError

@singledispatch
def zeros_like(x):
    raise NotImplementedError

class Nesterov():
    def __init__(self, weights, params, state=None):
        self.weights = weights
        self.params = iter(params) 
        self.state = [zeros_like(weight_data(w)) for w in weights] if state is None else state
        
    def step(self):
        lr, weight_decay, momentum = next(self.params)
        for w, g, v in zip((weight_data(w) for w in self.weights), (grad_data(w) for w in self.weights), self.state):
            add_(g, weight_decay, w) 
            mul_(v, momentum)
            add_(v, 1, g)
            add_(g, momentum, v)
            add_(w, -lr, g)

#def set_params(opt, params):
#    return type(opt)(opt.weights, opt.state, params)
    
def train_epoch(model, train_batches, test_batches, optimizer_step, timer, test_time_in_total=True):
    train_stats, train_time = run_batches(model, train_batches, True, optimizer_step), timer()
    test_stats, test_time = run_batches(model, test_batches, False), timer(test_time_in_total)
    return { 
        'train time': train_time, 'train loss': train_stats.mean('loss'), 'train acc': train_stats.mean('correct'), 
        'test time': test_time, 'test loss': test_stats.mean('loss'), 'test acc': test_stats.mean('correct'),
        'total time': timer.total_time, 
    }


#####################
## network visualisation (requires pydot)
#####################
class ColorMap(dict):
    palette = (
        'bebada,ffffb3,fb8072,8dd3c7,80b1d3,fdb462,b3de69,fccde5,bc80bd,ccebc5,ffed6f,1f78b4,33a02c,e31a1c,ff7f00,'
        '4dddf8,e66493,b07b87,4e90e3,dea05e,d0c281,f0e189,e9e8b1,e0eb71,bbd2a4,6ed641,57eb9c,3ca4d4,92d5e7,b15928'
    ).split(',')
    def __missing__(self, key):
        self[key] = self.palette[len(self) % len(self.palette)]
        return self[key]

def make_pydot(nodes, edges, direction='LR', sep=sep, **kwargs):
    import pydot
    parent = lambda path: path[:-1]
    stub = lambda path: path[-1]
    class Subgraphs(dict):
        def __missing__(self, path):
            subgraph = pydot.Cluster(sep.join(path), label=stub(path), style='rounded, filled', fillcolor='#77777744')
            self[parent(path)].add_subgraph(subgraph)
            return subgraph
    subgraphs = Subgraphs()
    subgraphs[()] = g = pydot.Dot(rankdir=direction, directed=True, **kwargs)
    g.set_node_defaults(
        shape='box', style='rounded, filled', fillcolor='#ffffff')
    for node, attr in nodes:
        path = tuple(node.split(sep))
        subgraphs[parent(path)].add_node(
            pydot.Node(name=node, label=stub(path), **attr))
    for src, dst, attr in edges:
        g.add_edge(pydot.Edge(src, dst, **attr))
    return g

get_params = lambda mod: {p.name: getattr(mod, p.name, '?') for p in signature(type(mod)).parameters.values()}


class DotGraph():
    colors = ColorMap()
    def __init__(self, net, size=15, direction='LR'):
        graph = build_graph(net)
        self.nodes = [(k, {
            'tooltip': '%s %.1000r' % (type(n).__name__, get_params(n)), 
            'fillcolor': '#'+self.colors[type(n)],
        }) for k, (n, i) in graph.items()] 
        self.edges = [(src, k, {}) for (k, (n, i)) in graph.items() for src in i]
        self.size, self.direction = size, direction

    def dot_graph(self, **kwargs):
        return make_pydot(self.nodes, self.edges, size=self.size, 
                            direction=self.direction, **kwargs)

    def svg(self, **kwargs):
        return self.dot_graph(**kwargs).create(format='svg').decode('utf-8')

    try:
        import pydot
        def _repr_svg_(self):
            return self.svg()
    except ImportError:
        def __repr__(self):
            return 'pydot is needed for network visualisation'

walk = lambda dict_, key: walk(dict_, dict_[key]) if key in dict_ else key
   
def remove_identity_nodes(net):  
    #remove identity nodes for more compact visualisations
    graph = build_graph(net)
    remap = {k: i[0] for k,(v,i) in graph.items() if isinstance(v, Identity)}
    return {k: (v, [walk(remap, x) for x in i]) for k, (v,i) in graph.items() if not isinstance(v, Identity)}

