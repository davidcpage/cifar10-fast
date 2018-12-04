from inspect import signature
from collections import namedtuple
import time
import numpy as np
import pandas as pd
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
    with_default_inputs = lambda vals: (val if isinstance(val, tuple) else (val, default_inputs[idx]) for idx,val in enumerate(vals))
    parts = lambda path, pfx: tuple(pfx) + path.parts if isinstance(path, RelativePath) else (path,) if isinstance(path, str) else path
    return {sep.join((*pfx, name)): (val, [sep.join(parts(x, pfx)) for x in inputs]) for (*pfx, name), (val, inputs) in zip(net.keys(), with_default_inputs(net.values()))}
    

#####################
## training utils
#####################

@singledispatch
def cat(*xs):
    raise NotImplementedError
    
@singledispatch
def to_numpy(x):
    raise NotImplementedError


class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]

class StatsLogger():
    def __init__(self, keys):
        self._stats = {k:[] for k in keys}

    def append(self, output):
        for k,v in self._stats.items():
            v.append(output[k].detach())
    
    def stats(self, key):
        return cat(*self._stats[key])
        
    def mean(self, key):
        return np.mean(to_numpy(self.stats(key)), dtype=np.float)

def run_batches(model, batches, training, optimizer_step=None, stats=None):
    stats = stats or StatsLogger(('loss', 'correct'))
    model.train(training)   
    for batch in batches:
        output = model(batch)
        stats.append(output) 
        if training:
            output['loss'].sum().backward()
            optimizer_step()
            model.zero_grad() 
    return stats
    
def train_epoch(model, train_batches, test_batches, optimizer_step, timer, test_time_in_total=True):
    train_stats, train_time = run_batches(model, train_batches, True, optimizer_step), timer()
    test_stats, test_time = run_batches(model, test_batches, False), timer(test_time_in_total)
    return { 
        'train time': train_time, 'train loss': train_stats.mean('loss'), 'train acc': train_stats.mean('correct'), 
        'test time': test_time, 'test loss': test_stats.mean('loss'), 'test acc': test_stats.mean('correct'),
        'total time': timer.total_time, 
    }

def train(model, optimizer, train_batches, test_batches, epochs, 
          loggers=(), test_time_in_total=True, timer=None):  
    timer = timer or Timer()
    for epoch in range(epochs):
        epoch_stats = train_epoch(model, train_batches, test_batches, optimizer.step, timer, test_time_in_total=test_time_in_total) 
        summary = union({'epoch': epoch+1, 'lr': optimizer.param_values()['lr']*train_batches.batch_size}, epoch_stats)
        for logger in loggers:
            logger.append(summary)    
    return summary

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
   
def remove_by_type(net, node_type):  
    #remove identity nodes for more compact visualisations
    graph = build_graph(net)
    remap = {k: i[0] for k,(v,i) in graph.items() if isinstance(v, node_type)}
    return {k: (v, [walk(remap, x) for x in i]) for k, (v,i) in graph.items() if not isinstance(v, node_type)}