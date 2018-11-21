import numpy as np
import torch
from torch import nn
import torch.utils.data
from dataclasses import asdict
import utils
torch.backends.cudnn.benchmark = True

# pylint: disable=E1101
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=E1101

@utils.to_numpy.register(torch.Tensor)
def to_numpy(x):
    return x.detach().cpu().numpy()  

@utils.add_.register(torch.Tensor)
def _(x, a, y):
    if a is 0: return
    if a is 1: x.add_(y)
    else: x.add_(a, y)

@utils.mul_.register(torch.Tensor)
def _(x, y):
    x.mul_(y)

@utils.zeros_like.register(torch.Tensor)
def _(x):
    return torch.zeros_like(x)

@utils.transfer.register(torch.Tensor)
def _(data, device):
    return data.to(device)

def from_numpy(x):
    type_map = {
        np.float16: torch.HalfTensor,
        np.float32: torch.Tensor,
        np.int32: torch.LongTensor,
        np.int: torch.LongTensor,
        int: torch.LongTensor
    }
    return type_map[x.dtype.type](x)

#####################
## data loading
#####################

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )
    
    def __iter__(self):
        if self.set_random_choices: self.dataloader.dataset.set_random_choices()
        return ({'input': x.to(device).half(), 'target': y.to(device).long()} for (x,y) in self.dataloader)
    
    def __len__(self): 
        return len(self.dataloader)


class Concat(utils.Concat):
    def __call__(self,  *xs):
        # pylint: disable=E1101
        return torch.cat(*xs, dim=self.dim)
        # pylint: enable=E1101

def convert_to_torch(layer):
    x_ent = lambda weight, reduction: nn.CrossEntropyLoss(weight, 
            size_average=(reduction != 'sum'), reduce=(reduction != 'none'))
    type_map = {
        utils.Conv2d: nn.Conv2d,
        utils.Linear: nn.Linear,
        utils.BatchNorm2d: nn.BatchNorm2d,
        utils.MaxPool2d: nn.MaxPool2d,
        utils.CrossEntropyLoss: x_ent,
        utils.ReLU: nn.ReLU,
        utils.Concat: Concat
    }
    if type(layer) in type_map:
        return type_map[type(layer)](**asdict(layer))
    return layer


class Network(nn.Module):  
    def __init__(self, net):
        super().__init__()
        self.graph = utils.build_graph(net)
        self.cache = None
        for n, (v, _) in self.graph.items(): 
            setattr(self, n, convert_to_torch(v))

    def forward(self, inputs):
        self.cache=dict(inputs)
        for n, (_, i) in self.graph.items():
            self.cache[n] = getattr(self, n)(*[self.cache[x] for x in i])
        return self.cache
    
    def half(self):
        for module in self.children():
            if type(module) is not nn.BatchNorm2d:
                module.half()    
        return self

    def load_weights(self, weights, device=device):
        self.load_state_dict({k: from_numpy(v) for k,v in weights.items()})
        return self.to(device)
    
    def trainable_params(self):
        return list(filter(lambda p: p.requires_grad, self.parameters()))

def warmup_cudnn(model, batch_size):
    #run forward and backward pass of the model on a batch of random inputs
    #to allow benchmarking of cudnn kernels 
    # pylint: disable=E1101
    batch = {
        'input': torch.Tensor(np.random.rand(batch_size, 3, 32, 32)).cuda().half(), 
        'target': torch.LongTensor(np.random.randint(0, 10, batch_size)).cuda()
    }
    # pylint: enable=E1101
    model.train(True)
    o = model(batch)
    o['loss'].backward()
    model.zero_grad()
    torch.cuda.synchronize()

