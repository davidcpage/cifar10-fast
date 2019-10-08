from torch_backend import *

#Network definition
def conv_bn_default(c_in, c_out, pool=None):
    block = {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
        'bn': BatchNorm(c_out), 
        'relu': nn.ReLU(True)
    }
    if pool: block['pool'] = pool
    return block

def residual(c, conv_bn, **kw):
    return {
        'in': Identity(),
        'res1': conv_bn(c, c, **kw),
        'res2': conv_bn(c, c, **kw),
        'add': (Add(), ['in', 'res2/relu']),
    }

def net(channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3'), conv_bn=conv_bn_default, prep=conv_bn_default):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    n = {
        'input': (None, []),
        'prep': prep(3, channels['prep']),
        'layer1': conv_bn(channels['prep'], channels['layer1'], pool=pool),
        'layer2': conv_bn(channels['layer1'], channels['layer2'], pool=pool),
        'layer3': conv_bn(channels['layer2'], channels['layer3'], pool=pool),
        'pool': nn.MaxPool2d(4),
        'flatten': Flatten(),
        'linear': nn.Linear(channels['layer3'], 10, bias=False),
        'logits': Mul(weight),
    }
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], conv_bn)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer])       
    return n

def tsv(logs):
    data = [(output['epoch'], output['total time']/3600, output['valid']['acc']*100) for output in logs]
    return '\n'.join(['epoch\thours\ttop1Accuracy']+[f'{epoch}\t{hours:.8f}\t{acc:.2f}' for (epoch, hours, acc) in data])
