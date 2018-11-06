from utils import *

#Network definition
def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
    return {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
        'bn': batch_norm(c_out, bn_weight_init=bn_weight_init, **kw), 
        'relu': nn.ReLU(True)
    }

def residual(c, **kw):
    return {
        'in': Identity(),
        'res1': conv_bn(c, c, **kw),
        'res2': conv_bn(c, c, **kw),
        'add': (Add(), [rel_path('in'), rel_path('res2', 'relu')]),
    }

def basic_net(channels, weight,  pool, **kw):
    return {
        'prep': conv_bn(3, channels['prep'], **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool),
        'classifier': {
            'pool': nn.MaxPool2d(4),
            'flatten': Flatten(),
            'linear': nn.Linear(channels['layer3'], 10, bias=False),
            'logits': Mul(weight),
        }
    }

def net(channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3'), **kw):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    n = basic_net(channels, weight, pool, **kw)
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], **kw)       
    return n

losses = {
    'loss':  (nn.CrossEntropyLoss(size_average=False), [('classifier','logits'), ('target',)]),
    'correct': (Correct(), [('classifier','logits'), ('target',)]),
}

class TSVLogger():
    def __init__(self):
        self.log = ['epoch\thours\ttop1Accuracy']
    def append(self, output):
        epoch, hours, acc = output['epoch'], output['total time']/3600, output['test acc']*100
        self.log.append(f'{epoch}\t{hours:.8f}\t{acc:.2f}')
    def __str__(self):
        return '\n'.join(self.log)
   
def main():
    DATA_DIR = './data'

    print('Downloading datasets')
    train_set_raw = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    test_set_raw = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True)

    lr_schedule = PiecewiseLinear([0, 5, 24], [0, 0.4, 0])
    batch_size = 512
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]

    model = TorchGraph(union(net(), losses)).to(device).half()
    opt = nesterov(trainable_params(model), momentum=0.9, weight_decay=5e-4*batch_size)
    
    print('Warming up cudnn on random inputs')
    for size in [batch_size, len(test_set_raw) % batch_size]:
        warmup_cudnn(model, size)
    
    print('Starting timer')
    t = Timer()
    
    print('Preprocessing training data')
    train_set = list(zip(transpose(normalise(pad(train_set_raw.train_data, 4))), train_set_raw.train_labels))
    print(f'Finished in {t():.2} seconds')
    print('Preprocessing test data')
    test_set = list(zip(transpose(normalise(test_set_raw.test_data)), test_set_raw.test_labels))
    print(f'Finished in {t():.2} seconds')
    
    TSV = TSVLogger()
    train(model, lr_schedule, opt, Transform(train_set, train_transforms), test_set, 
          batch_size=batch_size, loggers=(TableLogger(), TSV), timer=t, test_time_in_total=False, drop_last=True)
    
    with open('logs.tsv', 'w') as f:
        f.write(str(TSV))

main()