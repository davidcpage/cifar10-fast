from core import *
from torch_backend import *
import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_dir', type=str, default='.')
     
#Network definition
def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
    return {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
        'bn': BatchNorm(c_out, weight_init=bn_weight_init, **kw), 
        'relu': nn.ReLU(True)
    }

def residual(c, **kw):
    return {
        'in': Identity(),
        'res1': conv_bn(c, c, **kw),
        'res2': conv_bn(c, c, **kw),
        'add': (Add(), ['in', 'res2/relu']),
    }

def basic_net(channels, weight,  pool, **kw):
    return {
        'input': None,
        'prep': conv_bn(3, channels['prep'], **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool),
        'pool': nn.MaxPool2d(4),
        'flatten': Flatten(),
        'linear': nn.Linear(channels['layer3'], 10, bias=False),
        'logits': Mul(weight),
    }

def net(channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3'), **kw):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    n = basic_net(channels, weight, pool, **kw)
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], **kw)       
    return n

class TSVLogger():
    def __init__(self):
        self.log = ['epoch\thours\ttop1Accuracy']
    def append(self, output):
        epoch, hours, acc = output['epoch'], output['total time']/3600, output['valid']['acc']*100
        self.log.append(f'{epoch}\t{hours:.8f}\t{acc:.2f}')
    def __str__(self):
        return '\n'.join(self.log)
    

def main():  
    args = parser.parse_args()
    
    print('Downloading datasets')
    dataset = cifar10(args.data_dir)
    mean, std = [np.array(x, dtype=np.float32) for x in (cifar10_mean, cifar10_std)]

    epochs = 24
    lr_schedule = PiecewiseLinear([0, 5, epochs], [0, 0.4, 0])
    batch_size = 512
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]

    model = Network(net()).to(device).half()
    loss = x_ent_loss
    random_batch = lambda batch_size:  {
        'input': torch.Tensor(np.random.rand(batch_size,3,32,32)).cuda().half(), 
        'target': torch.LongTensor(np.random.randint(0,10,batch_size)).cuda()
    }
    print('Warming up cudnn on random inputs')
    for size in [batch_size, len(dataset['valid']['targets']) % batch_size]:
        warmup_cudnn(model, loss, random_batch(size))
    
    print('Starting timer')
    timer = Timer(synch=torch.cuda.synchronize)
    
    print('Preprocessing training data')
    train_set = list(zip(transpose(normalise(pad(dataset['train']['data'], 4), mean, std), 'NHWC', 'NCHW'), dataset['train']['targets']))
    print(f'Finished in {timer():.2} seconds')
    print('Preprocessing test data')
    test_set = list(zip(transpose(normalise(dataset['valid']['data'], mean, std), 'NHWC', 'NCHW'), dataset['valid']['targets']))
    print(f'Finished in {timer():.2} seconds')
    
    TSV = TSVLogger()
    
    train_batches = Batches(Transform(train_set, train_transforms), batch_size, shuffle=True, set_random_choices=True, drop_last=True)
    test_batches = Batches(test_set, batch_size, shuffle=False, drop_last=False)
    lr = lambda step: lr_schedule(step/len(train_batches))/batch_size
    opt_params = {'lr': lr, 'weight_decay': Const(5e-4*batch_size), 'momentum': Const(0.9)}
    logs, state = (Table(), TSV), {MODEL: model, LOSS: loss, OPTS: [SGD(trainable_params(model).values(), opt_params)]}
    for epoch in range(epochs):
        summary = union({'epoch': epoch+1}, train_epoch(state, timer, train_batches, test_batches))
        for l in logs: l.append(summary)
    
    with open(os.path.join(os.path.expanduser(args.log_dir), 'logs.tsv'), 'w') as f:
        f.write(str(TSV))        
        
main()