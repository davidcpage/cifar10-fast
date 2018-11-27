from utils import *
from torch_backend import Batches, Network, warmup_cudnn

#Network definition
def conv_bn(c_in, c_out):
    return {
        'conv': Conv2d(c_in, c_out, (3, 3), stride=1, padding=1, bias=False), 
        'bn': BatchNorm2d(c_out), 
        'relu': ReLU(),
    }

def residual(c):
    return {
        'in': Identity(),
        'res1': conv_bn(c, c),
        'res2': conv_bn(c, c),
        'add': (Add(), [rel_path('in'), rel_path('res2', 'relu')]),
    }

def basic_net(channels, weight,  pool):
    return {
        'prep': conv_bn(3, channels['prep']),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1']), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2']), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3']), pool=pool),
        'pool': MaxPool2d(4),
        'flatten': Flatten(),
        'linear': Linear(channels['layer3'], 10, bias=False),
        'classifier': Mul(weight),
    }

def net(channels=None, weight=0.125, pool=MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3')):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    n = basic_net(channels, weight, pool)
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer])
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer])       
    return n

losses = {
    'loss':  (CrossEntropyLoss(reduction='sum'), [('classifier',), ('target',)]),
    'correct': (Correct(), [('classifier',), ('target',)]),
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
    data, targets = get_cifar10(DATA_DIR)
    
    epochs=24
    lr_schedule = PiecewiseLinear([0, 5, epochs], [0, 0.4, 0])
    batch_size = 512
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]
    momentum = 0.9
    weight_decay = 5e-4

    network = union(net(), losses)
    model = Network(network).load_weights(initial_weights(network)).half()
  
    print('Warming up cudnn on random inputs')
    for size in [batch_size, len(targets['test']) % batch_size]:
        warmup_cudnn(model, size)
    
    print('Starting timer')
    timer = Timer()
    
    print('Preprocessing training data')
    train_set = list(zip(transpose(normalise(pad(data['train'], 4))), targets['train']))
    print(f'Finished in {timer():.2} seconds')
    print('Preprocessing test data')
    test_set = list(zip(transpose(normalise(data['test'])), targets['test']))
    print(f'Finished in {timer():.2} seconds')
    
    
    train_batches = Batches(Transform(train_set, train_transforms), batch_size, 
                        shuffle=True, set_random_choices=True, num_workers=0, drop_last=True)
    test_batches = Batches(test_set, batch_size, shuffle=False, num_workers=0)

    opt_params = [(lr_schedule(x)/batch_size, weight_decay*batch_size, momentum) for 
                      x in np.arange(0, epochs, 1/len(train_batches))]
    optimizer = Nesterov(model.trainable_params(), params=opt_params)
    
    table, tsv = TableLogger(), TSVLogger()
    for epoch in range(1, epochs+1):
        epoch_stats = train_epoch(model, train_batches, test_batches, optimizer.step, timer, test_time_in_total=False) 
        summary = union({'epoch': epoch, 'lr': lr_schedule(epoch)}, epoch_stats)
        table.append(summary)
        tsv.append(summary)
    
    with open('logs.tsv', 'w') as f:
        f.write(str(tsv))

main()