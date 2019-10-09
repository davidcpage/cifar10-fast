from core import *
from torch_backend import *
from dawn_utils import net, tsv
import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_dir', type=str, default='.')

batch_norm = partial(GhostBatchNorm, num_splits=16, weight_freeze=True)
relu = partial(nn.CELU, alpha=0.3)

def conv_bn(c_in, c_out, pool=None):
    block = {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
        'bn': batch_norm(c_out), 
        'relu': relu(),
    }
    if pool: block = {'conv': block['conv'], 'pool': pool, 'bn': block['bn'], 'relu': block['relu']}
    return block

def whitening_block(c_in, c_out, Λ=None, V=None, eps=1e-2):
    return {
        'whiten': whitening_filter(Λ, V, eps),
        'conv': nn.Conv2d(27, c_out, kernel_size=(1, 1), bias=False),
        'norm': batch_norm(c_out), 
        'act':  relu(),
    }

def main():  
    args = parser.parse_args()
    
    print('Downloading datasets')
    dataset = map_nested(torch.tensor, cifar10(args.data_dir))

    epochs, ema_epochs = 10, 2
    lr_schedule = PiecewiseLinear([0, epochs/5, epochs-ema_epochs], [0, 1.0, 0.1])
    batch_size = 512
    train_transforms = [Crop(32, 32), FlipLR()]

    print('Warming up torch')
    random_data = torch.tensor(np.random.randn(1000,3,32,32).astype(np.float16), device=device)
    Λ, V = eigens(patches(random_data))

    loss = label_smoothing_loss(0.2)
    random_batch = lambda batch_size:  {
        'input': torch.Tensor(np.random.rand(batch_size,3,32,32)).cuda().half(), 
        'target': torch.LongTensor(np.random.randint(0,10,batch_size)).cuda()
    }
    print('Warming up cudnn on random inputs')
    model = Network(net(weight=1/16, conv_bn=conv_bn, prep=partial(whitening_block, Λ=Λ, V=V))).to(device).half()
    for size in [batch_size, len(dataset['valid']['targets']) % batch_size]:
        warmup_cudnn(model, loss, random_batch(size))
    
    print('Starting timer')
    timer = Timer(synch=torch.cuda.synchronize)
    
    print('Preprocessing training data')
    dataset = map_nested(to(device), dataset)
    T = lambda x: torch.tensor(x, dtype=torch.float16, device=device)
    transforms = [
        to(dtype=torch.float16),
        partial(normalise, mean=T(cifar10_mean), std=T(cifar10_std)),
        partial(transpose, source='NHWC', target='NCHW'), 
    ]
    train_set = preprocess(dataset['train'], transforms + [partial(pad, border=4)])
    print(f'Finished in {timer():.2} seconds')
    print('Preprocessing test data')
    valid_set = preprocess(dataset['valid'], transforms)
    print(f'Finished in {timer():.2} seconds')

    Λ, V = eigens(patches(train_set['data'][:10000,:,4:-4,4:-4])) #center crop to remove padding
    model = Network(net(weight=1/16, conv_bn=conv_bn, prep=partial(whitening_block, Λ=Λ, V=V))).to(device).half()    
   
    train_batches = GPUBatches(batch_size=batch_size, transforms=train_transforms, dataset=train_set, shuffle=True,  drop_last=True, max_options=200)
    valid_batches = GPUBatches(batch_size=batch_size, dataset=valid_set, shuffle=False, drop_last=False)
    is_bias = group_by_key(('bias' in k, v) for k, v in trainable_params(model).items())
    opts = [
        SGD(is_bias[False], {'lr': (lambda step: lr_schedule(step/len(train_batches))/batch_size), 'weight_decay': Const(5e-4*batch_size), 'momentum': Const(0.9)}),
        SGD(is_bias[True], {'lr': (lambda step: lr_schedule(step/len(train_batches))*(64/batch_size)), 'weight_decay': Const(5e-4*batch_size/64), 'momentum': Const(0.9)})
    ]
    logs, state = Table(), {MODEL: model, VALID_MODEL: copy.deepcopy(model), LOSS: loss, OPTS: opts}

    for epoch in range(epochs):
        logs.append(union({'epoch': epoch+1}, train_epoch(state, timer, train_batches, valid_batches,
                                                          train_steps=(*default_train_steps, update_ema(momentum=0.99, update_freq=5)),
                                                          valid_steps=(forward_tta([(lambda x: x), flip_lr]), log_activations(('loss', 'acc'))))))

    with open(os.path.join(os.path.expanduser(args.log_dir), 'logs.tsv'), 'w') as f:
        f.write(tsv(logs.log))        
        
main()