from core import *
from torch_backend import *
from dawn_utils import net, tsv
import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_dir', type=str, default='.')
     
def main():  
    args = parser.parse_args()
    
    print('Downloading datasets')
    dataset = cifar10(args.data_dir)

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
    transforms = [
        partial(normalise, mean=np.array(cifar10_mean, dtype=np.float32), std=np.array(cifar10_std, dtype=np.float32)),
        partial(transpose, source='NHWC', target='NCHW'), 
    ]
    train_set = list(zip(*preprocess(dataset['train'], [partial(pad, border=4)] + transforms).values()))
    print(f'Finished in {timer():.2} seconds')
    print('Preprocessing test data')
    test_set = list(zip(*preprocess(dataset['valid'], transforms).values()))
    print(f'Finished in {timer():.2} seconds')
    
    train_batches = DataLoader(Transform(train_set, train_transforms), batch_size, shuffle=True, set_random_choices=True, drop_last=True)
    test_batches = DataLoader(test_set, batch_size, shuffle=False, drop_last=False)
    opts = [SGD(trainable_params(model).values(), {
        'lr': (lambda step: lr_schedule(step/len(train_batches))/batch_size), 'weight_decay': Const(5e-4*batch_size), 'momentum': Const(0.9)})]
    logs, state = Table(), {MODEL: model, LOSS: loss, OPTS: opts}
    for epoch in range(epochs):
        logs.append(union({'epoch': epoch+1}, train_epoch(state, timer, train_batches, test_batches)))

    with open(os.path.join(os.path.expanduser(args.log_dir), 'logs.tsv'), 'w') as f:
        f.write(tsv(logs.log))        
        
main()