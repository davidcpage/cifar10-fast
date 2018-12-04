# cifar10-fast

Demonstration of training a small ResNet on CIFAR10 to 94% test accuracy in 79 seconds as described [in this blog series](https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet/).

<img src="net.svg">

Instructions to reproduce on an `AWS p3.2xlarge` instance:
- setup an instance with AMI: `Deep Learning AMI (Ubuntu) Version 11.0` (`ami-c47c28bc` in `us-west-2`) 
- ssh into the instance: `ssh -i $KEY_PAIR ubuntu@$PUBLIC_IP_ADDRESS -L 8901:localhost:8901`
- on the remote machine
    - `source activate pytorch_p36`
    - `pip install pydot` (optional for network visualisation)
    - `git clone https://github.com/davidcpage/cifar10-fast.git`
    - `jupyter notebook --no-browser --port=8901`
 - open the jupyter notebook url in a browser, open `demo.ipynb` and run all the cells

 In my test, 35 out of 50 runs reached 94% test set accuracy with a median of 94.08%. Runtime for 24 epochs is roughly 79s.

 A second notebook `experiments.ipynb` contains code to reproduce the main results from the [posts](https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet/).

NB: `demo.ipynb` also works on the latest `Deep Learning AMI (Ubuntu) Version 16.0`, but some examples in `experiments.ipynb` trigger a core dump when using TensorCores in versions after `11.0`.
 
## DAWNBench 
 To reproduce [DAWNBench](https://dawn.cs.stanford.edu/benchmark/index.html#cifar10-train-time) timings, setup the `AWS p3.2xlarge` instance as above but instead of launching a jupyter notebook on the remote machine, change directory to `cifar10-fast` and run `python dawn.py` from the command line. Timings in DAWNBench format will be saved to `logs.tsv`. 
 
 Note that DAWNBench timings do not include validation time, as in [this FAQ](https://github.com/stanford-futuredata/dawn-bench-entries), but do include initial preprocessing, as indicated [here](https://groups.google.com/forum/#!topic/dawn-bench-community/YSDRTOLMaMU). DAWNBench timing is roughly 74 seconds which breaks down as 79s (as above) -7s (validation)+ 2s (preprocessing).

## Update 4th Dec 2018
- Core functionality has moved to `core.py` whilst PyTorch specific stuff is in `torch_backend.py` to allow easier experimentation with different frameworks.
- Stats (loss/accuracy) are collected on the GPU and bulk transferred to the CPU at the end of each epoch. This speeds up some experiments so timings in `demo.ipynb` and `experiments.ipynb` no longer match the blog posts.

 


