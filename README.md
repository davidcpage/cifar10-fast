# cifar10-fast

Demonstration of training a small ResNet on CIFAR10 to 94% test accuracy in 87 seconds as described [in this blog series](https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet/).

<img src="net.svg">

Instructions to reproduce on an `AWS p3.2xlarge` instance:
- setup an instance with AMI: `Deep Learning AMI (Ubuntu) Version 11.0` (`ami-c47c28bc` in `Amazon AMIs`)
- copy across `demo.ipynb`
- ssh into the instance: `ssh -i $KEY_PAIR ubuntu@$PUBLIC_IP_ADDRESS -L 8901:localhost:8901`
- on the remote machine
    - `source activate pytorch_p36`
    - `jupyter notebook --no-browser --port=8901`
 - open the jupyter notebook url in a browser, open `demo.ipynb` and run all the cells

 In my tests, 15 out of 20 runs reached 94% test set accuracy with a median of 94.06%. Runtime for 25 epochs is roughly 87s.


