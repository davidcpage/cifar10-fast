# cifar10-fast

Demonstration of training CIFAR10 to 94% test accuracy in 87s.

Instructions to reproduce on an AWS p3.2xlarge instance:
- setup an instance with AMI 'Deep Learning AMI (Ubuntu) Version 11.0'
- copy across setup.sh and demo.ipynb
- ssh into the instance and run
  - source activate pytorch_p36
  - sh setup.sh
- open a jupyter notebook (environment pytorch_p36) and run all the cells in demo.ipynb


