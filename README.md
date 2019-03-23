# tf-mdp [![Build Status][travis.svg]][travis] [![Documentation Status][readthedocs]][readthedocs-badge] [![License][license.svg]][license]

Probabilistic planning in continuous state-action MDPs using TensorFlow.

**tf-mdp** is an implementation based on the paper:

> Thiago P. Bueno; Leliane N. de Barros; Denis D. Mauá; Scott Sanner<br>
> **Deep Reactive Policies for Planning in Stochastic Nonlinear Domains**<br>
> In AAAI, 2019.

# Quickstart

**tf-mdp** is a Python3.5+ package available in PyPI.

```text
$ pip3 install tf-mdp
```

Please make sure you have a running TensorFlow version on your system before pip-installing this package.

# Features

**tf-mdp** solves discrete time MDPs with continuous state-action spaces.

The domains/instances are specified using the [RDDL][rddl] language.

It is built on the following packages available on the Python3 RDDL toolkit:

- [pyrddl][pyrddl]: RDDL lexer/parser.
- [rddlgym][rddlgym]: A toolkit for working with RDDL domains.
- [rddl2tf][rddl2tf]: RDDL2TensorFlow compiler.
- [tf-rddlsim][tf-rddlsim]: A RDDL simulator running in TensorFlow.

Please refer to each project documentation for further details.


# Usage

```text
$ tfmdp --help
usage: tfmdp [-h] [-l LAYERS [LAYERS ...]]
             [-a {none,sigmoid,tanh,relu,relu6,crelu,elu,selu,softplus,softsign}]
             [-iln] [-hln] [-kl1 KERNEL_L1_REGULARIZER]
             [-kl2 KERNEL_L2_REGULARIZER] [-bl1 BIAS_L1_REGULARIZER]
             [-bl2 BIAS_L2_REGULARIZER] [-est {pd,sf,hybrid}] [-n N_STEP]
             [-bl] [-b BATCH_SIZE] [-hr HORIZON] [-e EPOCHS]
             [-lr LEARNING_RATE]
             [-opt {Adadelta,Adagrad,Adam,GradientDescent,ProximalGradientDescent,ProximalAdagrad,RMSProp}]
             [-lfn {linear,mse}] [-ld LOGDIR] [-v]
             rddl

Probabilistic planning in continuous state-action MDPs using TensorFlow.

positional arguments:
  rddl                  RDDL file or rddlgym domain id

optional arguments:
  -h, --help            show this help message and exit
  -l LAYERS [LAYERS ...], --layers LAYERS [LAYERS ...]
                        number of units in each hidden layer in policy network
  -a {none,sigmoid,tanh,relu,relu6,crelu,elu,selu,softplus,softsign}, --activation {none,sigmoid,tanh,relu,relu6,crelu,elu,selu,softplus,softsign}
                        activation function for hidden layers in policy
                        network
  -iln, --input-layer-norm
                        input layer normalization flag
  -hln, --hidden-layer-norm
                        hidden layer normalization flag
  -kl1 KERNEL_L1_REGULARIZER, --kernel-l1-regularizer KERNEL_L1_REGULARIZER
                        kernel L1-regularizer constant (default=0.0)
  -kl2 KERNEL_L2_REGULARIZER, --kernel-l2-regularizer KERNEL_L2_REGULARIZER
                        kernel L2-regularizer constant (default=0.0)
  -bl1 BIAS_L1_REGULARIZER, --bias-l1-regularizer BIAS_L1_REGULARIZER
                        bias L1-regularizer constant (default=0.0)
  -bl2 BIAS_L2_REGULARIZER, --bias-l2-regularizer BIAS_L2_REGULARIZER
                        bias L2-regularizer constant (default=0.0)
  -est {pd,sf,hybrid}, --estimator {pd,sf,hybrid}
                        gradient estimator (pd=pathwise derivatives, sf=score
                        function, hybrid=pd+sf) (default=pd)
  -n N_STEP, --n-step N_STEP
                        n-step hybrid estimator
  -bl, --baseline       baseline flag
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        number of trajectories in a batch (default=256)
  -hr HORIZON, --horizon HORIZON
                        number of timesteps (default=40)
  -e EPOCHS, --epochs EPOCHS
                        number of timesteps (default=200)
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        optimizer learning rate (default=0.001)
  -opt {Adadelta,Adagrad,Adam,GradientDescent,ProximalGradientDescent,ProximalAdagrad,RMSProp}, --optimizer {Adadelta,Adagrad,Adam,GradientDescent,ProximalGradientDescent,ProximalAdagrad,RMSProp}
                        loss optimizer (default=RMSProp)
  -lfn {linear,mse}, --loss-fn {linear,mse}
                        loss function (default=linear)
  -ld LOGDIR, --logdir LOGDIR
                        log directory for data summaries (default=/tmp/tfmdp)
  -v, --verbose         verbosity mode
```

# Examples

```text
$ tfmdp Reservoir-20 -l 2048 -iln -a elu -b 256 -hr 40 -e 200 -lr 0.001 -lfn mse -v

Running tf-mdp v0.4.4 ...

>> RDDL:   Reservoir-20
>> logdir: /tmp/tfmdp

>> Policy Net:
layers = [2048]
activation = elu
input  layer norm = True
hidden layer norm = False

>> Hyperparameters:
epochs        = 200
learning rate = 0.001
batch size    = 256
horizon       = 40

>> Optimization:
estimator = pd
optimizer     = RMSProp
loss function = mse
kernel l1-regularization = 0.0
kernel l2-regularization = 0.0
bias l1-regularization = 0.0
bias l2-regularization = 0.0

>> Loading model ...
Done in 0.058076 sec.

>> Optimizing...
Epoch   199: loss = 1183033344.000000
Done in 251.438551 sec.

>> Performance:
total reward = -4019.2837, reward per timestep = -100.4821

tensorboard --logdir /tmp/tfmdp/train
```

```text
$ tfmdp HVAC-3 -l 256 128 64 32 -iln -a elu -b 256 -hr 40 -e 200 -lr 0.0001 -lfn mse -v

Running tf-mdp v0.4.4 ...

>> RDDL:   HVAC-3
>> logdir: /tmp/tfmdp

>> Policy Net:
layers = [256,128,64,32]
activation = elu
input  layer norm = True
hidden layer norm = False

>> Hyperparameters:
epochs        = 200
learning rate = 0.0001
batch size    = 256
horizon       = 40

>> Optimization:
estimator = pd
optimizer     = RMSProp
loss function = mse
kernel l1-regularization = 0.0
kernel l2-regularization = 0.0
bias l1-regularization = 0.0
bias l2-regularization = 0.0

>> Loading model ...
Done in 0.045406 sec.

>> Optimizing...
2019-03-23 15:57:55.761501: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.2
Epoch   199: loss = 24277512192.0000000
Done in 93.795296 sec.

>> Performance:
total reward = -306241.7500, reward per timestep = -7656.0437

tensorboard --logdir /tmp/tfmdp/train
```

```text
$ tfmdp Navigation-v2 -l 256 128 64 32 -a elu -b 128 -hr 20 -e 200 -lr 0.001 -lfn mse -v

Running tf-mdp v0.4.4 ...

>> RDDL:   Navigation-v2
>> logdir: /tmp/tfmdp

>> Policy Net:
layers = [256,128,64,32]
activation = elu
input  layer norm = False
hidden layer norm = False

>> Hyperparameters:
epochs        = 200
learning rate = 0.001
batch size    = 128
horizon       = 20

>> Optimization:
estimator = pd
optimizer     = RMSProp
loss function = mse
kernel l1-regularization = 0.0
kernel l2-regularization = 0.0
bias l1-regularization = 0.0
bias l2-regularization = 0.0

>> Loading model ...
Done in 0.039008 sec.

>> Optimizing...
2019-03-23 16:02:54.502184: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.2
Epoch   199: loss = 555.7142338
Done in 28.249370 sec.

>> Performance:
total reward = -79.6495, reward per timestep = -3.9825

tensorboard --logdir /tmp/tfmdp/train
```

# Documentation

Please refer to [https://tf-mdp.readthedocs.io/][readthedocs] for the code documentation.


# Support

If you are having issues with tf-mdp, please let me know at: [thiago.pbueno@gmail.com](mailto://thiago.pbueno@gmail.com).

# License

Copyright (c) 2018-2019 Thiago Pereira Bueno All Rights Reserved.

tf-mdp is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

tf-mdp is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with tf-mdp. If not, see http://www.gnu.org/licenses/.

[license.svg]: https://img.shields.io/aur/license/yaourt.svg
[license]: https://github.com/thiagopbueno/tf-mdp/blob/master/LICENSE
[pyrddl]: https://github.com/thiagopbueno/pyrddl
[rddl2tf]: https://github.com/thiagopbueno/rddl2tf
[rddl]: http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/RDDL.pdf
[rddlgym]: https://github.com/thiagopbueno/rddlgym
[readthedocs-badge]: https://tf-mdp.readthedocs.io/en/latest/?badge=latest
[readthedocs]: https://readthedocs.org/projects/tf-mdp/badge/?version=latest
[tf-rddlsim]: https://github.com/thiagopbueno/tf-rddlsim
[travis.svg]: https://travis-ci.org/thiagopbueno/tf-mdp.svg?branch=master
[travis]: https://travis-ci.org/thiagopbueno/tf-mdp
