# tf-mdp[![Py Versions][py-versions.svg]][pypi-project] [![PyPI version][pypi-version.svg]][pypi-version] [![Build Status][travis.svg]][travis-project] [![Documentation Status][rtd-badge.svg]][rtd-badge] [![License: GPL v3][license.svg]][license]

Probabilistic planning in continuous state-action MDPs using TensorFlow.

**tf-mdp** is an implementation based on the paper:

> Thiago P. Bueno; Leliane N. de Barros; Denis D. Mauá; Scott Sanner<br>
> **[Deep Reactive Policies for Planning in Stochastic Nonlinear Domains](https://aaai.org/ojs/index.php/AAAI/article/view/4744)**<br>
> In AAAI, 2019.

# Quickstart

**tf-mdp** is a Python3.6+ package available in PyPI.

```text
$ pip3 install tf-mdp
```

Please make sure you have a running TensorFlow version on your system before pip-installing this package.

# Features

**tf-mdp** solves discrete-time continuous state-action MDPs.

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
             [-iln] [-b BATCH_SIZE] [-hr HORIZON] [-e EPOCHS]
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

Running tf-mdp v0.5.4 ...

>> RDDL:   Reservoir-20
>> logdir: /tmp/tfmdp

>> Policy Net:
layers = [2048]
activation = elu
input  layer norm = True

>> Hyperparameters:
epochs        = 200
learning rate = 0.001
batch size    = 256
horizon       = 40

>> Optimization:
optimizer     = RMSProp
loss function = mse

>> Loading model ...
Done in 0.018952 sec.

>> Optimizing...
2021-06-23 22:56:18.873731: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2021-06-23 22:56:18.895765: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2199995000 Hz
2021-06-23 22:56:18.896462: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x46628b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-06-23 22:56:18.896514: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Epoch   199: loss = 1201677952.000000
Done in 28.525183 sec.

>> Performance:
total reward = -3653.9695, reward per timestep = -91.3492
```

```text
$ tfmdp HVAC-3 -l 256 128 64 32 -iln -a elu -b 256 -hr 40 -e 200 -lr 0.0001 -lfn mse -v

Running tf-mdp v0.5.4 ...

>> RDDL:   HVAC-3
>> logdir: /tmp/tfmdp

>> Policy Net:
layers = [256,128,64,32]
activation = elu
input  layer norm = True

>> Hyperparameters:
epochs        = 200
learning rate = 0.0001
batch size    = 256
horizon       = 40

>> Optimization:
optimizer     = RMSProp
loss function = mse

>> Loading model ...
Done in 0.017646 sec.

>> Optimizing...
2021-06-23 22:54:05.766434: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2021-06-23 22:54:05.787832: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2199995000 Hz
2021-06-23 22:54:05.788607: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x49a4d00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-06-23 22:54:05.788690: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Epoch   199: loss = 103798661120.0000000
Done in 15.748765 sec.

>> Performance:
total reward = -315724.4688, reward per timestep = -7893.1117
```

```text
$ tfmdp Navigation-v2 -l 256 128 64 32 -a elu -b 128 -hr 20 -e 200 -lr 0.001 -lfn mse -v

Running tf-mdp v0.5.4 ...

>> RDDL:   Navigation-v2
>> logdir: /tmp/tfmdp

>> Policy Net:
layers = [256,128,64,32]
activation = elu
input  layer norm = False

>> Hyperparameters:
epochs        = 200
learning rate = 0.001
batch size    = 128
horizon       = 20

>> Optimization:
optimizer     = RMSProp
loss function = mse

>> Loading model ...
Done in 0.012209 sec.

>> Optimizing...
2021-06-23 22:50:59.732002: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2021-06-23 22:50:59.751959: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2199995000 Hz
2021-06-23 22:50:59.752494: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5bc6a20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-06-23 22:50:59.752514: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Epoch   199: loss = 6452.3613285
Done in 6.466699 sec.

>> Performance:
total reward = -78.3427, reward per timestep = -3.9171
```

# Documentation

Please refer to [https://tf-mdp.readthedocs.io/][readthedocs] for the code documentation.


# Support

If you are having issues with tf-mdp, please let me know at: [thiago.pbueno@gmail.com](mailto://thiago.pbueno@gmail.com).

# License

Copyright (c) 2018-2021 Thiago Pereira Bueno All Rights Reserved.

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


[pyrddl]: https://github.com/thiagopbueno/pyrddl
[rddl2tf]: https://github.com/thiagopbueno/rddl2tf
[rddl]: http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/RDDL.pdf
[rddlgym]: https://github.com/thiagopbueno/rddlgym
[tf-rddlsim]: https://github.com/thiagopbueno/tf-rddlsim

[py-versions.svg]: https://img.shields.io/pypi/pyversions/tf-mdp.svg?logo=python&logoColor=white
[pypi-project]: https://pypi.org/project/tf-mdp

[pypi-version.svg]: https://badge.fury.io/py/tf-mdp.svg
[pypi-version]: https://badge.fury.io/py/tf-mdp

[travis.svg]: https://img.shields.io/travis/thiagopbueno/tf-mdp/master.svg?logo=travis
[travis-project]: https://travis-ci.org/thiagopbueno/tf-mdp

[rtd-badge.svg]: https://readthedocs.org/projects/tf-mdp/badge/?version=latest
[rtd-badge]: https://tf-mdp.readthedocs.io/en/latest/?badge=latest

[license.svg]: https://img.shields.io/badge/License-GPL%20v3-blue.svg
[license]: https://github.com/thiagopbueno/tf-mdp/blob/master/LICENSE
