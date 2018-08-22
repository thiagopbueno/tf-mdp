# This file is part of tf-mdp.

# tf-mdp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# tf-mdp is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with tf-mdp. If not, see <http://www.gnu.org/licenses/>.


from pyrddl.parser import RDDLParser
from tfrddlsim.rddl2tf.compiler import Compiler
from tfmdp.planner import PolicyOptimizationPlanner

import datetime
import itertools
import sys
import time


PREFIX = 'results/{}'.format('{:%Y-%m-%d}'.format(datetime.datetime.today()))


def make_logdir(channels, layers, batch_size, learning_rate):
    return '/channels={}_layers={}_batch={}_lr={}'.format(channels, '+'.join(map(str, layers)), batch_size, learning_rate)


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def parse_rddl(path):
    parser = RDDLParser()
    parser.build()
    rddl = parser.parse(read_file(path))
    return rddl


def compile(rddl):
    rddl2tf = Compiler(rddl, batch_mode=True)
    return rddl2tf


def print_params(channels, layers, batch_size, learning_rate):
    print()
    print('>> Policy Net:')
    print('channels = {}'.format(channels))
    print('layers   = [{}]'.format(','.join(map(str, layers))))
    print()
    print('>> Training parameters:')
    print('batch size    = {}'.format(batch_size))
    print('learning rate = {}'.format(learning_rate))
    print()


def run(rddl, channels, layers, batch_size, learning_rate, horizon, epochs):
    rddl2tf = compile(rddl)
    logdir = PREFIX + make_logdir(channels, layers, batch_size, learning_rate)
    planner = PolicyOptimizationPlanner(rddl2tf, channels, layers, logdir)
    planner.build(learning_rate, batch_size, horizon)
    _, logdir = planner.run(epochs)
    print()
    print(logdir)


if __name__ == '__main__':

    rddl = parse_rddl(sys.argv[1])

    epochs = 1000
    horizon = 20

    HYPERPARAMETERS = {
        'channels': [1, 4, 16],
        'layers': [[1024], [512, 256], [256, 128, 64], [128, 64, 32, 16]],
        'batch_size': [256, 512, 1024, 2048],
        'learning_rate': [0.001, 0.0001, 0.00001, 0.000001]
    }

    params = ['channels', 'layers', 'batch_size', 'learning_rate']
    values = [HYPERPARAMETERS[param] for param in params]

    for (c, l, b, lr) in itertools.product(*values):
        print('>>>>>> Training ...')
        print_params(c, l, b, lr)
        start = time.time()
        run(rddl, c, l, b, lr, horizon, epochs)
        end = time.time()
        uptime = end - start
        print()
        print('<<<<<< Done in {:.6f} sec.'.format(uptime))
        print()

    print('tensorboard --logdir {}'.format(PREFIX))
