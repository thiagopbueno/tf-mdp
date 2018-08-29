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

import itertools
import os
import sys
import time


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def parse_rddl(path):
    parser = RDDLParser()
    parser.build()
    rddl = parser.parse(read_file(path))
    return rddl


def parse_json(path):
    import json
    params = read_file(path)
    return json.loads(params)


def compile(rddl):
    rddl2tf = Compiler(rddl, batch_mode=True)
    return rddl2tf


def print_params(channels, layers, batch_size, learning_rate, horizon, epochs):
    print()
    print('>> Policy Net: channels = {}, layers = [{}]'.format(channels, ','.join(map(str, layers))))
    print('>> Training: batch_size = {}, learning_rate = {}, horizon = {}, epochs = {}'.format(batch_size, learning_rate, horizon, epochs))
    print()


def run(rddl, logdir, channels, layers, batch_size, learning_rate, horizon, epochs):
    rddl2tf = compile(rddl)
    planner = PolicyOptimizationPlanner(rddl2tf, channels, layers, logdir=logdir)
    planner.build(learning_rate, batch_size, horizon)
    _, logdir = planner.run(epochs)
    print()
    print(logdir)


def make_logdir(*args):
    return 'results/' + '/'.join(args) + '/'


def make_run_name(channels, layers, batch_size, learning_rate):
    return 'channels={}_layers={}_batch={}_lr={}'.format(channels, '+'.join(map(str, layers)), batch_size, learning_rate)


if __name__ == '__main__':

    rddl = parse_rddl(sys.argv[1])
    params = parse_json(sys.argv[2])

    horizon = params['horizon']
    epochs = params['epochs']
    hyperparameters = params['hyperparameters']

    domain = rddl.domain.name
    instance = rddl.instance.name
    logdir = make_logdir(domain, instance, 'horizon=' + str(horizon), 'epochs=' + str(epochs))

    params = ['channels', 'layers', 'batch_size', 'learning_rate']
    values = [hyperparameters[param] for param in params]
    values = list(itertools.product(*values))

    for i, (c, l, b, lr) in enumerate(values):

        print('>>>>>> Training ({}/{}) ...'.format(i+1, len(values)))
        run_logdir = logdir + make_run_name(c, l, b, lr)
        print_params(c, l, b, lr, horizon, epochs)
        print('>> logdir =', run_logdir)

        if os.path.isdir(run_logdir):
            continue

        start = time.time()
        run(rddl, run_logdir, c, l, b, lr, horizon, epochs)
        end = time.time()
        uptime = end - start
        print()
        print('<<<<<< Done in {:.6f} sec.'.format(uptime))
        print()

    print()
    print('tensorboard --logdir {}'.format(logdir))
