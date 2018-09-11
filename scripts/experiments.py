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


import rddlgym

from tfmdp.planner import PolicyOptimizationPlanner

import itertools
import os
import sys
import time


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def parse_json(path):
    import json
    params = read_file(path)
    return json.loads(params)


def print_params(layers, batch_size, learning_rate, horizon, epochs):
    print()
    print('>> Policy Net: layers = [{}]'.format(','.join(map(str, layers))))
    print('>> Training: batch_size = {}, learning_rate = {}, horizon = {}, epochs = {}'.format(batch_size, learning_rate, horizon, epochs))
    print()


def run(model_id, logdir, layers, batch_size, learning_rate, horizon, epochs):
    compiler = rddlgym.make(model_id, mode=rddlgym.SCG)
    compiler.batch_mode_on()
    layernorm = True
    planner = PolicyOptimizationPlanner(compiler, layers, layernorm, logdir=logdir)
    planner.build(learning_rate, batch_size, horizon)
    _, logdir = planner.run(epochs)
    print()
    print(logdir)


def make_logdir(*args):
    return 'results/' + '/'.join(args) + '/'


def make_run_name(layers, batch_size, learning_rate):
    return 'layers={}_batch={}_lr={}'.format('+'.join(map(str, layers)), batch_size, learning_rate)


if __name__ == '__main__':

    model_id = sys.argv[1]
    params = parse_json(sys.argv[2])
    output = sys.argv[3]

    rddl = rddlgym.make(model_id, mode=rddlgym.AST)
    domain = rddl.domain.name
    instance = rddl.instance.name

    horizon = params['horizon']
    epochs = params['epochs']
    hyperparameters = params['hyperparameters']

    logdir = make_logdir(domain, instance, 'horizon=' + str(horizon), 'epochs=' + str(epochs))

    params = ['layers', 'batch_size', 'learning_rate']
    values = [hyperparameters[param] for param in params]
    values = list(itertools.product(*values))

    for i, (l, b, lr) in enumerate(values):

        print('>>>>>> Training ({}/{}) ...'.format(i+1, len(values)))
        run_logdir = logdir + make_run_name(l, b, lr)
        print_params(l, b, lr, horizon, epochs)
        print('>> logdir =', run_logdir)

        if os.path.isdir(run_logdir):
            continue

        start = time.time()
        run(rddl, run_logdir, l, b, lr, horizon, epochs)
        end = time.time()
        uptime = end - start
        print()
        print('<<<<<< Done in {:.6f} sec.'.format(uptime))
        print()

    print()
    print('tensorboard --logdir {}'.format(logdir))
