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

import numpy as np
import itertools
import os
import sys
import time
import json


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def parse_json(path):
    import json
    params = read_file(path)
    return json.loads(params)


def print_params(layers, activation, batch_size, learning_rate, horizon, epochs):
    print()
    print('>> Policy Net: layers = [{}], activation = {}'.format(','.join(map(str, layers)), activation))
    print('>> Training: batch_size = {}, learning_rate = {}, horizon = {}, epochs = {}'.format(batch_size, learning_rate, horizon, epochs))
    print()


def run(model_id, logdir, layers, activation, batch_size, learning_rate, horizon, epochs):
    compiler = rddlgym.make(model_id, mode=rddlgym.SCG)
    compiler.batch_mode_on()
    input_layer_norm = True
    hidden_layer_norm = False
    planner = PolicyOptimizationPlanner(compiler, layers, activation, input_layer_norm, hidden_layer_norm, logdir=logdir)
    planner.build(learning_rate, batch_size, horizon)
    rewards, policy, _ = planner.run(epochs)
    return rewards, policy


def log_performance(filename, results):
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(MyEncoder, self).default(obj)
    with open(filename, 'w') as file:
        file.write(json.dumps(results, cls=MyEncoder, indent=4))


def make_logdir(*args):
    return 'results/' + '/'.join(args) + '/'


def make_run_name(layers, activation, batch_size, learning_rate):
    return 'layers={}_act={}_batch={}_lr={}'.format('+'.join(map(str, layers)), activation, batch_size, learning_rate)


if __name__ == '__main__':

    model_id = sys.argv[1]
    params = parse_json(sys.argv[2])
    output = sys.argv[3]
    N = int(sys.argv[4])

    rddl = rddlgym.make(model_id, mode=rddlgym.AST)
    domain = rddl.domain.name
    instance = rddl.instance.name

    horizon = params['horizon']
    epochs = params['epochs']
    hyperparameters = params['hyperparameters']

    logdir = make_logdir(domain, instance, 'horizon=' + str(horizon), 'epochs=' + str(epochs))

    params = ['layers', 'activation', 'batch_size', 'learning_rate']
    values = [hyperparameters[param] for param in params]
    values = list(itertools.product(*values))

    results = {}

    for i, (layers, activation, batch_size, learning_rate) in enumerate(values):

        print('>>>>>> Training ({}/{}) ...'.format(i+1, len(values)))
        test_name = make_run_name(layers, activation, batch_size, learning_rate)
        test_logdir = logdir + test_name
        print_params(layers, activation, batch_size, learning_rate, horizon, epochs)

        if os.path.isdir(test_logdir):
            continue

        rewards = []
        trainings = []
        uptimes = []
        for j in range(N):
            print('>>> Iteration ({}/{})'.format(j+1, N))

            run_logdir = test_logdir + '/run{}'.format(j+1)
            print('>> logdir =', run_logdir)

            start = time.time()
            training, policy = run(model_id, run_logdir, layers, activation, batch_size, learning_rate, horizon, epochs)
            end = time.time()
            uptime = end - start
            uptimes.append(uptime)

            print()
            print('<<<<<< Done in {:.6f} sec.'.format(uptime))
            print()

            trainings.append(training)
            rewards.append(training[-1][1])

        results[test_name] = {
            'avg': np.mean(rewards),
            'stddev': np.std(rewards),
            'avg_time': np.mean(uptimes),
            'stddev_time': np.std(uptimes),
            'trainings': trainings
        }

    log_performance(output, results)

    print()
    print('tensorboard --logdir {}'.format(logdir))
