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


import rddl2tf.compiler

from tfmdp.policy.drp import DeepReactivePolicy
from tfmdp.model.sequential.montecarlo import MonteCarloSampling

from  tfmdp.train.losses import loss_fn
from tfmdp.train.optimizers import optimizers
# from tfmdp.train.callbacks import Callback
from tfmdp.planning.planner import PolicyOptimizationPlanner

import sys
import tensorflow as tf

from typing import Callable, Dict, List, Optional, Sequence

Callback = Callable[[None],None]
Callbacks = Dict[str, Sequence[Callback]]


class PathwiseOptimizationPlanner(PolicyOptimizationPlanner):
    '''PathwiseOptimizationPlanner leverages pathwise derivative (PD)
    gradient estimates to optimize the policy parameters for a given horizon,
    batch size, and learning rate.

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler.
        config (Dict): The planner configuration parameters.
    '''

    def __init__(self, compiler: rddl2tf.compiler.Compiler, config: Dict) -> None:
        super(PathwiseOptimizationPlanner, self).__init__(compiler, config)

        self.horizon = config['horizon']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']

    def build(self, policy: DeepReactivePolicy,
                    loss: str,
                    optimizer: str) -> None:
        '''Builds the PD planner for the given `loss` function and gradient `optimizer`.

        Args:
            policy (:obj:`tfmdp.policy.drp.DeepReactivePolicy`): A deep reactive policy.
            loss (str): A differentiable loss function used to train the policy.
            optimizer (str): A gradient descent optimizer.
        '''
        self.initial_state = self.compiler.compile_initial_state(self.batch_size)

        with self.compiler.graph.as_default():

            with tf.name_scope('model'):
                self.model = MonteCarloSampling(self.compiler)
                self.model.build(policy)
                self.trajetory, self.final_state, self.total_reward = self.model(self.initial_state,
                                                                                 self.horizon)

            with tf.name_scope('loss'):
                self.avg_total_reward = tf.reduce_mean(self.total_reward, name='avg_total_reward')
                self.loss = loss_fn[loss](self.avg_total_reward)

            with tf.name_scope('optimizer'):
                self.optimizer = optimizers[optimizer](self.learning_rate)
                self.train_op = self.optimizer.minimize(self.loss)

    def run(self, epochs: int,
                  callbacks: Optional[Callbacks] = None,
                  show_progress: Optional[bool] = True) -> None:
        '''Runs the policy optimizer for a given number of `epochs`.

        Optionally, it executes `callbacks` to extend planning behavior
        during training.

        Args:
            epochs (int): The number of training epochs.
            callbacks (Optional[Dict[str, List[Callback]]]): Mapping from events to lists of callables.
        '''
        with tf.Session(graph=self.compiler.graph) as sess:
            sess.run(tf.global_variables_initializer())

            reward = -sys.maxsize
            losses, rewards = [], []

            for step in range(epochs):
                _, loss_, reward_ = sess.run([self.train_op, self.loss, self.avg_total_reward])

                if reward_ > reward:
                    reward = reward_
                    rewards.append((step, reward_))
                    losses.append((step, loss_))

                if show_progress:
                    print('Epoch {0:5}: loss = {1:3.6f}\r'.format(step, loss_), end='')

            return losses, rewards

    def summary(self) -> None:
        '''Prints a string summary of the planner.'''
        raise NotImplementedError
