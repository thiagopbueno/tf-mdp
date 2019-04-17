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
# from tfmdp.model.sequential.montecarlo import MonteCarloSampling
from tfmdp.model.sequential.reparameterization import ReparameterizationSampling

from  tfmdp.train.losses import loss_fn
from tfmdp.train.optimizers import optimizers
# from tfmdp.train.callbacks import Callback
from tfmdp.planning.planner import PolicyOptimizationPlanner

import sys
import tensorflow as tf

from typing import Callable, Dict, List, Optional, Sequence, Tuple

Callback = Callable[[None],None]
Callbacks = Dict[str, Sequence[Callback]]


class MinimaxOptimizationPlanner(PolicyOptimizationPlanner):
    '''MinimaxOptimizationPlanner learns a robust policy by bilevel optimization
    over the policy parameters and the reparameterization noise variables.

    The inner optimization level maximizes the average cost wrt the inputs
    noise variables (while the policy is fixed) in order to find low-return
    trajectories.

    The outter optimization level minimizes the average cost wrt to the policy
    parameters (while the noise variables are fixed).

    Under the hood, it leverages pathwise derivative (PD) gradient estimates
    for both optimization levels.

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler.
        config (Dict): The planner configuration parameters.
    '''

    def __init__(self, compiler: rddl2tf.compiler.Compiler, config: Dict) -> None:
        super(MinimaxOptimizationPlanner, self).__init__(compiler, config)

        self.horizon = config['horizon']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.regularization_rate = config['regularization_rate']
        self.logdir = config.get('logdir', '/tmp/tfmdp/')

    def build(self, policy: DeepReactivePolicy,
                    loss: str,
                    optimizer: str) -> None:
        '''Builds the PD planner for the given `loss` function and gradient `optimizer`.

        Args:
            policy (:obj:`tfmdp.policy.drp.DeepReactivePolicy`): A deep reactive policy.
            loss (str): A differentiable loss function used to train the policy.
            optimizer (str): A gradient descent optimizer.
        '''
        self.policy = policy
        self.loss = loss_fn[loss]
        self.optimizer = optimizers[optimizer]

        self.initial_state = self.compiler.compile_initial_state(self.batch_size)

        with self.compiler.graph.as_default():
            self._build_model_ops()
            self._build_loss_ops()
            self._build_regularization_loss_ops()
            self._build_optimizer_ops()
            self._build_summary_ops()

    def _build_model_ops(self):
        with tf.name_scope('model'):
            self.model = ReparameterizationSampling(self.compiler, config={})
            self.model.build(self.policy)
            output = self.model(self.initial_state, self.horizon)
            self.trajetory, self.final_state, self.total_reward = output

    def _build_loss_ops(self):
        with tf.name_scope('loss'):
            self.avg_total_reward = tf.reduce_mean(self.total_reward, name='avg_total_reward')
            self.loss = self.loss(self.avg_total_reward)

    def _build_regularization_loss_ops(self):
        with tf.name_scope('regularization'):
            self.regularization_loss = []

            for variables, dists in zip(self.model.noise_map, self.model.reparameterization_map):

                noises = variables[1]
                if noises is None:
                    continue

                for noise, dist in zip(noises, dists[1]):
                    dist = dist[0]
                    log_probs = dist.log_prob(noise)
                    log_prob_per_batch = tf.reduce_sum(log_probs, axis=[1, 2])
                    log_prob = tf.reduce_mean(log_prob_per_batch)
                    self.regularization_loss.append(log_prob)

            self.regularization_loss = sum(self.regularization_loss)

    def _build_optimizer_ops(self):
        with tf.name_scope('optimizer'):
            self.optimizer = self.optimizer(self.learning_rate)

            with tf.name_scope('outter'):
                self.policy_variables = self.policy.trainable_variables
                self.outter_loss = self.loss
                self.outter_train_op = self.optimizer.minimize(self.outter_loss, var_list=self.policy_variables)

            with tf.name_scope('inner'):
                self.noise_variables = self.model.trainable_variables
                self.inner_loss = -self.loss + self.regularization_rate * self.regularization_loss
                self.inner_train_op = self.optimizer.minimize(self.inner_loss, var_list=self.noise_variables)

    def _build_summary_ops(self):
        tf.summary.histogram('total_reward', self.total_reward)
        tf.summary.scalar('avg_total_reward', self.avg_total_reward)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('regularization_loss', self.regularization_loss)
        self.summary = tf.summary.merge_all()

    def run(self, epochs: Tuple[int, int],
                  callbacks: Optional[Callbacks] = None,
                  show_progress: Optional[bool] = True) -> None:
        '''Runs the policy optimizer for a given number of `epochs`.

        Optionally, it executes `callbacks` to extend planning behavior
        during training.

        Args:
            epochs (int): The number of training epochs.
            callbacks (Optional[Dict[str, List[Callback]]]): Mapping from events to lists of callables.
        '''
        outter_epochs, inner_epochs = epochs

        with tf.Session(graph=self.compiler.graph) as sess:
            writer = tf.summary.FileWriter(self.logdir, sess.graph)

            sess.run(tf.global_variables_initializer())

            for outter_step in range(outter_epochs):

                for inner_step in range(inner_epochs):
                    _, loss_ = sess.run([self.inner_train_op, self.loss])

                    summary_ = sess.run(self.summary)
                    writer.add_summary(summary_)

                    if show_progress:
                        print('(inner)  Epoch {0:5}: loss = {1:3.6f}\r'.format(inner_step, loss_), end='')

                _, loss_ = sess.run([self.outter_train_op, self.loss])

                summary_ = sess.run(self.summary)
                writer.add_summary(summary_)

                if show_progress:
                    print('\n(outter) Epoch {0:5}: loss = {1:3.6f}'.format(outter_step, loss_))

    def summary(self) -> None:
        '''Prints a string summary of the planner.'''
        raise NotImplementedError
