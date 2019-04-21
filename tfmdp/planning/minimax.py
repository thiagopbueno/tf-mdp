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
        self.train_batch_size = config['train_batch_size']
        self.test_batch_size = config['test_batch_size']
        self.inner_learning_rate = config['inner_learning_rate']
        self.outter_learning_rate = config['outter_learning_rate']
        self.regularization_rate = config['regularization_rate']
        self.schedule = config.get('loss_schedule')
        self.loss_rate = config.get('loss_rate')

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

        with self.compiler.graph.as_default():
            self._build_test_ops()
            self._build_train_ops()
            self._build_regularization_loss_ops()
            self._build_loss_ops()
            self._build_optimizer_ops()
            self._build_summary_ops()

    def _build_test_ops(self):
        with tf.name_scope('test'):
            self.test_initial_state = self.compiler.compile_initial_state(self.test_batch_size)
            self.test_model = MonteCarloSampling(self.compiler)
            self.test_model.build(self.policy)
            _, _, self.test_total_reward = self.test_model(self.test_initial_state, self.horizon)
            self.test_avg_total_reward = tf.reduce_mean(self.test_total_reward, name='test_avg_total_reward')
            self.test_min_total_reward = tf.reduce_min(self.test_total_reward, name='test_min_total_reward')

    def _build_train_ops(self):
        with tf.name_scope('model'):
            self.initial_state = self.compiler.compile_initial_state(self.train_batch_size)
            self.train_model = ReparameterizationSampling(self.compiler, config={})
            self.train_model.build(self.policy)
            _, _, self.train_total_reward = self.train_model(self.initial_state, self.horizon)
            self.train_avg_total_reward = tf.reduce_mean(self.train_total_reward, name='train_avg_total_reward')
            self.train_min_total_reward = tf.reduce_min(self.train_total_reward, name='train_min_total_reward')

    def _build_loss_ops(self):
        with tf.name_scope('loss'):
            self.train_loss = self.loss(self.train_avg_total_reward)
            self.test_loss = self.loss(self.test_avg_total_reward)

            with tf.name_scope('inner_loss'):
                self.inner_loss = -self.train_loss + self.regularization_rate * self.regularization_loss

            with tf.name_scope('outter_loss'):
                if self.loss_rate is None:
                    self.outter_loss = self.train_loss
                else:
                    self.outter_loss = (1 - self.loss_rate) * self.test_loss + \
                                       self.loss_rate * self.train_loss

    def _build_regularization_loss_ops(self):
        with tf.name_scope('regularization'):
            self.regularization_loss = []

            for variables, dists in zip(self.train_model.noise_map, self.train_model.reparameterization_map):

                noises = variables[1]
                if noises is None:
                    continue

                for noise, dist in zip(noises, dists[1]):
                    dist = dist[0]
                    log_probs = dist.log_prob(noise)
                    log_prob_per_batch = tf.reduce_sum(log_probs, axis=[1, 2])
                    log_prob = tf.reduce_mean(log_prob_per_batch)
                    self.regularization_loss.append(log_prob)

            self.regularization_loss = -sum(self.regularization_loss)

    def _build_optimizer_ops(self):
        with tf.name_scope('optimizer'):

            with tf.name_scope('outter'):
                self.outter_optimizer = self.optimizer(self.outter_learning_rate)
                self.policy_variables = self.policy.trainable_variables
                self.outter_grads_and_vars = self.outter_optimizer.compute_gradients(self.outter_loss, var_list=self.policy_variables)
                self.outter_train_op = self.outter_optimizer.apply_gradients(self.outter_grads_and_vars)

            with tf.name_scope('inner'):
                self.inner_optimizer = self.optimizer(self.inner_learning_rate)
                self.noise_variables = self.train_model.trainable_variables
                self.inner_grads_and_vars = self.inner_optimizer.compute_gradients(self.inner_loss, var_list=self.noise_variables)
                self.inner_train_op = self.inner_optimizer.apply_gradients(self.inner_grads_and_vars)

    def _build_summary_ops(self):
        if self.logdir is None:
            return

        with tf.name_scope('summary'):
            # test performance
            tf.summary.histogram('test_total_reward', self.test_total_reward)
            tf.summary.scalar('test_avg_total_reward', self.test_avg_total_reward)
            tf.summary.scalar('test_min_total_reward', self.test_min_total_reward)

            # train
            tf.summary.scalar('train_loss', self.train_loss)
            tf.summary.scalar('test_loss', self.test_loss)
            tf.summary.scalar('regularization_loss', self.regularization_loss)
            tf.summary.scalar('self.inner_loss', self.inner_loss)
            tf.summary.scalar('self.outter_loss', self.outter_loss)

            # Noise gradients and variables
            for (grad, var) in self.inner_grads_and_vars:
                tf.summary.histogram(var.name, var)
                tf.summary.histogram(grad.name, grad)
                tf.summary.scalar('{}_norm'.format(grad.name), tf.norm(grad))

            # Policy gradients and variables
            for (grad, var) in self.outter_grads_and_vars:
                tf.summary.histogram(var.name, var)
                tf.summary.histogram(grad.name, grad)
                tf.summary.scalar('{}_norm'.format(grad.name), tf.norm(grad))

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

            if self.logdir:
                writer = tf.summary.FileWriter(self.logdir, sess.graph)

            sess.run(tf.global_variables_initializer())

            reward = -sys.maxsize
            rewards = []

            global_step = 0
            for outter_step in range(outter_epochs):

                for noise_var in self.noise_variables:
                    sess.run(noise_var.initializer)

                for inner_step in range(inner_epochs):
                    _ = sess.run(self.inner_train_op)
                    train_loss_, regularization_loss_ = sess.run([self.train_loss, self.regularization_loss])
                    global_step += 1

                    if self.logdir:
                        summary_ = sess.run(self.summary)
                        writer.add_summary(summary_, global_step)

                    if show_progress:
                        print('(inner)  Epoch {:5}: loss = {:3.4f}, reg loss = {:3.4f}\r'.format(
                            inner_step, train_loss_, regularization_loss_), end='')

                _ = sess.run(self.outter_train_op)
                train_loss_, test_loss_ = sess.run([self.train_loss, self.test_loss])
                global_step += 1

                if self.logdir:
                    summary_ = sess.run(self.summary)
                    writer.add_summary(summary_, global_step)

                if show_progress:
                    print('\n(outter) Epoch {:5}: train_loss = {:3.4f}, test_loss = {:3.4f}'.format(outter_step, train_loss_, test_loss_))

                test_avg_total_reward_ = sess.run(self.test_avg_total_reward)
                if test_avg_total_reward_ > reward:
                    reward = test_avg_total_reward_
                    rewards.append((outter_step, test_avg_total_reward_))

                    if self.output:
                        ckpt = self.policy.save(sess, self.output)

            return rewards

    def summary(self) -> None:
        '''Prints a string summary of the planner.'''
        raise NotImplementedError
