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


from tfmdp.train.policy import DeepReactivePolicy

from rddl2tf.compiler import Compiler
from tfrddlsim.simulation.policy_simulator import PolicySimulator

import sys
import numpy as np
import tensorflow as tf

from typing import List, Optional, Sequence


class PolicyOptimizer(object):

    def __init__(self,
            compiler: Compiler,
            policy: DeepReactivePolicy,
            logdir: Optional[str] = None) -> None:
        self._compiler = compiler
        self._policy = policy
        self._logdir = logdir if logdir is not None else '/tmp'

    @property
    def graph(self) -> tf.Graph:
        '''Returns the compiler's graph.'''
        return self._compiler.graph

    def build(self, learning_rate: float, batch_size: int, horizon: int) -> None:
        with self.graph.as_default():
            with tf.name_scope('policy_optimizer'):
                self._build_trajectory_graph(horizon, batch_size)
                self._build_loss_graph()
                self._build_optimization_graph(learning_rate)
                self._build_summary_graph()

    def run(self, epochs: int, show_progress: bool = True) -> None:

        with tf.Session(graph=self.graph) as sess:

            self._train_writer = tf.summary.FileWriter(self._logdir + '/train')
            self._test_writer = tf.summary.FileWriter(self._logdir + '/test')

            sess.run(tf.global_variables_initializer())

            reward = -sys.maxsize
            losses = []
            rewards = []

            for step in range(epochs):
                _, loss_, reward_ = sess.run([self._train_op, self.loss, self.avg_total_reward])

                summary_ = sess.run(self._merged)
                self._train_writer.add_summary(summary_, step)

                if reward_ > reward:
                    reward = reward_
                    rewards.append((step, reward_))
                    losses.append((step, loss_))
                    self._test_writer.add_summary(summary_, step)
                    self._policy.save(sess)

                if show_progress:
                    print('Epoch {0:5}: loss = {1:3.6f}\r'.format(step, loss_), end='')

            print()
            print('rewards =', rewards)

            return losses, rewards

    def _build_trajectory_graph(self, horizon: int, batch_size: int) -> None:
        '''Builds the (state, action, interm, reward) trajectory ops.'''
        simulator = PolicySimulator(self._compiler, self._policy, batch_size)
        trajectories = simulator.trajectory(horizon)
        self.initial_state = trajectories[0]
        self.states = trajectories[1]
        self.actions = trajectories[2]
        self.rewards = trajectories[4]

    def _build_loss_graph(self) -> None:
        '''Builds the loss ops.'''
        self.total_reward = tf.squeeze(tf.reduce_sum(self.rewards, axis=1))
        self.avg_total_reward, self.variance_total_reward = tf.nn.moments(self.total_reward, axes=[0])
        self.stddev_total_reward = tf.sqrt(self.variance_total_reward)
        self.max_total_reward = tf.reduce_max(self.total_reward)
        self.min_total_reward = tf.reduce_min(self.total_reward)
        # self.loss = -self.avg_total_reward
        self.loss = tf.square(self.avg_total_reward)

    def _build_optimization_graph(self, learning_rate: float) -> None:
        '''Builds the training ops.'''
        self._optimizer = tf.train.RMSPropOptimizer(learning_rate)
        self._train_op = self._optimizer.minimize(self.loss)

    def _build_summary_graph(self):
        '''Builds the summary ops.'''
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('avg_total_reward', self.avg_total_reward)
        tf.summary.scalar('stddev_total_reward', self.stddev_total_reward)
        tf.summary.scalar('max_total_reward', self.max_total_reward)
        tf.summary.scalar('min_total_reward', self.min_total_reward)
        tf.summary.histogram('total_reward', self.total_reward)
        self._merged = tf.summary.merge_all()
