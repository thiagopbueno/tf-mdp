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

from tfmdp.train.policy import DeepReactivePolicy
from tfmdp.train.optimizer import PolicyOptimizer

import numpy as np
import tensorflow as tf

import unittest


class TestPolicyOptimizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # hyper-parameters
        cls.learning_rate = 0.0001
        cls.epochs = 32
        cls.batch_size = 64
        cls.horizon = 20

        # RDDL
        with open('rddl/Reservoir.rddl', mode='r') as file:
            RESERVOIR = file.read()

        # parser
        parser = RDDLParser()
        parser.build()
        rddl = parser.parse(RESERVOIR)

        # compiler
        cls.compiler = Compiler(rddl, batch_mode=True)
        cls.initial_state = cls.compiler.compile_initial_state(cls.batch_size)

        # policy
        cls.channels = 4
        cls.layers = [64, 32, 16]
        cls.policy = DeepReactivePolicy(cls.compiler, cls.channels, cls.layers)

        # optimizer
        cls.optimizer = PolicyOptimizer(cls.compiler, cls.policy)
        cls.optimizer.build(cls.learning_rate, cls.batch_size, cls.horizon)

    def test_state_trajectory(self):
        states = self.optimizer.states
        state_size = self.compiler.state_size
        self.assertIsInstance(states, tuple, 'state trajectory is factored')
        self.assertEqual(len(states), len(state_size), 'state trajectory has all states fluents')
        for fluent, fluent_size in zip(states, state_size):
            tensor_size = [self.batch_size, self.horizon] + list(fluent_size)
            self.assertIsInstance(fluent, tf.Tensor, 'state fluent is a tensor')
            self.assertListEqual(fluent.shape.as_list(), tensor_size, 'fluent size is [batch_size, horizon, state_fluent_size]')

    def test_action_trajectory(self):
        actions = self.optimizer.actions
        action_size = self.compiler.action_size
        self.assertIsInstance(actions, tuple, 'action trajectory is factored')
        self.assertEqual(len(actions), len(action_size),
            'action trajectory has all actions fluents')
        for fluent, fluent_size in zip(actions, action_size):
            tensor_size = [self.batch_size, self.horizon] + list(fluent_size)
            self.assertIsInstance(fluent, tf.Tensor, 'action fluent is a tensor')
            self.assertListEqual(fluent.shape.as_list(), tensor_size,
                'fluent size is [batch_size, horizon, action_fluent_size]')

    def test_rewards_trajectory(self):
        rewards = self.optimizer.rewards
        rewards_shape = [self.batch_size, self.horizon, 1]
        self.assertIsInstance(rewards, tf.Tensor, 'reward trajectory is a tensor')
        self.assertListEqual(rewards.shape.as_list(), rewards_shape ,
            'reward shape is [batch_size, horizon, 1]')

    def test_total_reward(self):
        total_reward = self.optimizer.total_reward
        self.assertIsInstance(total_reward, tf.Tensor, 'total reward is a tensor')
        self.assertEqual(total_reward.dtype, tf.float32, 'total reward is a real tensor')
        self.assertListEqual(total_reward.shape.as_list(), [self.batch_size],
            'total reward has a scalar value for each trajectory')

    def test_avg_total_reward(self):
        avg_total_reward = self.optimizer.avg_total_reward
        self.assertIsInstance(avg_total_reward, tf.Tensor, 'average total reward is a tensor')
        self.assertEqual(avg_total_reward.dtype, tf.float32, 'average total reward is a real tensor')
        self.assertListEqual(avg_total_reward.shape.as_list(), [], 'average total reward is a scalar')

    def test_optimization_objective(self):
        loss = self.optimizer.loss
        self.assertIsInstance(loss, tf.Tensor, 'loss function is a tensor')
        self.assertEqual(loss.dtype, tf.float32, 'loss function is a real tensor')
        self.assertListEqual(loss.shape.as_list(), [], 'loss function is a scalar')

    def test_loss_optimizer(self):
        optimizer = self.optimizer._optimizer
        self.assertIsInstance(optimizer, tf.train.RMSPropOptimizer)
        train_op = self.optimizer._train_op
        self.assertEqual(train_op.name, 'policy_optimizer/RMSProp')

    def test_run(self):
        self.optimizer.run(self.epochs, show_progress=False)
