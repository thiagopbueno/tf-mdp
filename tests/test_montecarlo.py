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

from tfmdp.policy.feedforward import FeedforwardPolicy
from tfmdp.model.cell.basic_cell import BasicMarkovCell
from tfmdp.model.sequential.mrm import Trajectory
from tfmdp.model.sequential.montecarlo import MonteCarloSampling

import tensorflow as tf
import unittest


class TestMonteCarloSampling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # hyper-parameters
        cls.horizon = 40
        cls.batch_size = 128

        # rddl
        cls.compiler = rddlgym.make('Reservoir-8', rddlgym.SCG)
        cls.compiler.batch_mode_on()

        # initial state
        cls.initial_state = cls.compiler.compile_initial_state(cls.batch_size)

        # default action
        cls.default_action = cls.compiler.compile_default_action(cls.batch_size)

        # policy
        cls.policy = FeedforwardPolicy(cls.compiler, {'layers': [32, 32], 'activation': 'elu', 'input_layer_norm': True})
        cls.policy.build()

        # model
        cls.config = {}
        cls.model = MonteCarloSampling(cls.compiler, cls.config)
        cls.model.build(cls.policy)

    def test_build(self):
        self.assertIsInstance(self.model.cell, BasicMarkovCell)
        self.assertEqual(self.model.cell.policy, self.policy)

    def test_call(self):
        output = self.model(self.initial_state, self.horizon)
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 3)

        trajectory, final_state, total_reward = output

        self.assertIsInstance(trajectory, Trajectory)

        for tensor, fluent in zip(trajectory.states, self.initial_state):
            self.assertEqual(int(tensor.shape[0]), self.batch_size)
            self.assertEqual(int(tensor.shape[1]), self.horizon)
            self.assertListEqual(tensor.shape.as_list()[2:], fluent.shape.as_list()[1:])

        for tensor, fluent in zip(trajectory.actions, self.default_action):
            self.assertEqual(int(tensor.shape[0]), self.batch_size)
            self.assertEqual(int(tensor.shape[1]), self.horizon)
            self.assertListEqual(tensor.shape.as_list()[2:], fluent.shape.as_list()[1:])

        self.assertIsInstance(final_state, tuple)
        self.assertEqual(len(final_state), len(self.initial_state))

        self.assertIsInstance(total_reward, tf.Tensor)
        self.assertEqual(total_reward.dtype, tf.float32)
        self.assertListEqual(total_reward.shape.as_list(), [self.batch_size])
