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
from tfmdp.model.cell.basic_cell import BasicMarkovCell, OutputTuple

from tfmdp.model.utils import to_tensor

import tensorflow as tf
import unittest


class TestBasicMarkovCell(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # hyper-parameters
        cls.horizon = 40
        cls.batch_size = 16

        # rddl
        cls.compiler = rddlgym.make('Reservoir-8', mode=rddlgym.SCG)
        cls.compiler.batch_mode_on()

        # initial state
        cls.initial_state = cls.compiler.compile_initial_state(cls.batch_size)

        # default action
        cls.default_action = cls.compiler.compile_default_action(cls.batch_size)

        # policy
        cls.policy = FeedforwardPolicy(cls.compiler, {'layers': [64, 64], 'activation': 'relu', 'input_layer_norm': True})
        cls.policy.build()

        # cell
        cls.cell = BasicMarkovCell(cls.compiler, cls.policy)

        with cls.cell.graph.as_default():
            # timestep
            cls.timestep = tf.constant(cls.horizon, dtype=tf.float32)
            cls.timestep = tf.expand_dims(cls.timestep, -1)
            cls.timestep = tf.stack([cls.timestep] * cls.batch_size)

    def test_state_size(self):
        self.assertTupleEqual(self.cell.state_size, self.compiler.rddl.state_size)

    def test_output_size(self):
        state_size = self.cell.state_size
        action_size = self.cell.action_size
        interm_size = self.cell.interm_size
        reward_size = 1
        self.assertTupleEqual(self.cell.output_size, (state_size, action_size, interm_size, reward_size))

    def test_call(self):
        output, next_state = self.cell(self.timestep, self.initial_state)

        self.assertIsInstance(output, OutputTuple)
        self.assertEqual(len(output), 4)

        self.assertEqual(output.state, output[0])
        self.assertEqual(output.action, output[1])
        self.assertEqual(output.interms, output[2])
        self.assertEqual(output.reward, output[3])

        self.assertEqual(output.state, next_state)

        for action_fluent, default_action_fluent in zip(output.action, self.default_action):
            self.assertEqual(action_fluent.shape, default_action_fluent.shape)

        self.assertListEqual(output.reward.shape.as_list(), [self.batch_size, 1])

        for fluent, next_fluent in zip(self.initial_state, next_state):
            self.assertEqual(fluent.shape, next_fluent.shape)
