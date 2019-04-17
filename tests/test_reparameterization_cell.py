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

import rddl2tf.reparam

from tfmdp.policy.feedforward import FeedforwardPolicy
from tfmdp.model.cell.reparameterization_cell import ReparameterizationCell, OutputTuple

from tfmdp.model import utils

import tensorflow as tf
import unittest


class TestReparameterizationCell(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # hyper-parameters
        cls.horizon = 40
        cls.batch_size = 16

        # rddl
        cls.compiler = rddlgym.make('Navigation-v2', mode=rddlgym.SCG)
        cls.compiler.batch_mode_on()

        # initial state
        cls.initial_state = cls.compiler.compile_initial_state(cls.batch_size)

        # default action
        cls.default_action = cls.compiler.compile_default_action(cls.batch_size)

        # policy
        cls.policy = FeedforwardPolicy(cls.compiler, {'layers': [64, 64], 'activation': 'relu', 'input_layer_norm': True})
        cls.policy.build()

        with cls.compiler.graph.as_default():

            # reparameterization
            cls.noise_shapes = rddl2tf.reparam.get_cpfs_reparameterization(cls.compiler.rddl)
            cls.noise_variables = utils.get_noise_variables(cls.noise_shapes, cls.batch_size, cls.horizon)
            cls.noise_inputs, cls.encoding = utils.encode_noise_as_inputs(cls.noise_variables)

            # timestep
            cls.timestep = tf.constant(cls.horizon, dtype=tf.float32)
            cls.timestep = tf.expand_dims(cls.timestep, -1)
            cls.timestep = tf.stack([cls.timestep] * cls.batch_size)

            # inputs
            cls.inputs = tf.concat([cls.timestep, cls.noise_inputs[:, 0, :]], axis=1)

        # cell
        cls.config = { 'encoding': cls.encoding }
        cls.cell = ReparameterizationCell(cls.compiler, cls.policy, cls.config)

    def test_call(self):
        output, next_state = self.cell(self.inputs, self.initial_state)

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
