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

from tfmdp import utils
from tfmdp.policy.layers.state_layer import StateLayer
from tfmdp.policy.layers.action_layer import ActionLayer
from tfmdp.policy.drp import activation_fn
from tfmdp.policy.feedforward import FeedforwardPolicy

import numpy as np
import tensorflow as tf
import unittest


class TestFeedforwardPolicy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # hyper-parameters
        cls.batch_size = 16
        cls.horizon = 15

        # model
        cls.compiler = rddlgym.make('Reservoir-8', mode=rddlgym.SCG)
        cls.compiler.init()
        cls.compiler.batch_size = cls.batch_size

        # initial state
        cls.initial_state = cls.compiler.initial_state()

        # default action
        cls.default_action = cls.compiler.default_action()

        # policy
        cls.config = {
            'layers': [128, 64, 32],
            'activation': 'elu',
            'input_layer_norm': True
        }
        cls.policy = FeedforwardPolicy(cls.compiler, cls.config)
        cls.policy.build()

    def test_name(self):
        params_string = utils.get_params_string(self.policy.config)
        self.assertEqual(self.policy.name, 'drp-ff-{}'.format(params_string))

    def test_trainable_variables(self):
        self.assertIsInstance(self.policy.trainable_variables, list)
        self.assertEqual(len(self.policy.trainable_variables), 2 * (1 + len(self.config['layers']) + len(self.default_action)))

    def test_size(self):
        self.assertEqual(self.policy.size, sum(np.prod(var.shape.as_list()) for var in self.policy.trainable_variables))

    def test_build_input_layer(self):
        self.assertIsInstance(self.policy._input_layer, StateLayer)

    def test_build_hidden_layers(self):
        self.assertIsInstance(self.policy._hidden_layers, tuple)
        self.assertEqual(len(self.policy._hidden_layers), len(self.config['layers']))
        for layer, units in zip(self.policy._hidden_layers, self.config['layers']):
            self.assertIsInstance(layer, tf.compat.v1.layers.Dense)
            self.assertEqual(layer.units, units)
            self.assertEqual(layer.activation, activation_fn[self.config['activation']])

    def test_build_output_layer(self):
        self.assertIsInstance(self.policy._output_layer, ActionLayer)

    def test_call(self):
        action1 = self.policy(self.initial_state, self.horizon)
        with self.policy.graph.as_default():
            policy_vars1 = tf.compat.v1.trainable_variables()
        self.assertEqual(len(policy_vars1), len(self.policy.trainable_variables))

        self.assertIsInstance(action1, tuple)
        self.assertEqual(len(action1), len(self.default_action))
        for default_action_fluent, action_fluent in zip(self.default_action, action1):
            self.assertEqual(default_action_fluent.dtype, action_fluent.dtype)
            self.assertEqual(default_action_fluent.shape, action_fluent.shape)

        action2 = self.policy(self.initial_state, self.horizon)
        with self.policy.graph.as_default():
            policy_vars2 = tf.compat.v1.trainable_variables()
        self.assertEqual(len(policy_vars2), len(self.policy.trainable_variables))

        self.assertEqual(len(policy_vars1), len(policy_vars2))
