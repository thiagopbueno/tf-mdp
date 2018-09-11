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

from tfmdp.train.policy import DeepReactivePolicy

import numpy as np
import tensorflow as tf

import unittest


class TestDeepReactivePolicy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # hyper-parameters
        cls.batch_size = 64
        cls.horizon = 15

        # model
        cls.compiler = rddlgym.make('Reservoir-8', mode=rddlgym.SCG)
        cls.compiler.batch_mode_on()

        # initial state
        cls.initial_state = cls.compiler.compile_initial_state(cls.batch_size)

        # Deep Reactive Policy
        cls.layers = [128, 64]
        cls.policy = DeepReactivePolicy(cls.compiler, cls.layers)
        cls.action = cls.policy(cls.initial_state, cls.horizon)

    def test_policy_name(self):
        actual = self.policy.name
        expected = 'drp-fc-layers={}'.format('+'.join(map(str, self.layers)))
        self.assertEqual(actual, expected)

    def test_state_inputs(self):
        self.assertIsInstance(self.policy.state_inputs, tuple)
        self.assertEqual(len(self.policy.state_inputs), len(self.initial_state))
        for layer, fluent in zip(self.policy.state_inputs, self.initial_state):
            self.assertIsInstance(layer, tf.Tensor)
            self.assertEqual(layer.dtype, fluent.dtype)
            self.assertEqual(len(layer.shape), 2)
            self.assertEqual(layer.shape.as_list()[0], self.batch_size)
            self.assertEqual(np.prod(layer.shape.as_list()), np.prod(fluent.shape.as_list()))

    def test_input_layer(self):
        self.assertIsInstance(self.policy.input_layer, tf.Tensor)

        sizes = list(np.prod(fluent.shape.as_list()) / self.batch_size for fluent in self.initial_state)
        total_size = sum(sizes)
        shape = [self.batch_size, total_size]
        self.assertListEqual(self.policy.input_layer.shape.as_list(), shape)

        with self.compiler.graph.as_default():
            state_fluents = self.compiler.state_fluent_ordering
            state_size = self.compiler.state_size
            for name, shape in zip(state_fluents, state_size):
                name = name.replace('/', '-')

                # layer norm
                beta = 'policy/input/{}/LayerNorm/beta'.format(name)
                beta = tf.trainable_variables(beta)
                self.assertEqual(len(beta), 1)
                self.assertListEqual(beta[0].shape.as_list(), list(shape))

                gamma = 'policy/input/{}/LayerNorm/gamma'.format(name)
                gamma = tf.trainable_variables(gamma)
                self.assertEqual(len(gamma), 1)
                self.assertListEqual(gamma[0].shape.as_list(), list(shape))

    def test_hidden_layers(self):
        self.assertIsInstance(self.policy.hidden, tuple)
        self.assertEqual(len(self.policy.hidden), len(self.layers) + 1)

        with self.compiler.graph.as_default():
            for l, units in enumerate(self.layers):
                vars = tf.trainable_variables('policy/hidden{}'.format(l+1))
                self.assertEqual(len(vars), 2)

                kernel = 'policy/hidden{}/dense/kernel'.format(l+1)
                vars = tf.trainable_variables(kernel)
                self.assertEqual(len(vars), 1)
                self.assertEqual(vars[0].shape[1], units)

                bias = 'policy/hidden{}/dense/bias'.format(l+1)
                vars = tf.trainable_variables(bias)
                self.assertEqual(len(vars), 1)
                self.assertEqual(vars[0].shape[0], units)

    def test_output_layer(self):
        action_fluents = self.compiler.action_fluent_ordering
        action_size = self.compiler.action_size
        self.assertIsInstance(self.policy.output_layer, tuple)
        self.assertEqual(len(self.policy.output_layer), len(action_fluents))

        with self.compiler.graph.as_default():

            for name, shape, layer in zip(action_fluents, action_size, self.policy.output_layer):
                self.assertIsInstance(layer, tf.Tensor)
                self.assertEqual(layer.dtype, tf.float32)
                self.assertEqual(len(layer.shape), 2)
                self.assertEqual(layer.shape[0], self.batch_size)
                self.assertEqual(layer.shape[1], np.prod(shape))

                name = name.replace('/', '-')
                vars = tf.trainable_variables('policy/output/{}'.format(name))
                self.assertEqual(len(vars), 2)

                kernel = 'policy/output/{}/dense/kernel'.format(name)
                vars = tf.trainable_variables(kernel)
                self.assertEqual(len(vars), 1)

                bias = 'policy/output/{}/dense/bias'.format(name)
                vars = tf.trainable_variables(bias)
                self.assertEqual(len(vars), 1)

    def test_action_outputs(self):
        action_fluents = self.compiler.action_fluent_ordering
        action_size = self.compiler.action_size
        action_dtype = self.compiler.action_dtype
        self.assertIsInstance(self.policy.action_outputs, tuple)
        self.assertEqual(len(self.policy.action_outputs), len(action_fluents))
        for name, shape, dtype, tensor in zip(action_fluents, action_size, action_dtype, self.policy.action_outputs):
            self.assertIsInstance(tensor, tf.Tensor)
            self.assertEqual(tensor.dtype, dtype)
            self.assertListEqual(tensor.shape.as_list(), [self.batch_size] + list(shape))
