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

from tfmdp.policy.layers.action_layer import ActionLayer

import numpy as np
import tensorflow as tf
import unittest


class TestActionLayer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # hyper-parameters
        cls.batch_size = 16
        cls.horizon = 15

        # model
        cls.compiler = rddlgym.make('Reservoir-8', mode=rddlgym.SCG)
        cls.compiler.batch_mode_on()

        # initial state
        cls.initial_state = cls.compiler.compile_initial_state(cls.batch_size)

        # default action
        cls.default_action = cls.compiler.compile_default_action(cls.batch_size)

    def setUp(self):
        with self.compiler.graph.as_default():
            self.layer = ActionLayer(self.compiler.rddl.action_size)

    def test_trainable_variables(self):
        self.assertIsInstance(self.layer.trainable_variables, list)
        self.assertEqual(len(self.layer.trainable_variables), 0)
        with self.compiler.graph.as_default():
            h = tf.ones((self.batch_size, 64))
            _ = self.layer(h)
        self.assertEqual(len(self.layer.trainable_variables), 2 * len(self.default_action))

    def test_logits(self):
        self.assertEqual(len(self.layer.logits), len(self.default_action))
        for logit, size in zip(self.layer.logits, self.compiler.rddl.action_size):
            self.assertIsInstance(logit, tf.layers.Dense)
            self.assertEqual(logit.activation, None)
            self.assertEqual(logit.units, np.prod(size))

    def test_call(self):
        with self.compiler.graph.as_default():
            h = tf.ones((self.batch_size, 64))

            output1 = self.layer(h)
            self.assertIsInstance(output1, tuple)
            for layer, action_size in zip(output1, self.compiler.rddl.action_size):
                self.assertEqual(layer.shape[0], self.batch_size)
                self.assertListEqual(layer.shape.as_list()[1:], list(action_size))
