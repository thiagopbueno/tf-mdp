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
from tfmdp.train.mrm import MarkovRecurrentModel, ReparameterizationType
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

        # model
        cls.compiler = rddlgym.make('Reservoir-8', mode=rddlgym.SCG)
        cls.compiler.batch_mode_on()

        # fluents
        cls.initial_state = cls.compiler.compile_initial_state(cls.batch_size)
        cls.state_fluents = cls.compiler.state_fluent_ordering
        cls.state_size = cls.compiler.state_size
        cls.action_fluents = cls.compiler.action_fluent_ordering
        cls.action_size = cls.compiler.action_size

        # policy
        cls.layers = [64, 32, 16]
        cls.policy = DeepReactivePolicy(cls.compiler, cls.layers, tf.nn.elu, input_layer_norm=True)

        # model
        cls.model = MarkovRecurrentModel(cls.compiler, cls.policy, cls.batch_size)
        cls.model.build(cls.horizon, lambda x: -x, ReparameterizationType.FULLY_REPARAMETERIZED)

        # optimizer
        cls.optimizer = PolicyOptimizer(cls.model)
        cls.optimizer.build(cls.learning_rate, cls.batch_size, cls.horizon, tf.train.RMSPropOptimizer)

    def test_policy_variables(self):

        with self.compiler.graph.as_default():

            input_layer_variables = tf.trainable_variables('trajectory/policy/input/')
            self.assertEqual(len(input_layer_variables), 2 * len(self.state_fluents))

            hidden_layer_variables = tf.trainable_variables('trajectory/policy/hidden')
            self.assertEqual(len(hidden_layer_variables), 2 * len(self.layers))

            output_layer_variables = tf.trainable_variables('trajectory/policy/output/')
            self.assertEqual(len(output_layer_variables), 2 * len(self.action_fluents))

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

    def test_optimization_run(self):
        losses, rewards = self.optimizer.run(self.epochs, show_progress=False)
        self.assertIsInstance(self.policy._checkpoint, str)
        self.assertIsInstance(losses, list)
        self.assertIsInstance(rewards, list)
        self.assertEqual(len(losses), len(rewards))
        self.assertTrue(all(rewards[i+1] > rewards[i] for i in range(len(rewards[1:]))))
