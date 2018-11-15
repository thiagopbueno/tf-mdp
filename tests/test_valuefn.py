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

from rddl2tf.compiler import Compiler

from tfmdp.train.policy import DeepReactivePolicy
from tfmdp.train.valuefn import Value

import numpy as np
import tensorflow as tf

import unittest

class TestValueFn(unittest.TestCase):

    def setUp(self):
        self.rddl1 = rddlgym.make('Navigation-v3', mode=rddlgym.AST)
        self.compiler1 = Compiler(self.rddl1, batch_mode=True)

        self.layers = [64]
        self.policy1 = DeepReactivePolicy(self.compiler1, self.layers, tf.nn.elu, input_layer_norm=False)

        self.valuefn1 = Value(self.compiler1, self.policy1)

    def test_build_trajectory_graph(self):
        horizon = 40
        batch_size = 128
        self.valuefn1.build(horizon, batch_size)

        states = self.valuefn1._states
        state_size = self.compiler1.state_size
        self.assertIsInstance(states, tuple)
        self.assertEqual(len(states), len(state_size))
        for fluent, size in zip(states, state_size):
            self.assertIsInstance(fluent, tf.Tensor)
            self.assertListEqual(fluent.shape.as_list(), [batch_size, horizon] + list(size))

        timesteps = self.valuefn1._timesteps
        self.assertIsInstance(timesteps, tf.Tensor)
        self.assertListEqual(timesteps.shape.as_list(), [batch_size, horizon])

    def test_build_value_estimates_graph(self):
        horizon = 40
        batch_size = 128
        self.valuefn1.build(horizon, batch_size)

        estimates = self.valuefn1._estimates
        self.assertIsInstance(estimates, tf.Tensor)
        self.assertListEqual(estimates.shape.as_list(), [batch_size, horizon])

    def test_regression_graph(self):
        horizon = 40
        batch_size = 128
        self.valuefn1.build(horizon, batch_size)

        features = self.valuefn1._dataset_features
        self.assertIsInstance(features, tuple)
        self.assertEqual(len(features), 2)

        timesteps, states = features
        self.assertIsInstance(timesteps, tf.Tensor)
        self.assertListEqual(timesteps.shape.as_list(), [batch_size * horizon])

        state_size = self.compiler1.state_size
        self.assertIsInstance(states, tuple)
        self.assertEqual(len(states), len(state_size))
        for fluent, size in zip(states, state_size):
            self.assertIsInstance(fluent, tf.Tensor)
            self.assertListEqual(fluent.shape.as_list(), [batch_size * horizon] + list(size))

        targets = self.valuefn1._dataset_targets
        self.assertIsInstance(targets, tf.Tensor)
        self.assertListEqual(targets.shape.as_list(), [batch_size * horizon])

    def test_prediction_graph(self):
        horizon = 40
        batch_size = 128
        self.valuefn1.build(horizon, batch_size)

        predictions = self.valuefn1._predictions
        self.assertIsInstance(predictions, tf.Tensor)
        self.assertListEqual(predictions.shape.as_list(), [None])

    def test_loss_graph(self):
        horizon = 40
        batch_size = 128
        self.valuefn1.build(horizon, batch_size)

        loss = self.valuefn1._loss

    def test_optimization_graph(self):
        horizon = 20
        batch_size = 4
        self.valuefn1.build(horizon, batch_size)

        optimizer = self.valuefn1._optimizer
        grad_and_vars = self.valuefn1._grad_and_vars
        train_op = self.valuefn1._train_op

    def test_training_batch(self):
        horizon = 5
        batch_size = 4
        self.valuefn1.build(horizon, batch_size)

        with tf.Session(graph=self.compiler1.graph) as sess:
            sess.run(tf.global_variables_initializer())

            training_batch = 2
            dataset = self.valuefn1._regression_dataset(sess)
            batchs = self.valuefn1._training_batch_generator(training_batch, dataset)

            state_size = self.compiler1.state_size

            n = 0
            for state, timestep, target in batchs:
                self.assertIsInstance(state, tuple)
                self.assertEqual(len(state), len(state_size))

                for fluent, size in zip(state, state_size):
                    self.assertIsInstance(fluent, np.ndarray)
                    self.assertListEqual(list(fluent.shape), [training_batch] + list(size))

                self.assertIsInstance(timestep, np.ndarray)
                self.assertListEqual(list(timestep.shape), [training_batch])

                self.assertIsInstance(target, np.ndarray)
                self.assertListEqual(list(target.shape), [training_batch])

                n += 1

            self.assertEqual(n, int(batch_size * horizon / training_batch))

    def test_value_fn_fitting(self):
        horizon = 40
        batch_size = 128
        self.valuefn1.build(horizon, batch_size)

        with tf.Session(graph=self.compiler1.graph) as sess:
            sess.run(tf.global_variables_initializer())

            epochs = 20
            batch_size = 64
            loss = self.valuefn1.fit(sess, batch_size, epochs, show_progress=False)
            self.assertIsInstance(loss, list)

