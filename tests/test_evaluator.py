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
from tfmdp.train.optimizer import PolicyOptimizer
from tfmdp.test.evaluator import PolicyEvaluator

import numpy as np
import tensorflow as tf

import unittest


class TestPolicyOptimizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # hyper-parameters
        cls.learning_rate = 0.0001
        cls.epochs = 20
        cls.batch_size = 64
        cls.horizon = 20

        # model
        cls.compiler = rddlgym.make('Reservoir-8', mode=rddlgym.SCG)
        cls.compiler.batch_mode_on()

        # fluents
        cls.initial_state = cls.compiler.compile_initial_state(cls.batch_size)
        cls.state_fluents = cls.compiler.state_fluent_ordering
        cls.action_fluents = cls.compiler.action_fluent_ordering

        # policy
        cls.layers = [64, 32, 16]
        cls.policy = DeepReactivePolicy(cls.compiler, cls.layers)

        # optimizer
        cls.optimizer = PolicyOptimizer(cls.compiler, cls.policy)
        cls.optimizer.build(cls.learning_rate, cls.batch_size, cls.horizon)
        cls.optimizer.run(cls.epochs, show_progress=False)

        # evaluator
        cls.evaluator = PolicyEvaluator(cls.compiler, cls.policy)

    def test_policy_variables(self):

        with self.compiler.graph.as_default():

            input_layer_variables = tf.trainable_variables('trajectory/policy/input/')
            self.assertEqual(len(input_layer_variables), 2 * len(self.state_fluents))

            hidden_layer_variables = tf.trainable_variables('trajectory/policy/hidden')
            self.assertEqual(len(hidden_layer_variables), 2 * len(self.layers))

            output_layer_variables = tf.trainable_variables('trajectory/policy/output/')
            self.assertEqual(len(output_layer_variables), 2 * len(self.action_fluents))

    def test_evaluation_run(self):
        horizon = 45
        batch_size = 1024
        trajectories, _, _ = self.evaluator.run(horizon, batch_size)
        self.assertIsInstance(trajectories, tuple)
        self.assertEqual(len(trajectories), 5)

        initial_state, states, actions, interms, rewards = trajectories

        for fluent in initial_state:
            self.assertEqual(fluent.shape[0], batch_size)

        for fluent in states:
            self.assertTupleEqual(fluent.shape[:2], (batch_size, horizon))

        for fluent in actions:
            self.assertTupleEqual(fluent.shape[:2], (batch_size, horizon))

        for fluent in interms:
            self.assertTupleEqual(fluent.shape[:2], (batch_size, horizon))

        self.assertTupleEqual(rewards.shape[:2], (batch_size, horizon))
