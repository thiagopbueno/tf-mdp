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
from tfmdp.model.sequential.reparameterization import ReparameterizationSampling
from tfmdp.train.optimizers import optimizers
from tfmdp.planning.minimax import MinimaxOptimizationPlanner

import tensorflow as tf
import unittest


class TestMinimaxOptimizationPlanner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # hyper-parameters
        cls.batch_size = 64
        cls.horizon = 20
        cls.learning_rate = 0.001

        # rddl
        cls.compiler = rddlgym.make('Navigation-v2', rddlgym.SCG)
        cls.compiler.batch_mode_on()

        # policy
        cls.policy = FeedforwardPolicy(cls.compiler, {'layers': [256], 'activation': 'elu', 'input_layer_norm': False})
        cls.policy.build()

        # planner
        cls.config = {
            'batch_size': cls.batch_size,
            'horizon': cls.horizon,
            'learning_rate': cls.learning_rate
        }
        cls.planner = MinimaxOptimizationPlanner(cls.compiler, cls.config)
        cls.planner.build(cls.policy, loss='mse', optimizer='RMSProp')

    def test_build(self):
        self.assertIsInstance(self.planner.model, ReparameterizationSampling)
        self.assertIsInstance(self.planner.optimizer, optimizers['RMSProp'])

        noise_variables = self.planner.noise_variables
        policy_variables = self.planner.policy_variables
        with self.compiler.graph.as_default():
            trainable_variables = tf.trainable_variables()

        self.assertEqual(len(trainable_variables), len(noise_variables) + len(policy_variables))
        self.assertSetEqual(set(trainable_variables), set(noise_variables) | set(policy_variables))

    @unittest.skip
    def test_run(self):
        self.fail()
