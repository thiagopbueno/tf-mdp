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
from tfmdp.model.sequential.montecarlo import MonteCarloSampling
from tfmdp.train.optimizers import optimizers
from tfmdp.planning.pdplanner import PathwiseOptimizationPlanner

import unittest


class TestPathwiseOptimizationPlanner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # hyper-parameters
        cls.batch_size = 64
        cls.horizon = 20
        cls.learning_rate = 0.001

        # rddl
        cls.compiler = rddlgym.make('Reservoir-8', rddlgym.SCG)
        cls.compiler.batch_mode_on()

        # policy
        cls.policy = FeedforwardPolicy(cls.compiler, {'layers': [256], 'activation': 'elu', 'input_layer_norm': True})
        cls.policy.build()

        # planner
        cls.config = {
            'batch_size': cls.batch_size,
            'horizon': cls.horizon,
            'learning_rate': cls.learning_rate
        }
        cls.planner = PathwiseOptimizationPlanner(cls.compiler, cls.config)
        cls.planner.build(cls.policy, loss='mse', optimizer='RMSProp')

    def test_build(self):
        self.assertIsInstance(self.planner.model, MonteCarloSampling)
        self.assertIsInstance(self.planner.optimizer, optimizers['RMSProp'])

    def test_run(self):
        epochs = 10
        self.planner.run(epochs, show_progress=False)
