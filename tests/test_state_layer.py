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

from tfmdp.policy.layers.state_layer import StateLayer

import numpy as np
import tensorflow as tf
import unittest


class TestStateLayer(unittest.TestCase):

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

    def setUp(self):
        with self.compiler.graph.as_default():
            self.layer = StateLayer()
            self.output = self.layer(self.initial_state)

    def test_trainable_variables(self):
        self.assertListEqual(self.layer.trainable_variables, [])

    def test_call(self):
        state_size = self.compiler.rddl.state_size
        total_state_size = sum(np.prod(size) for size in state_size)
        self.assertListEqual(self.output.shape.as_list(), [self.batch_size, total_state_size])
