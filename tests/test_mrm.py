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
from tfmdp.train.mrm import MRMCell


import tensorflow as tf

import unittest


class TestMRMCell(unittest.TestCase):

    def setUp(self):
        self.rddl1 = rddlgym.make('Navigation-v2', mode=rddlgym.AST)
        self.compiler1 = Compiler(self.rddl1, batch_mode=True)

        self.layers = [64, 32, 16]
        self.policy1 = DeepReactivePolicy(self.compiler1, self.layers, tf.nn.elu, input_layer_norm=True)

        self.batch_size1 = 100
        self.cell1 = MRMCell(self.compiler1, self.policy1, self.batch_size1)

    def test_state_size(self):
        expected1 = ((2,),)
        actual1 = self.cell1.state_size
        self.assertTupleEqual(actual1, expected1)

    def test_output_size(self):
        expected1 = (((2,),), ((2,),), ((2,), (2,)), 1, 1)
        actual1 = self.cell1.output_size
        self.assertTupleEqual(actual1, expected1)
