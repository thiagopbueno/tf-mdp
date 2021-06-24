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
from rddl2tf.compilers import ReparameterizationCompiler

from tfmdp.policy.feedforward import FeedforwardPolicy
from tfmdp.model.cell.reparameterization_cell import ReparameterizationCell
from tfmdp.model.sequential.mrm import Trajectory
from tfmdp.model.sequential.reparameterization import ReparameterizationSampling

import tensorflow as tf
import unittest


class TestMonteCarloSampling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # hyper-parameters
        cls.horizon = 20
        cls.batch_size = 64

        # rddl
        rddl = rddlgym.make('Navigation-v2', rddlgym.AST)
        cls.compiler = ReparameterizationCompiler(rddl, cls.batch_size)
        cls.compiler.init()

        # initial state
        cls.initial_state = cls.compiler.initial_state()

        # default action
        cls.default_action = cls.compiler.default_action()

        # policy
        cls.policy = FeedforwardPolicy(cls.compiler, {'layers': [32, 32], 'activation': 'elu', 'input_layer_norm': False})
        cls.policy.build()

        # model
        cls.config = {}
        cls.model = ReparameterizationSampling(cls.compiler, cls.config)
        cls.model.build(cls.policy)
        cls.output = cls.model(cls.initial_state, cls.horizon)

    def test_build(self):
        self.assertIsInstance(self.model.cell, ReparameterizationCell)
        self.assertEqual(self.model.cell.policy, self.policy)

    def test_call(self):
        noise_size = self.model.noise.shape.as_list()[-1]
        timesteps_size = self.model.timesteps.shape.as_list()[-1]
        inputs_size = noise_size + timesteps_size
        self.assertListEqual(self.model.inputs.shape.as_list(), [self.batch_size, self.horizon, inputs_size])
        self.assertListEqual(self.model.timesteps.shape.as_list()[:2], self.model.noise.shape.as_list()[:2])

        self.assertIsInstance(self.output, tuple)
        self.assertEqual(len(self.output), 3)

        trajectory, final_state, total_reward = self.output

        self.assertIsInstance(trajectory, Trajectory)

        for tensor, fluent in zip(trajectory.states, self.initial_state):
            self.assertEqual(int(tensor.shape[0]), self.batch_size)
            self.assertEqual(int(tensor.shape[1]), self.horizon)
            self.assertListEqual(tensor.shape.as_list()[2:], fluent.shape.as_list()[1:])

        for tensor, fluent in zip(trajectory.actions, self.default_action):
            self.assertEqual(int(tensor.shape[0]), self.batch_size)
            self.assertEqual(int(tensor.shape[1]), self.horizon)
            self.assertListEqual(tensor.shape.as_list()[2:], fluent.shape.as_list()[1:])

        self.assertIsInstance(final_state, tuple)
        self.assertEqual(len(final_state), len(self.initial_state))

        self.assertIsInstance(total_reward, tf.Tensor)
        self.assertEqual(total_reward.dtype, tf.float32)
        self.assertListEqual(total_reward.shape.as_list(), [self.batch_size])

    def test_trainable_variables(self):
        var_lst1 = self.model.trainable_variables
        var_lst2 = []
        for _, var_lst in self.model.noise_map:
            if var_lst is not None:
                var_lst2 += var_lst
        self.assertSetEqual(set(var_lst1), set(var_lst2))
