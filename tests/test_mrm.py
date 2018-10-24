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

    def test_initial_state(self):
        cells = [self.cell1]
        batch_sizes = [self.batch_size1]
        for cell, batch_size in zip(cells, batch_sizes):
            initial_state = cell.initial_state()
            self.assertIsInstance(initial_state, tuple)
            self.assertEqual(len(initial_state), len(cell.state_size))
            for t, shape in zip(initial_state, cell.state_size):
                self.assertIsInstance(t, tf.Tensor)
                expected_shape = [batch_size] + list(shape)
                if len(expected_shape) == 1:
                    expected_shape += [1]
                self.assertListEqual(t.shape.as_list(), expected_shape)

    def test_output_simulation_step(self):
        horizon = 40
        cells = [self.cell1]
        batch_sizes = [self.batch_size1]
        for cell, batch_size in zip(cells, batch_sizes):
            with cell.graph.as_default():
                # initial_state
                initial_state = cell.initial_state()

                # timestep
                timestep = tf.constant(horizon, dtype=tf.float32)
                timestep = tf.expand_dims(timestep, -1)
                timestep = tf.stack([timestep] * batch_size)

                # stop_flag
                stop_flag = tf.constant(0.0, shape=(batch_size,1), dtype=tf.float32)

                input = tf.concat([timestep, stop_flag], axis=1)

                # simulation step
                output, _ = cell(input, initial_state)
                self.assertIsInstance(output, tuple)
                self.assertEqual(len(output), 5)

    def test_next_state_output(self):
        horizon = 40
        cells = [self.cell1]
        batch_sizes = [self.batch_size1]
        for cell, batch_size in zip(cells, batch_sizes):
            with cell.graph.as_default():
                # initial_state
                initial_state = cell.initial_state()

                # timestep
                timestep = tf.constant(horizon, dtype=tf.float32)
                timestep = tf.expand_dims(timestep, -1)
                timestep = tf.stack([timestep] * batch_size)

                # stop_flag
                stop_flag = tf.constant(0.0, shape=(batch_size,1), dtype=tf.float32)

                input = tf.concat([timestep, stop_flag], axis=1)

                # simulation step
                output, _ = cell(input, initial_state)
                next_state, _, _, _, _ = output
                next_state_size, _, _, _, _ = cell.output_size

                self.assertIsInstance(next_state, tuple)
                self.assertEqual(len(next_state), len(next_state_size))
                for s, sz in zip(next_state, next_state_size):
                    self.assertIsInstance(s, tf.Tensor)
                    self.assertEqual(s.dtype, tf.float32)
                    self.assertListEqual(s.shape.as_list(), [batch_size] + list(sz))

    def test_action_output(self):
        horizon = 40
        cells = [self.cell1]
        batch_sizes = [self.batch_size1]
        for cell, batch_size in zip(cells, batch_sizes):
            with cell.graph.as_default():
                # initial_state
                initial_state = cell.initial_state()

                # timestep
                timestep = tf.constant(horizon, dtype=tf.float32)
                timestep = tf.expand_dims(timestep, -1)
                timestep = tf.stack([timestep] * batch_size)

                # stop_flag
                stop_flag = tf.constant(0.0, shape=(batch_size,1), dtype=tf.float32)

                input = tf.concat([timestep, stop_flag], axis=1)

                # simulation step
                output, _ = cell(input, initial_state)
                _, action, _, _, _ = output
                _, action_size, _, _, _ = cell.output_size

                self.assertIsInstance(action, tuple)
                self.assertEqual(len(action), len(action_size))
                for a, sz in zip(action, action_size):
                    self.assertIsInstance(a, tf.Tensor)
                    self.assertEqual(a.dtype, tf.float32)
                    self.assertListEqual(a.shape.as_list(), [batch_size] + list(sz))

    def test_reward_output(self):
        horizon = 40
        cells = [self.cell1]
        batch_sizes = [self.batch_size1]
        for cell, batch_size in zip(cells, batch_sizes):
            with cell.graph.as_default():
                # initial_state
                initial_state = cell.initial_state()

                # timestep
                timestep = tf.constant(horizon, dtype=tf.float32)
                timestep = tf.expand_dims(timestep, -1)
                timestep = tf.stack([timestep] * batch_size)

                # stop_flag
                stop_flag = tf.constant(0.0, shape=(batch_size,1), dtype=tf.float32)

                input = tf.concat([timestep, stop_flag], axis=1)

                # simulation step
                output, _ = cell(input, initial_state)
                _, _, _, reward, _ = output
                _, _, _, reward_size, _ = cell.output_size

                self.assertIsInstance(reward, tf.Tensor)
                self.assertListEqual(reward.shape.as_list(), [batch_size, reward_size])

    def test_log_prob_output(self):
        horizon = 40
        cells = [self.cell1]
        batch_sizes = [self.batch_size1]
        for cell, batch_size in zip(cells, batch_sizes):
            with cell.graph.as_default():
                # initial_state
                initial_state = cell.initial_state()

                # timestep
                timestep = tf.constant(horizon, dtype=tf.float32)
                timestep = tf.expand_dims(timestep, -1)
                timestep = tf.stack([timestep] * batch_size)

                # stop_flag
                stop_flag = tf.constant(0.0, shape=(batch_size,1), dtype=tf.float32)

                input = tf.concat([timestep, stop_flag], axis=1)

                # simulation step
                output, _ = cell(input, initial_state)
                _, _, _, _, log_prob = output
                _, _, _, _, log_prob_size = cell.output_size

                self.assertIsInstance(log_prob, tf.Tensor)
                self.assertListEqual(log_prob.shape.as_list(), [batch_size, log_prob_size])
