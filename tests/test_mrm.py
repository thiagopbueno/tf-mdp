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
from tfmdp.train.mrm import MarkovCell, MarkovRecurrentModel, ReparameterizationType

import numpy as np
import tensorflow as tf

import unittest


class TestMarkovCell(unittest.TestCase):

    def setUp(self):
        self.rddl1 = rddlgym.make('Navigation-v3', mode=rddlgym.AST)
        self.compiler1 = Compiler(self.rddl1, batch_mode=True)

        self.layers = [64, 32, 16]
        self.policy1 = DeepReactivePolicy(self.compiler1, self.layers, tf.nn.elu, input_layer_norm=False)

        self.batch_size1 = 8
        self.cell1 = MarkovCell(self.compiler1, self.policy1, self.batch_size1)

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

                # reparam_flag
                reparam_flag = tf.constant(0.0, shape=(batch_size,1), dtype=tf.float32)
                input = tf.concat([timestep, reparam_flag], axis=1)

                # simulation step
                output, _ = cell(input, initial_state)
                self.assertIsInstance(output, tuple)
                self.assertEqual(len(output), 5)

    def test_next_state_in_not_reparameterized_cell(self):
        grad_a, grad_s = self._test_grad_next_state(MarkovRecurrentModel.NOT_REPARAMETERIZED_FLAG)
        self.assertTrue(np.sum(grad_a) == 0.0)
        self.assertTrue(np.sum(grad_s) == 0.0)

    def test_next_state_in_fully_reparameterized_cell(self):
        grad_a, grad_s = self._test_grad_next_state(MarkovRecurrentModel.FULLY_REPARAMETERIZED_FLAG)
        self.assertTrue(np.sum(grad_a) != 0.0)
        self.assertTrue(np.sum(grad_s) != 0.0)

    def _test_grad_next_state(self, reparameterization_type):
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

                # reparam_flag
                reparam_flag = tf.constant(reparameterization_type, shape=(batch_size,1), dtype=tf.float32)
                input = tf.concat([timestep, reparam_flag], axis=1)

                # simulation step
                output, _ = cell(input, initial_state)
                next_state, action, _, _, _ = output

                loss = tf.reduce_sum(next_state[0])
                grad_a, grad_s = tf.gradients(ys=loss, xs=[action[0], initial_state[0]])

                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    return sess.run([grad_a, grad_s])

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

                # reparam_flag
                reparam_flag = tf.constant(0.0, shape=(batch_size,1), dtype=tf.float32)

                input = tf.concat([timestep, reparam_flag], axis=1)

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

                # reparam_flag
                reparam_flag = tf.constant(0.0, shape=(batch_size,1), dtype=tf.float32)

                input = tf.concat([timestep, reparam_flag], axis=1)

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

                # reparam_flag
                reparam_flag = tf.constant(0.0, shape=(batch_size,1), dtype=tf.float32)

                input = tf.concat([timestep, reparam_flag], axis=1)

                # simulation step
                output, _ = cell(input, initial_state)
                _, _, _, reward, _ = output
                _, _, _, reward_size, _ = cell.output_size

                self.assertIsInstance(reward, tf.Tensor)
                self.assertListEqual(reward.shape.as_list(), [batch_size, 1])

    def test_log_prob_not_reparameterized_cell(self):
        _test_log_prob_output(MarkovRecurrentModel.NOT_REPARAMETERIZED_FLAG)

    def test_log_prob_fully_reparameterized_cell(self):
        _test_log_prob_output(MarkovRecurrentModel.FULLY_REPARAMETERIZED_FLAG)

    def _test_log_prob_output(self, reparameterization_type):
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

                # reparam_flag
                reparam_flag = tf.constant(reparameterization_type, shape=(batch_size,1), dtype=tf.float32)

                input = tf.concat([timestep, reparam_flag], axis=1)

                # simulation step
                output, _ = cell(input, initial_state)
                _, _, _, _, log_prob = output
                _, _, _, _, log_prob_size = cell.output_size

                self.assertIsInstance(log_prob, tf.Tensor)
                self.assertListEqual(log_prob.shape.as_list(), [batch_size, log_prob_size])

    def test_log_prob_fully_reparameterized_cell(self):
        self._test_grad_log_prob(MarkovRecurrentModel.FULLY_REPARAMETERIZED_FLAG)

    def test_log_prob_not_reparameterized_cell(self):
        self._test_grad_log_prob(MarkovRecurrentModel.NOT_REPARAMETERIZED_FLAG)

    def _test_grad_log_prob(self, reparameterization_type):
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

                # reparam_flag
                reparam_flag = tf.constant(reparameterization_type, shape=(batch_size,1), dtype=tf.float32)
                input = tf.concat([timestep, reparam_flag], axis=1)

                # simulation step
                output, _ = cell(input, initial_state)
                next_state, action, _, _, log_prob = output

                loss = tf.reduce_sum(log_prob)
                grad_a, grad_s, grad_ss = tf.gradients(ys=loss, xs=[action[0], initial_state[0], next_state[0]])
                self.assertIsNone(grad_ss)

                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    grad_a, grad_s= sess.run([grad_a, grad_s])
                    self.assertTrue(np.sum(grad_a) != 0.0)
                    self.assertTrue(np.sum(grad_s) != 0.0)


class TestMarkovRecurrentModel(unittest.TestCase):

    def setUp(self):
        self.rddl1 = rddlgym.make('Navigation-v3', mode=rddlgym.AST)
        self.compiler1 = Compiler(self.rddl1, batch_mode=True)

        self.layers = [64, 32, 16]
        self.policy1 = DeepReactivePolicy(self.compiler1, self.layers, tf.nn.elu, input_layer_norm=False)

        self.batch_size1 = 100
        self.mrm1 = MarkovRecurrentModel(self.compiler1, self.policy1, self.batch_size1)

    def test_timesteps(self):
        horizon = 40
        simulators = [self.mrm1]
        batch_sizes = [self.batch_size1]

        for mrm, batch_size in zip(simulators, batch_sizes):

            with mrm.graph.as_default():
                timesteps = mrm._timesteps(horizon)

            self.assertIsInstance(timesteps, tf.Tensor)
            self.assertListEqual(timesteps.shape.as_list(), [batch_size, horizon, 1])

            with tf.Session(graph=mrm.graph) as sess:
                timesteps = sess.run(timesteps)

                for t in timesteps:
                    self.assertListEqual(list(t), list(np.arange(horizon-1, -1, -1)))

    def test_reparam_flags(self):
        horizon = 20
        simulators = [self.mrm1]
        batch_sizes = [self.batch_size1]

        for mrm, batch_size in zip(simulators, batch_sizes):

            with mrm.graph.as_default():

                reparam_flags1 = mrm._reparam_flags(horizon, ReparameterizationType.FULLY_REPARAMETERIZED)
                self.assertIsInstance(reparam_flags1, tf.Tensor)
                self.assertListEqual(reparam_flags1.shape.as_list(), [batch_size, horizon, 1])

                reparam_flags2 = mrm._reparam_flags(horizon, ReparameterizationType.NOT_REPARAMETERIZED)
                self.assertIsInstance(reparam_flags2, tf.Tensor)
                self.assertListEqual(reparam_flags2.shape.as_list(), [batch_size, horizon, 1])

                n_step = 5
                reparam_flags3 = mrm._reparam_flags(horizon, (ReparameterizationType.PARTIALLY_REPARAMETERIZED, n_step))
                self.assertIsInstance(reparam_flags3, tf.Tensor)
                self.assertListEqual(reparam_flags3.shape.as_list(), [batch_size, horizon, 1])

                not_reparam_constant = MarkovRecurrentModel.NOT_REPARAMETERIZED_FLAG
                fully_reparam_constant = MarkovRecurrentModel.FULLY_REPARAMETERIZED_FLAG
                with tf.Session(graph=mrm.graph) as sess:
                    f1, f2, f3 = sess.run([reparam_flags1, reparam_flags2, reparam_flags3])
                    self.assertTrue(np.all(f1 == fully_reparam_constant))
                    self.assertTrue(np.all(f2 == not_reparam_constant))

                    self.assertTrue(all(flag == not_reparam_constant for t, flag in enumerate(f3[0, :, 0]) if (t+1) % n_step == 0))
                    self.assertTrue(all(flag == fully_reparam_constant for t, flag in enumerate(f3[0, :, 0]) if (t+1) % n_step != 0))
                    for batch_flags in f3:
                        self.assertTrue(np.all(batch_flags == f3[0]))

    def test_inputs(self):
        horizon = 40
        simulators = [self.mrm1]
        batch_sizes = [self.batch_size1]

        for mrm, batch_size in zip(simulators, batch_sizes):

            with mrm.graph.as_default():

                timesteps = mrm._timesteps(horizon)

                flags1 = mrm._reparam_flags(horizon, ReparameterizationType.FULLY_REPARAMETERIZED)
                inputs1 = mrm._inputs(timesteps, flags1)
                self.assertIsInstance(inputs1, tf.Tensor)
                self.assertListEqual(inputs1.shape.as_list(), [batch_size, horizon, 2])

                flags2 = mrm._reparam_flags(horizon, ReparameterizationType.NOT_REPARAMETERIZED)
                inputs2 = mrm._inputs(timesteps, flags2)
                self.assertIsInstance(inputs2, tf.Tensor)
                self.assertListEqual(inputs2.shape.as_list(), [batch_size, horizon, 2])

                with tf.Session(graph=mrm.graph) as sess:
                    inputs1, inputs2 = sess.run([inputs1, inputs2])

                    for t in inputs1:
                        self.assertListEqual(list(t[:,0]), list(np.arange(horizon-1, -1, -1)))

                    for t in inputs2:
                        self.assertListEqual(list(t[:,0]), list(np.arange(horizon-1, -1, -1)))

                    self.assertTrue(np.all(inputs1[:,:,1] == np.zeros((batch_size, horizon))))
                    self.assertTrue(np.all(inputs2[:,:,1] == np.ones((batch_size, horizon))))

    def test_trajectory_fully_reparameterized(self):
        self._test_trajectory(ReparameterizationType.FULLY_REPARAMETERIZED)

    def test_trajectory_not_reparameterized(self):
        self._test_trajectory(ReparameterizationType.NOT_REPARAMETERIZED)

    def _test_trajectory(self, reparam_type):
        horizon = 40
        compilers = [self.compiler1]
        simulators = [self.mrm1]
        batch_sizes = [self.batch_size1]

        for compiler, mrm, batch_size in zip(compilers, simulators, batch_sizes):

            with mrm.graph.as_default():
                initial_state = mrm._cell.initial_state()

                timesteps = mrm._timesteps(horizon)
                flags = mrm._reparam_flags(horizon, reparam_type)
                inputs = mrm._inputs(timesteps, flags)

                trajectory = mrm._trajectory(initial_state, inputs)

            self.assertIsInstance(trajectory, tuple)
            self.assertEqual(len(trajectory), 5)

            # sizes
            state_size, action_size, interm_size, reward_size, log_prob_size = mrm.output_size

            # states
            self.assertIsInstance(trajectory.states, tuple)
            self.assertEqual(len(trajectory[1]), len(state_size))
            for s, sz in zip(trajectory[1], state_size):
                self.assertIsInstance(s, tf.Tensor)
                self.assertListEqual(s.shape.as_list(), [batch_size, horizon] + list(sz), '{}'.format(s))

            # interms
            self.assertIsInstance(trajectory.interms, tuple)
            self.assertEqual(len(trajectory.interms), len(interm_size))
            for s, sz in zip(trajectory.interms, interm_size):
                self.assertIsInstance(s, tf.Tensor)
                self.assertListEqual(s.shape.as_list(), [batch_size, horizon] + list(sz), '{}'.format(s))

            # actions
            self.assertIsInstance(trajectory.actions, tuple)
            self.assertEqual(len(trajectory.actions), len(action_size))
            for a, sz in zip(trajectory.actions, action_size):
                self.assertIsInstance(a, tf.Tensor)
                self.assertListEqual(a.shape.as_list(), [batch_size, horizon] + list(sz))

            # rewards
            self.assertIsInstance(trajectory.rewards, tf.Tensor)
            self.assertListEqual(trajectory.rewards.shape.as_list(), [batch_size, horizon, reward_size])

            # log_probs
            self.assertIsInstance(trajectory.log_probs, tf.Tensor)
            self.assertListEqual(trajectory.log_probs.shape.as_list(), [batch_size, horizon, log_prob_size])

    def test_reward_to_go(self):
        horizon = 40
        compilers = [self.compiler1]
        simulators = [self.mrm1]
        batch_sizes = [self.batch_size1]

        for compiler, mrm, batch_size in zip(compilers, simulators, batch_sizes):

            with mrm.graph.as_default():
                initial_state = mrm._cell.initial_state()

                timesteps = mrm._timesteps(horizon)
                flags = mrm._reparam_flags(horizon, ReparameterizationType.NOT_REPARAMETERIZED)
                inputs = mrm._inputs(timesteps, flags)

                trajectory = mrm._trajectory(initial_state, inputs)
                q = mrm._reward_to_go(trajectory.rewards)

            self.assertIsInstance(q, tf.Tensor)
            self.assertEqual(q.dtype, tf.float32)
            self.assertEqual(q.shape, trajectory.rewards.shape)

    def test_total_reward(self):
        horizon = 40
        compilers = [self.compiler1]
        simulators = [self.mrm1]
        batch_sizes = [self.batch_size1]

        for compiler, mrm, batch_size in zip(compilers, simulators, batch_sizes):
            mrm.build(horizon, lambda x: -x, ReparameterizationType.FULLY_REPARAMETERIZED)
            total_reward = mrm.total_reward
            self.assertIsInstance(total_reward, tf.Tensor)
            self.assertEqual(total_reward.dtype, tf.float32)
            self.assertListEqual(total_reward.shape.as_list(), [batch_size])

    def test_surrogate_loss_fully_reparameterized(self):
        surrogate_batch_cost, costs, q, log_probs = self._test_surrogate_loss(ReparameterizationType.FULLY_REPARAMETERIZED)
        self.assertTrue(np.all(surrogate_batch_cost == costs))

    def test_surrogate_loss_not_reparameterized(self):
        surrogate_batch_cost, costs, q, log_probs = self._test_surrogate_loss(ReparameterizationType.NOT_REPARAMETERIZED)
        self.assertTrue(np.all(surrogate_batch_cost == costs + log_probs * q))

    def _test_surrogate_loss(self, reparameterization_type):
        horizon = 40
        compilers = [self.compiler1]
        simulators = [self.mrm1]
        batch_sizes = [self.batch_size1]

        for compiler, mrm, batch_size in zip(compilers, simulators, batch_sizes):
            mrm.build(horizon, lambda x: -x, reparameterization_type)
            costs = mrm.costs
            q = mrm.q
            log_probs = mrm.trajectory.log_probs
            surrogate_batch_cost = mrm.surrogate_batch_cost

            with tf.Session(graph=mrm.graph) as sess:
                sess.run(tf.global_variables_initializer())
                return sess.run([surrogate_batch_cost, costs, q, log_probs])
