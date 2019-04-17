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

from rddl2tf.reparam import get_cpfs_reparameterization

from tfmdp.model import utils

import numpy as np
import tensorflow as tf
import unittest


class TestNoiseUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # hyper-parameters
        cls.batch_size = 16
        cls.horizon = 20

        cls.compiler = rddlgym.make('Navigation-v2', mode=rddlgym.SCG)
        cls.compiler.batch_mode_on()

        cls.noise_shapes = get_cpfs_reparameterization(cls.compiler.rddl)

        with cls.compiler.graph.as_default():
            cls.noise_variables = utils.get_noise_variables(cls.noise_shapes, cls.batch_size, cls.horizon)
            cls.inputs, cls.encoding = utils.encode_noise_as_inputs(cls.noise_variables)

    def test_get_noise_variables(self):
        self.assertIsInstance(self.noise_variables, list)
        self.assertEqual(len(self.noise_variables), len(self.noise_shapes))

        for tensor, shape in zip(self.noise_variables, self.noise_shapes):

            self.assertEqual(tensor[0], shape[0])
            if shape[1] == []:
                self.assertIsNone(tensor[1])
            else:
                self.assertIsInstance(tensor[1], list)
                for xi, sh in zip(tensor[1], shape[1]):
                    self.assertListEqual(xi.shape.as_list(), [self.batch_size, self.horizon, *sh[1]])

    def test_noise_encoding(self):
        noise_variables = dict(self.noise_variables)

        total_encoding_size = 0
        for name, slices in self.encoding:
            noise = noise_variables[name]
            if noise is not None:
                self.assertEqual(len(slices), len(noise))
                for (start, end, shape), tensor in zip(slices, noise):
                    self.assertListEqual(shape, tensor.shape.as_list()[2:])
                    size = np.prod(tensor.shape.as_list()[2:])
                    self.assertEqual(size, end-start+1)
                    total_encoding_size += size

        self.assertIsInstance(self.inputs, tf.Tensor)
        self.assertListEqual(self.inputs.shape.as_list(), [self.batch_size, self.horizon, total_encoding_size])

    def test_noise_decoding(self):
        noise_variables_lst = [(name, var_list) for name, var_list in self.noise_variables if var_list is not None]

        for t in range(self.horizon):
            inputs_per_timestep = self.inputs[:, t, :]

            noise_variables_per_timestep = utils.decode_inputs_as_noise(inputs_per_timestep, self.encoding)
            self.assertIsInstance(noise_variables_per_timestep, list)
            self.assertEqual(len(noise_variables_per_timestep), len(noise_variables_lst))
            for xi_per_timestep, xi in zip(noise_variables_per_timestep, noise_variables_lst):
                self.assertEqual(xi_per_timestep[0], xi[0])
                self.assertEqual(len(xi_per_timestep[1]), len(xi[1]))
                for xi1, xi2 in zip(xi_per_timestep[1], xi[1]):
                    xi2_shape = xi2.shape.as_list()
                    del xi2_shape[1]
                    self.assertListEqual(xi1.shape.as_list(), xi2_shape)

