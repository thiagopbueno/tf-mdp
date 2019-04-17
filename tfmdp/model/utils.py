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


from rddl2tf.fluent import TensorFluent

import numpy as np
import tensorflow as tf
from typing import List, Optional, Sequence, Tuple, Union

Shape = Sequence[int]
NoiseShape = List[Tuple[str, Shape]]
Noise = List[Tuple[str, List[tf.Tensor]]]

NoiseEncoding = Sequence[Tuple[str, Sequence[Tuple[int, int, Shape]]]]


def cell_size(sizes: Sequence[Shape]) -> Sequence[Union[Shape, int]]:
    return tuple(sz if sz != () else (1,) for sz in sizes)


def to_tensor(fluents: Sequence[TensorFluent]) -> Sequence[tf.Tensor]:
    return tuple(f.tensor for f in fluents)


def get_noise_variables(noise_shapes: NoiseShape,
                        batch_size: int,
                        horizon: Optional[int] = None) -> Noise:
    noise_variables = []

    for name, shapes in noise_shapes:
        if not shapes:
            noise_variables.append((name, None))
        else:
            noises = []

            name_scope = name.replace("'", '').replace('/', '_')
            with tf.name_scope(name_scope):
                for i, (dist, shape) in enumerate(shapes):
                    shape = [batch_size, horizon, *shape]
                    xi = tf.get_variable('noise_{}_{}'.format(i, dist), shape=shape, dtype=tf.float32)
                    noises.append(xi)

            noise_variables.append((name, noises))

    return noise_variables


def encode_noise_as_inputs(noise_variables: Noise) -> Tuple[tf.Tensor, NoiseEncoding]:
    xi_variables = []
    encoding = []

    i = 0
    for name, xi_list in noise_variables:
        if xi_list is None:
            continue

        slices = []

        for xi in xi_list:
            batch_size = xi.shape[0]
            horizon = xi.shape[1]
            xi_shape = xi.shape.as_list()[2:]
            xi_size =  np.prod(xi_shape)

            xi_variables.append(tf.reshape(xi, [batch_size, horizon, xi_size]))

            slices.append((i, i+xi_size-1, xi_shape))
            i += xi_size

        encoding.append((name, slices))

    inputs = tf.concat(xi_variables, axis=2)

    return (inputs, encoding)


def decode_inputs_as_noise(inputs: tf.Tensor, encoding: NoiseEncoding) -> Noise:
    noise_variables = []

    for name, slices in encoding:
        xi_lst = []

        for start, end, shape in slices:
            xi = inputs[:, start:end+1]
            xi = tf.reshape(xi, [-1, *shape])
            xi_lst.append(xi)

        noise_variables.append((name, xi_lst))

    return noise_variables
