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


import tensorflow as tf

from typing import Dict, Optional, Sequence


class StateLayer(tf.layers.Layer):
    '''StateLayer should be used as an input layer in a DRP.

    It flattens each state fluent and returns a single
    concatenated tensor.

    Args:
        input_layer_norm (bool): The boolean flag for enabling layer normalization.
    '''

    def __init__(self, input_layer_norm: bool = False) -> None:
        super(StateLayer, self).__init__(name='state_layer')
        self.flatten = tf.layers.Flatten()
        self.input_layer_norm = input_layer_norm

    @property
    def trainable_variables(self) -> None:
        '''Returns the list of all layer variables/weights.'''
        variables = []
        # TODO
        return variables

    def call(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        '''Returns the concatenation of all state fluent tensors previously flatten.

        Args:
            inputs (Sequence[tf.Tensor]): A tuple of state fluent tensors.

        Returns:
            tf.Tensor: A single output tensor.
        '''
        state_layers = []
        for tensor in inputs:
            layer = self.flatten(tensor)
            if self.input_layer_norm:
                layer = tf.contrib.layers.layer_norm(layer)
            state_layers.append(layer)
        return tf.concat(state_layers, axis=1)
