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


import numpy as np
import tensorflow as tf

from typing import Optional, Sequence, Tuple

LowerBound = Optional[tf.Tensor]
UpperBound = Optional[tf.Tensor]
ActionBounds = Sequence[Tuple[LowerBound, UpperBound]]


class ActionLayer(tf.layers.Layer):
    '''ActionLayer should be used as the output layer in a DRP.

    It generates multi-head dense output layers with the same shape as
    action fluents. Otionally, it restricts the output tensors
    based on action bounds.

    Args:
        action_size (Sequence[Sequence[int]]): The list of action fluent sizes.
    '''

    def __init__(self, action_size: int) -> None:
        super(ActionLayer, self).__init__(name='action_layer')
        self.action_size = action_size
        self.logits = tuple(tf.layers.Dense(np.prod(sz)) for sz in action_size)

    @property
    def trainable_variables(self) -> None:
        '''Returns the list of all layer variables/weights.'''
        variables = []
        for logit_layer in self.logits:
            variables += logit_layer.trainable_variables
        return variables

    def call(self,
             inputs: tf.Tensor,
             action_bounds: Optional[ActionBounds] = None) -> Sequence[tf.Tensor]:
        '''Returns the tensors of the multi-head layer's output.

        Args:
            inputs (tf.Tensor): A hidden layer's output.
            action_bounds (Optional[Sequence[Tuple[Optional[tf.Tensor], Optional[tf.Tensor]]]]): The action bounds.

        Returns:
            Sequence[tf.Tensor]: A tuple of action tensors.
        '''
        action_layers = []

        for layer, size in zip(self.logits, self.action_size):
            fluent = layer(inputs)
            fluent = tf.reshape(fluent, [-1] + list(size))
            action_layers.append(fluent)

        if action_bounds is not None:
            action_layers = [self._get_output_tensor(tensor, bounds)
                                for tensor, bounds in zip(action_layers, action_bounds)]

        return tuple(action_layers)

    def _get_output_tensor(self,
                           tensor: tf.Tensor,
                           bounds: Tuple[LowerBound, UpperBound]) -> tf.Tensor:
        '''Returns the value constrained output tensor.

        Args:
            tensor (tf.Tensor): The layer's output tensor corresponding to an action fluent.
            bounds (Tuple[Optional[tf.Tensor], Optional[tf.Tensor]]): The action fluent bounds.

        Returns:
            (tf.Tensor): the constrained output tensor.
        '''
        lower, upper = bounds
        if lower is not None:
            lower = lower.cast(tf.float32)
            lower = tf.stop_gradient(lower.tensor)
        if upper is not None:
            upper = upper.cast(tf.float32)
            upper = tf.stop_gradient(upper.tensor)

        if lower is not None and upper is not None:
            tensor = lower + (upper - lower) * tf.sigmoid(tensor)
        elif lower is not None and upper is None:
            tensor = lower + tf.exp(tensor)
        elif lower is None and upper is not None:
            tensor = upper - tf.exp(tensor)

        return tensor
