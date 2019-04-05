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

from tfmdp import utils
from tfmdp.policy.drp import DeepReactivePolicy, activation_fn

import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Sequence


class FeedforwardPolicy(DeepReactivePolicy):

    def __init__(self, compiler, config):
        super(FeedforwardPolicy, self).__init__(compiler, config)

    @property
    def name(self) -> str:
        '''Returns the canonical DRP name.'''
        params_string = utils.get_params_string(self.config)
        return 'drp-ff-{}'.format(params_string)

    @property
    def size(self) -> int:
        '''Returns the number of trainable parameters.'''
        trainable_variables = self.vars
        shapes = [v.shape.as_list() for v in trainable_variables]
        return sum(np.prod(shape) for shape in shapes)

    @property
    def vars(self) -> Sequence[tf.Variable]:
        '''Returns a list of the trainable variables.'''
        policy_vars = []
        policy_vars += self._input_layer.trainable_variables
        for layer in self._hidden_layers:
            policy_vars += layer.trainable_variables
        for layer in self._output_layers:
            policy_vars += layer.trainable_variables
        return policy_vars

    def build(self) -> None:
        '''Create the DRP layers and trainable weights.'''
        with self.graph.as_default():
            self._build_input_layer()
            self._build_hidden_layers()
            self._build_output_layers()

    def _build_input_layer(self) -> None:
        self._input_layer = tf.layers.Flatten()

    def _build_hidden_layers(self) -> None:
        activation = activation_fn[self.config['activation']]
        self._hidden_layers = tuple(tf.layers.Dense(units, activation=activation)
                                    for units in self.config['layers'])

    def _build_output_layers(self) -> None:
        self._output_layers = tuple(tf.layers.Dense(np.prod(size))
                                    for size in self.compiler.rddl.action_size)

    def __call__(self,
                 state: Sequence[tf.Tensor],
                 timestep: tf.Tensor) -> Sequence[tf.Tensor]:
        '''Returns action fluents for the current `state` and `timestep`.

        Args:
            state (Sequence[tf.Tensor]): A tuple of state fluents.
            timestep (tf.Tensor): The current timestep.

        Returns:
            Sequence[tf.Tensor]: A tuple of action fluents.
        '''
        with self.graph.as_default():
            with tf.variable_scope('policy', reuse=tf.AUTO_REUSE):

                # input layer
                state_layers = list(map(self._input_layer, state))
                input_layer = tf.concat(state_layers, axis=1)

                # hidden layers
                h = input_layer
                for layer in self._hidden_layers:
                    h = layer(h)

                # output layers
                output_layers = tuple(layer(h) for layer in self._output_layers)

                # action
                return self._action_outputs(state, output_layers)

    def _action_outputs(self, state, output_layers):
        batch_size = int(state[0].shape[0])

        bounds = self.compiler.compile_action_bound_constraints(state)

        action_fluents = self.compiler.rddl.domain.action_fluent_ordering
        action_size = self.compiler.rddl.action_size

        action = []
        for fluent_name, fluent_size, layer in zip(action_fluents, action_size, output_layers):
            action_tensor = tf.reshape(layer, [batch_size] + list(fluent_size))
            action_tensor = self._get_output_tensor(action_tensor, bounds[fluent_name])
            action.append(action_tensor)

        return tuple(action)

    def _get_output_tensor(self, tensor, bounds):
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
