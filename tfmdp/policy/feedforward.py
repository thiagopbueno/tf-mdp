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


import rddl2tf

from tfmdp import utils
from tfmdp.policy.drp import DeepReactivePolicy, activation_fn
from tfmdp.policy.layers.state_layer import StateLayer
from tfmdp.policy.layers.action_layer import ActionLayer

import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Sequence


class FeedforwardPolicy(DeepReactivePolicy):
    '''FeedforwardPolicy implements a DRP as a multi-layer perceptron.

    It is parameterized by the following configuration params:
        - config['layers']: a list of number of units; and
        - config['activation']: an activation function.

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler.
        config (Dict): The policy configuration parameters.
    '''

    def __init__(self,
                 compiler: rddl2tf.compiler.Compiler,
                 config: dict) -> None:
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
        with self.graph.as_default():
            policy_vars = tf.trainable_variables(r'.*policy')
            return policy_vars

    def build(self) -> None:
        '''Create the DRP layers and trainable weights.'''
        with self.graph.as_default():
            with tf.variable_scope('policy'):
                self._build_input_layer()
                self._build_hidden_layers()
                self._build_output_layer()

    def _build_input_layer(self) -> None:
        '''Builds the DRP input layer using a `tfmdp.policy.layers.state_layer.StateLayer`.'''
        self._input_layer = StateLayer(self.config['input_layer_norm'])

    def _build_hidden_layers(self) -> None:
        '''Builds all hidden layers as `tf.layers.Dense` layers.'''
        activation = activation_fn[self.config['activation']]
        self._hidden_layers = tuple(tf.layers.Dense(units, activation=activation)
                                    for units in self.config['layers'])

    def _build_output_layer(self) -> None:
        '''Builds the DRP output layer using a `tfmdp.policy.layers.action_layer.ActionLayer`.'''
        self._output_layer = ActionLayer(self.compiler.rddl.action_size)

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
                input_layer = self._input_layer(state)

                # hidden layers
                h = input_layer
                for layer in self._hidden_layers:
                    h = layer(h)

                # output layer
                action_fluents = self.compiler.rddl.domain.action_fluent_ordering
                action_bounds = self.compiler.compile_action_bound_constraints(state)
                action_bounds = [action_bounds[fluent_name] for fluent_name in action_fluents]
                return self._output_layer(h, action_bounds=action_bounds)
