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

from tfmdp.policy.drp import DeepReactivePolicy

from tfmdp.model.cell.basic_cell import BasicMarkovCell
from tfmdp.model import utils

import collections
import tensorflow as tf

from typing import Dict, Optional, Sequence, Tuple, Union

Shape = Sequence[int]
FluentPair = Tuple[str, rddl2tf.fluent.TensorFluent]

NonFluentsTensor = Sequence[tf.Tensor]
StateTensor = Sequence[tf.Tensor]
StatesTensor = Sequence[tf.Tensor]
ActionsTensor = Sequence[tf.Tensor]
IntermsTensor = Sequence[tf.Tensor]

CellOutput = Tuple[StatesTensor, ActionsTensor, IntermsTensor, tf.Tensor]
CellState = Sequence[tf.Tensor]

OutputTuple = collections.namedtuple('OutputTuple', 'state action interms reward')


class ReparameterizationCell(BasicMarkovCell):
    '''ReparameterizationCell extends the tfmdp.model.cell.basic_cell class
    to implement a version of a MarkovCell where all stochastic nodes
    are reparameterized. Noise variables are given as inputs.

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler.
        config (Dict): The cell configuration parameters.
    '''

    def __init__(self,
                 compiler: rddl2tf.compiler.Compiler,
                 policy: DeepReactivePolicy,
                 config: Optional[Dict] = None):
        self.compiler = compiler
        self.policy = policy
        self.config = config

    def __call__(self,
                 inputs: tf.Tensor,
                 state: Sequence[tf.Tensor],
                 scope: Optional[str] = None) -> Tuple[CellOutput, CellState]:
        '''Returns the cell's output tuple and next state tensors.

        Output tuple packs together the next state, action, interms,
        and reward tensors in order.

        Args:
            inputs (tf.Tensor): The encoded (timestep, noise) input tensor.
            state (Sequence[tf.Tensor]): The current state tensors.
            scope (Optional[str]): The cell name scope.

        Returns:
            (CellOutput, CellState): A pair with the cell's output tuple and next state.
        '''

        # inputs
        timestep = tf.expand_dims(inputs[:, 0], -1)

        noise = inputs[:, 1:]

        # noise
        noise = utils.decode_inputs_as_noise(noise, self.config['encoding'])
        noise = dict(noise)

        # action
        action = self.policy(state, timestep)

        # next state
        interms, next_state = self.compiler.cpfs(state, action, noise=noise)

        # reward
        reward = self.compiler.reward(state, action, next_state)

        # outputs
        next_state = utils.to_tensor(next_state)
        interms = utils.to_tensor(interms)
        output = OutputTuple(next_state, action, interms, reward)

        return (output, next_state)
