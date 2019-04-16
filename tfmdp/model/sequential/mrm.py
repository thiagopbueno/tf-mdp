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


import rddl2tf.compiler

from tfmdp.policy.drp import DeepReactivePolicy

import abc
import collections
import tensorflow as tf

from typing import Dict, Sequence, Tuple


Trajectory = collections.namedtuple('Trajectory', 'states actions interms rewards')


class MarkovRecurrentModel(metaclass=abc.ABCMeta):
    '''MarkovRecurrentModel abstract base class.

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler.
        config (Dict): The recurrent model configuration parameters.
    '''

    def __init__(self, compiler: rddl2tf.compiler.Compiler, config: Dict) -> None:
        self.compiler = compiler
        self.config = config

    @abc.abstractmethod
    def build(self, policy: DeepReactivePolicy) -> None:
        '''Builds the recurrent cell ops by embedding the `policy` in the transition sampling.

        Args:
            policy (:obj:`tfmdp.policy.drp.DeepReactivePolicy`): A deep reactive policy.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self,
                 initial_state: Sequence[tf.Tensor],
                 horizon: int) -> Tuple[Trajectory, Sequence[tf.Tensor], tf.Tensor]:
        '''Samples a batch state-action-reward trajectory with given
        `initial_state` and `horizon`, and returns the corresponding total reward.

        Args:
            initial_state (Sequence[tf.Tensor]): The initial state tensors.
            horizon (int): The number of timesteps in each sampled trajectory.

        Returns:
            Tuple[Trajectory, Sequence[tf.Tensor], tf.Tensor]: A triple of (namedtuple, tensors, tensor)
            representing the trajectory, final state, and total reward.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def summary(self) -> None:
        '''Prints a string summary of the recurrent model.'''
        raise NotImplementedError

    @property
    def graph(self) -> tf.Graph:
        '''Returns the model's computation graph.'''
        return self.compiler.graph

    def to_json(self) -> str:
        '''Returns the model configuration parameters serialized in JSON format.'''
        return json.dumps(self.config, sort_keys=True, indent=4)

    @classmethod
    def from_json(cls, compiler: rddl2tf.compiler.Compiler,
                       json_config: str) -> 'MarkovRecurrentModel':
        '''Instantiates a model from a `json_config` string.

        Args:
            compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler.
            json_config (str): A model configuration encoded in JSON format.

        Returns:
            :obj:`tfmdp.model.sequential.mrm.MarkovRecurrentModel`: A MarkovRecurrentModel object.
        '''
        config = json.loads(json_string)
        return cls(compiler, config)

    def timesteps(self, horizon: int, batch_size: int) -> tf.Tensor:
        with self.graph.as_default():
            with tf.name_scope('timesteps'):
                start, limit, delta = horizon - 1, -1, -1
                timesteps_range = tf.range(start, limit, delta, dtype=tf.float32)
                timesteps_range = tf.expand_dims(timesteps_range, -1)
                batch_timesteps = tf.stack([timesteps_range] * batch_size)
                return batch_timesteps
