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
from tfmdp.model.sequential.mrm import MarkovRecurrentModel, Trajectory
from tfmdp.model.cell.basic_cell import BasicMarkovCell


import tensorflow as tf
from typing import Dict, Sequence, Optional, Tuple


class MonteCarloSampling(MarkovRecurrentModel):
    '''MonteCarloSampling class builds the symbolic computation graph
    for approximating the finite-horizon expected return of a policy
    for a given initial state by means of Monte-Carlo approximation.

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler.
        config (Dict): The recurrent model configuration parameters.
    '''

    def __init__(self,
                 compiler: rddl2tf.compiler.Compiler,
                 config: Optional[Dict] = None) -> None:
        super(MonteCarloSampling, self).__init__(compiler, config)

    def build(self, policy: DeepReactivePolicy) -> None:
        '''Builds a basic Markov cell ops by embedding the `policy` in the transition sampling.

        Args:
            policy (:obj:`tfmdp.policy.drp.DeepReactivePolicy`): A deep reactive policy.
        '''
        self.cell = BasicMarkovCell(self.compiler, policy)

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
        with self.graph.as_default():

            with tf.name_scope('trajectory'):
                batch_size = int(initial_state[0].shape[0])
                inputs = self.timesteps(horizon, batch_size)
                outputs, final_state = tf.nn.dynamic_rnn(self.cell,
                                                         inputs,
                                                         initial_state=initial_state,
                                                         dtype=tf.float32)

                states = tuple(fluent[0] for fluent in outputs[0])
                actions = tuple(fluent[0] for fluent in outputs[1])
                interms = tuple(fluent[0] for fluent in outputs[2])
                rewards = outputs[3]
                trajectory = Trajectory(states, actions, interms, rewards)

            with tf.name_scope('total_reward'):
                total_reward = tf.reduce_sum(tf.squeeze(trajectory.rewards), axis=1)

        return (trajectory, final_state, total_reward)

    def summary(self) -> None:
        '''Prints a string summary of the recurrent model.'''
        raise NotImplementedError
