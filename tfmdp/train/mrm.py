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


from rddl2tf.compiler import Compiler
from rddl2tf.fluent import TensorFluent
from tfmdp.train.policy import DeepReactivePolicy

from collections import namedtuple
from enum import Enum
import tensorflow as tf

from typing import Iterable, Sequence, Optional, Tuple, Union

Shape = Sequence[int]
FluentTriple = Tuple[str, TensorFluent, TensorFluent]

NonFluentsTensor = Sequence[tf.Tensor]
StateTensor = Sequence[tf.Tensor]
StatesTensor = Sequence[tf.Tensor]
ActionsTensor = Sequence[tf.Tensor]
IntermsTensor = Sequence[tf.Tensor]

CellOutput = Tuple[StatesTensor, ActionsTensor, IntermsTensor, tf.Tensor]
CellState = Sequence[tf.Tensor]


class MarkovCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, compiler: Compiler, policy: DeepReactivePolicy, batch_size: int) -> None:
        self._compiler = compiler
        self._policy = policy
        self._batch_size = batch_size

    @property
    def graph(self) -> tf.Graph:
        '''Returns the computation graph.'''
        return self._compiler.graph

    @property
    def state_size(self) -> Sequence[Shape]:
        '''Returns the MDP state size.'''
        return self._compiler.state_size

    @property
    def action_size(self) -> Sequence[Shape]:
        '''Returns the MDP action size.'''
        return self._compiler.action_size

    @property
    def interm_size(self) -> Sequence[Shape]:
        '''Returns the MDP intermediate state size.'''
        return self._compiler.interm_size

    @property
    def output_size(self) -> Tuple[Sequence[Shape], Sequence[Shape], Sequence[Shape], int, int]:
        '''Returns the simulation cell output size.'''
        return (self.state_size, self.action_size, self.interm_size, 1, 1)

    def initial_state(self) -> StateTensor:
        '''Returns the initial state tensor.'''
        return self._compiler.compile_initial_state(self._batch_size)

    def __call__(self,
            input: tf.Tensor,
            state: Sequence[tf.Tensor],
            scope: Optional[str] = None) -> Tuple[CellOutput, CellState]:

        timestep, stop_flag = tf.split(input, [1, 1], axis=1)

        # action
        action = self._policy(state, input)

        # next state
        transition_scope = self._compiler.transition_scope(state, action)
        interm_fluents, next_state_fluents = self._compiler.compile_probabilistic_cpfs(transition_scope, self._batch_size)

        # log_probs
        log_prob = self._log_prob(interm_fluents, next_state_fluents)

        # reward
        transition_scope.update({name: fluent for name, fluent, _ in next_state_fluents})
        reward = self._compiler.compile_reward(transition_scope)
        reward = reward.tensor

        # outputs
        interm_state = self._output(interm_fluents)
        next_state = self._output(next_state_fluents)
        output = (next_state, action, interm_state, reward, log_prob)

        return (output, next_state)

    def _log_prob(self, interm_fluents, next_state_fluents):

        with self.graph.as_default():

            interm_log_probs = [log_prob.tensor for _, _, log_prob in interm_fluents]
            interm_log_prob = tf.reduce_sum(
                tf.concat(interm_log_probs, axis=1, name='interm_log_probs'),
                axis=1,
                name='interm_log_prob')

            next_state_log_probs = [log_prob.tensor for _, _, log_prob in next_state_fluents]
            next_state_log_prob = tf.reduce_sum(
                tf.concat(next_state_log_probs, axis=1, name='next_state_log_probs'),
                axis=1,
                name='next_state_log_prob')

            log_prob = tf.expand_dims(interm_log_prob + next_state_log_prob, -1, name='log_prob')
            return log_prob

    @classmethod
    def _tensors(cls, fluents: Sequence[FluentTriple]) -> Iterable[tf.Tensor]:
        '''Yields the `fluents`' tensors.'''
        for _, fluent, _ in fluents:
            yield fluent.tensor

    @classmethod
    def _dtype(cls, tensor: tf.Tensor) -> tf.Tensor:
        '''Converts `tensor` to tf.float32 datatype if needed.'''
        if tensor.dtype != tf.float32:
            tensor = tf.cast(tensor, tf.float32)
        return tensor

    @classmethod
    def _output(cls, fluents: Sequence[FluentTriple]) -> Sequence[tf.Tensor]:
        '''Returns output tensors for `fluents`.'''
        return tuple(cls._dtype(t) for t in cls._tensors(fluents))


class ReparameterizationType(Enum):
    FULLY_REPARAMETERIZED = 0
    NOT_REPARAMETERIZED = 1
    PARTIALLY_REPARAMETERIZED = 2


Trajectory = namedtuple('Trajectory', 'states actions interms rewards log_probs')


class MarkovRecurrentModel():

    _FULLY_REPARAMETERIZED_FLAG = 0.0
    _NOT_REPARAMETERIZED_FLAG = 1.0

    def __init__(self, compiler: Compiler, policy: DeepReactivePolicy, batch_size: int) -> None:
        self._policy = policy
        self._cell = MarkovCell(compiler, policy, batch_size)

    @property
    def graph(self):
        '''Returns the computation graph.'''
        return self._cell.graph

    @property
    def batch_size(self) -> int:
        '''Returns the size of the simulation batch.'''
        return self._cell._batch_size

    @property
    def output_size(self) -> Tuple[Sequence[Shape], Sequence[Shape], Sequence[Shape], int, int]:
        '''Returns the simulation output size.'''
        return self._cell.output_size

    def build(self, horizon, reparam_type):
        with self.graph.as_default():
            with tf.name_scope('MRM'):
                self._build_trajectory_graph(horizon, reparam_type)
                self._build_total_reward_graph()
                self._build_surrogate_reward_graph()

    def _build_trajectory_graph(self, horizon, reparam_type):
        self.initial_state = self._cell.initial_state()

        self.timesteps = self._timesteps(horizon)
        self.stop_flags = self._stop_flags(horizon, reparam_type)
        self.inputs = self._inputs(self.timesteps, self.stop_flags)

        self.trajectory = self._trajectory(self.initial_state, self.inputs)

    def _build_total_reward_graph(self):
        with tf.name_scope('total_reward'):
            self.total_reward = tf.squeeze(tf.reduce_sum(self.trajectory.rewards, axis=1))

    def _build_surrogate_reward_graph(self):
        rewards = self.trajectory.rewards
        log_probs = self.trajectory.log_probs

        with tf.name_scope('surrogate_reward'):
            self.q = self._reward_to_go(rewards)

            self.reparam_rewards = tf.where(
                tf.equal(self.stop_flags, self._FULLY_REPARAMETERIZED_FLAG),
                rewards,
                rewards + log_probs * self.q,
                name='reparam_rewards')

            self.total_surrogate_reward = tf.reduce_sum(self.reparam_rewards, axis=1, name='total_surrogate_reward')
            self.surrogate_reward = tf.reduce_mean(self.total_surrogate_reward, name='surrogate_reward')

    def _timesteps(self, horizon: int) -> tf.Tensor:
        '''Returns the input tensor for the given `horizon`.'''
        start, limit, delta = horizon - 1, -1, -1
        timesteps_range = tf.range(start, limit, delta, dtype=tf.float32)
        timesteps_range = tf.expand_dims(timesteps_range, -1)
        batch_timesteps = tf.stack([timesteps_range] * self.batch_size, name='timesteps')
        return batch_timesteps

    def _stop_flags(self, horizon: int, reparam_type: ReparameterizationType):
        shape = (self.batch_size, horizon, 1)
        if reparam_type == ReparameterizationType.FULLY_REPARAMETERIZED:
            flags = tf.constant(self._FULLY_REPARAMETERIZED_FLAG, shape=shape, dtype=tf.float32, name='flags_fully_reparameterized')
        elif reparam_type == ReparameterizationType.NOT_REPARAMETERIZED:
            flags = tf.constant(self._NOT_REPARAMETERIZED_FLAG, shape=shape, dtype=tf.float32, name='flags_not_reparameterized')
        return flags

    def _inputs(self, timesteps, stop_flags):
        return tf.concat([timesteps, stop_flags], axis=2, name='inputs')

    def _trajectory(self, initial_state, inputs) -> Trajectory:
        outputs, final_state = tf.nn.dynamic_rnn(
            self._cell,
            inputs,
            initial_state=initial_state,
            dtype=tf.float32,
            scope="trajectory")
        outputs = [self._output(fluents) for fluents in outputs[:3]] + list(outputs[3:])
        return Trajectory(*outputs)

    def _reward_to_go(self, rewards):
        q = tf.stop_gradient(tf.cumsum(rewards, axis=1, exclusive=True, reverse=True), name='reward_to_go')
        return q

    @classmethod
    def _output(cls, fluents):
        return tuple(f[0] for f in fluents)
