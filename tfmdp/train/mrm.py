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
from tfmdp.train.valuefn import Value

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
        return self._sizes(self._compiler.state_size)

    @property
    def action_size(self) -> Sequence[Shape]:
        '''Returns the MDP action size.'''
        return self._sizes(self._compiler.action_size)

    @property
    def interm_size(self) -> Sequence[Shape]:
        '''Returns the MDP intermediate state size.'''
        return self._sizes(self._compiler.interm_size)

    @property
    def output_size(self) -> Tuple[Sequence[Shape], Sequence[Shape], Sequence[Shape], int, int]:
        '''Returns the simulation cell output size.'''
        return (self.state_size, self.action_size, self.interm_size, 1, 1)

    def initial_state(self) -> StateTensor:
        '''Returns the initial state tensor.'''
        s0 = []
        for fluent in self._compiler.compile_initial_state(self._batch_size):
            s0.append(self._output_size(fluent))
        s0 = tuple(s0)
        return s0

    def __call__(self,
            input: tf.Tensor,
            state: Sequence[tf.Tensor],
            scope: Optional[str] = None) -> Tuple[CellOutput, CellState]:

        # inputs
        timestep, reparam = self._inputs(input)

        # action
        action = self._policy(state, timestep)

        # next state
        transition_scope = self._compiler.transition_scope(state, action)
        interm_fluents, next_state_fluents = self._compiler.compile_probabilistic_cpfs(transition_scope, self._batch_size, reparam)

        # log_probs
        log_prob = self._log_prob(interm_fluents, next_state_fluents)

        # reward
        transition_scope.update({name: fluent for name, fluent, _ in next_state_fluents})
        reward = self._compiler.compile_reward(transition_scope)
        reward = self._output_size(reward.tensor)

        # outputs
        interm_state = self._output(interm_fluents)
        next_state = self._output(next_state_fluents)
        output = (next_state, action, interm_state, reward, log_prob)

        return (output, next_state)

    def _inputs(self, input):
        timestep, flags = tf.split(input, [1, 1], axis=1)
        reparam = self._reparam(flags)
        return timestep, reparam

    def _reparam(self, reparam_flag):
        with self.graph.as_default():
            with tf.name_scope('reparam_flags'):
                return tf.equal(tf.squeeze(reparam_flag), MarkovRecurrentModel.FULLY_REPARAMETERIZED_FLAG)

    def _log_prob(self, interm_fluents, next_state_fluents):
        with self.graph.as_default():
            with tf.name_scope('transition_log_prob'):

                interm_log_probs = [log_prob.tensor for _, _, log_prob in interm_fluents if log_prob is not None]
                if len(interm_log_probs) == 0:
                    interm_log_prob = tf.constant(0.0, dtype=tf.float32, shape=(self._batch_size,))
                else:
                    interm_log_prob = tf.reduce_sum(
                        tf.concat(interm_log_probs, axis=1, name='interm_log_probs'),
                        axis=1,
                        name='interm_log_prob')

                next_state_log_probs = [log_prob.tensor for _, _, log_prob in next_state_fluents if log_prob is not None]
                if len(next_state_log_probs) == 0:
                    next_state_log_prob = tf.constant(0.0, dtype=tf.float32, shape=(self._batch_size,))
                else:
                    next_state_log_prob = tf.reduce_sum(
                        tf.concat(next_state_log_probs, axis=1, name='next_state_log_probs'),
                        axis=1,
                        name='next_state_log_prob')

                log_prob = tf.expand_dims(interm_log_prob + next_state_log_prob, -1, name='log_prob')
                return log_prob

    @classmethod
    def _sizes(cls, sizes: Sequence[Shape]) -> Sequence[Union[Shape, int]]:
        return tuple(sz if sz != () else (1,) for sz in sizes)

    @classmethod
    def _output_size(cls, tensor):
        if tensor.shape.ndims == 1:
            tensor = tf.expand_dims(tensor, -1)
        return tensor

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

    FULLY_REPARAMETERIZED_FLAG = 0.0
    NOT_REPARAMETERIZED_FLAG = 1.0

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

    def build(self, horizon, loss_op, reparam_type=ReparameterizationType.FULLY_REPARAMETERIZED, baseline_fn=None):
        self.horizon = horizon
        self._baseline_fn = baseline_fn
        with self.graph.as_default():
            with tf.name_scope('MRM'):
                self._build_trajectory_graph(horizon, reparam_type)
                self._build_total_cost_graph()
                if baseline_fn:
                    self._build_baseline_graph(baseline_fn)
                self._build_surrogate_cost_graph(loss_op, baseline_fn)

    def _build_trajectory_graph(self, horizon, reparam_type):
        self.initial_state = self._cell.initial_state()

        self.timesteps = self._timesteps(horizon)
        self.reparam_flags = self._reparam_flags(horizon, reparam_type)
        self.inputs = self._inputs(self.timesteps, self.reparam_flags)

        self.trajectory = self._trajectory(self.initial_state, self.inputs)

    def _build_total_cost_graph(self):
        with tf.name_scope('total_reward'):
            self.total_reward = tf.reduce_sum(tf.squeeze(self.trajectory.rewards), axis=1)

    def _build_baseline_graph(self, baseline_fn):
        states = self.trajectory.states

        with tf.name_scope('baseline'):
            b_t = []
            for t in range(self.horizon):
                state = tuple(fluent[:, t, :] for fluent in states)
                t = tf.cast(tf.squeeze(self.timesteps[:, t, :]), tf.int32)
                b = baseline_fn(state, t)
                b_t.append(b)
            self._baseline = tf.stop_gradient(tf.expand_dims(tf.stack(b_t, axis=1), -1))

    def _build_surrogate_cost_graph(self, loss_op, baseline_fn):
        rewards = self.trajectory.rewards
        log_probs = self.trajectory.log_probs

        with tf.name_scope('surrogate_cost'):

            self.costs = loss_op(rewards)

            self.q = self._reward_to_go(self.costs)

            if baseline_fn:
                self.surrogate_batch_cost = tf.where(
                    tf.equal(self.reparam_flags, self.FULLY_REPARAMETERIZED_FLAG),
                    self.costs,
                    self.costs + log_probs * (self.q - self._baseline),
                    name='batch_cost')
            else:
                self.surrogate_batch_cost = tf.where(
                    tf.equal(self.reparam_flags, self.FULLY_REPARAMETERIZED_FLAG),
                    self.costs,
                    self.costs + log_probs * self.q,
                    name='batch_cost')

            # self.total_surrogate_reward = tf.reduce_sum(tf.squeeze(self.reparam_rewards), axis=1, name='total_surrogate_reward')

    def _timesteps(self, horizon: int) -> tf.Tensor:
        '''Returns the input tensor for the given `horizon`.'''
        with tf.name_scope('timesteps'):
            start, limit, delta = horizon - 1, -1, -1
            timesteps_range = tf.range(start, limit, delta, dtype=tf.float32)
            timesteps_range = tf.expand_dims(timesteps_range, -1)
            batch_timesteps = tf.stack([timesteps_range] * self.batch_size)
        return batch_timesteps

    def _reparam_flags(self, horizon: int, reparam_type: ReparameterizationType):
        shape = (self.batch_size, horizon, 1)
        if reparam_type == ReparameterizationType.FULLY_REPARAMETERIZED:
            flags = tf.constant(self.FULLY_REPARAMETERIZED_FLAG, shape=shape, dtype=tf.float32, name='flags_fully_reparameterized')
        elif reparam_type == ReparameterizationType.NOT_REPARAMETERIZED:
            flags = tf.constant(self.NOT_REPARAMETERIZED_FLAG, shape=shape, dtype=tf.float32, name='flags_not_reparameterized')
        elif isinstance(reparam_type, tuple):
            if reparam_type[0] == ReparameterizationType.PARTIALLY_REPARAMETERIZED:
                flags = []
                for t in range(horizon):
                    if (t+1) % reparam_type[1] == 0:
                        flags.append(self.NOT_REPARAMETERIZED_FLAG)
                    else:
                        flags.append(self.FULLY_REPARAMETERIZED_FLAG)
                with tf.name_scope('flags_n_step_partially_reparameterized'):
                    flags = tf.stack([flags] * self.batch_size, name='')
                    flags = tf.expand_dims(flags, -1)
        return flags

    def _inputs(self, timesteps, reparam_flags):
        return tf.concat([timesteps, reparam_flags], axis=2, name='inputs')

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
