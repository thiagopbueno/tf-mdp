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


from tfmdp.train.policy import DeepReactivePolicy

from rddl2tf.compiler import Compiler
from tfrddlsim.simulation.policy_simulator import PolicySimulator

import time
import numpy as np
import tensorflow as tf

from typing import Sequence, Tuple
NonFluentsArray = Sequence[np.array]
StateArray = Sequence[np.array]
StatesArray = Sequence[np.array]
ActionsArray = Sequence[np.array]
IntermsArray = Sequence[np.array]
SimulationOutput = Tuple[NonFluentsArray, StateArray, StatesArray, ActionsArray, IntermsArray, np.array]


class PolicyEvaluator(object):
    '''PolicyEvaluator is a wraper around tfrddlsim.simulation.PolicySimulator
    for evaluating deep reactive policies.

    Args:
        compiler (:obj:`tfrddlsim.rddl2tf.compiler.Compiler`): A RDDL2TensorFlow compiler.
        policy (:obj:`tfmdp.train.policy.DeepReactivePolicy`): A deep reactive policy.
    '''

    def __init__(self, compiler: Compiler, policy: DeepReactivePolicy) -> None:
        self._compiler = compiler
        self._policy = policy

    @property
    def graph(self) -> tf.Graph:
        '''Returns the compiler's graph.'''
        return self._compiler.graph

    def run(self, horizon: int, batch_size: int) -> SimulationOutput:
        '''Runs the trajectory simulation ops for the deep reactive policy.

        Returns:
            Tuple[StateArray, StatesArray, ActionsArray, IntermsArray, np.array]: Simulation output tuple.
        '''
        start = time.time()
        self._simulator = PolicySimulator(self._compiler, self._policy, batch_size)
        trajectories = self._simulator.trajectory(horizon)
        end = time.time()
        building_time = end - start

        start = time.time()
        with tf.Session(graph=self.graph) as sess:
            self._policy.restore(sess)
            trajectories_ = sess.run(trajectories)
        end = time.time()
        inference_time = end - start

        return trajectories_, building_time, inference_time
