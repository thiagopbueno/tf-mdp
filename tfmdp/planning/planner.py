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

import tfmdp.train.losses
import tfmdp.train.optimizers
# from tfmdp.train.callbacks import Callback

import abc
import tensorflow as tf

from typing import Callable, Dict, List, Optional, Sequence

Callback = Callable[[None],None]
Callbacks = Dict[str, Sequence[Callback]]


class PolicyOptimizationPlanner(metaclass=abc.ABCMeta):
    '''PolicyOptimizationPlanner abstract base class.

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler.
        config (Dict): The planner configuration parameters.
    '''

    def __init__(self, compiler: rddl2tf.compiler.Compiler, config: Dict) -> None:
        self.compiler = compiler
        self.config = config

    @abc.abstractmethod
    def build(self, policy: DeepReactivePolicy,
                    loss: str,
                    optimizer: str) -> None:
        '''Builds the planner by building and integrating each policy optimization component.

        Args:
            policy (:obj:`tfmdp.policy.drp.DeepReactivePolicy`): A deep reactive policy.
            loss (str): A differentiable loss function used to train the policy.
            optimizer (str): A gradient descent optimizer.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, epochs: int, callbacks: Optional[Callbacks] = None) -> None:
        '''Runs the policy optimizer for a given number of `epochs`.

        Optionally, it executes `callbacks` to extend planning behavior
        during training.

        Args:
            epochs (int): The number of training epochs.
            callbacks (Optional[Dict[str, List[Callback]]]): Mapping from events to lists of callables.
        '''
        raise NotImplementedError

    def to_json(self) -> str:
        '''Returns the planner configuration parameters serialized in JSON format.'''
        return json.dumps(self.config, sort_keys=True, indent=4)

    @classmethod
    def from_json(cls, compiler: rddl2tf.compiler.Compiler,
                       json_config: str) -> 'PolicyOptimizationPlanner':
        '''Instantiates a policy optimization planner from a `json_config` string.

        Args:
            compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler.
            json_config (str): A planner configuration encoded in JSON format.

        Returns:
            :obj:`tfmdp.policy.drpOptimizationPlanner`: A planner object.
        '''
        config = json.loads(json_string)
        return cls(compiler, config)

    @abc.abstractmethod
    def summary(self) -> None:
        '''Prints a string summary of the planner.'''
        raise NotImplementedError
