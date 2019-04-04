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
import tensorflow as tf
from typing import Dict


class MarkovRecurrentModel(metaclass=abc.ABCMeta):
    '''MarkovRecurrentMode abstract base class.

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler.
        config (Dict): The recurrent model configuration parameters.
    '''

    def __init__(self, compiler: rddl2tf.compiler.Compiler, config: Dict) -> None:
        self.compiler = compiler
        self.config = config

    @abc.abstractmethod
    def build(self, policy: DeepReactivePolicy) -> None:
        '''Builds the recurrent model ops by integrating the `policy` in the trajectory sampling.

        Args:
            policy (:obj:`tfmdp.policy.drp.DeepReactivePolicy`): A deep reactive policy.
        '''
        raise NotImplementedError

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
            :obj:`tfmdp.model.mrm.MarkovRecurrentModel`: A MarkovRecurrentModel object.
        '''
        config = json.loads(json_string)
        return cls(compiler, config)

    @abc.abstractmethod
    def summary(self) -> None:
        '''Prints a string summary of the recurrent model.'''
        raise NotImplementedError
