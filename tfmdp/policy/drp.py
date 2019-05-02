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

import abc
import json
import os
import tensorflow as tf

from typing import Dict, Optional, Sequence


activation_fn = {
    'none': None,
    'sigmoid': tf.sigmoid,
    'tanh': tf.tanh,
    'relu': tf.nn.relu,
    'relu6': tf.nn.relu6,
    'crelu': tf.nn.crelu,
    'elu': tf.nn.elu,
    'selu': tf.nn.selu,
    'softplus': tf.nn.softplus,
    'softsign': tf.nn.softsign
}


class DeepReactivePolicy(metaclass=abc.ABCMeta):
    ''' DeepReactivePolicy abstract base class.

    It defines the basic API for building, saving and restoring
    reactive policies implemented as deep neural nets.

    A reactive policy defines a mapping from current state fluents
    to action fluents.

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler.
        config (Dict): The reactive policy configuration parameters.
    '''

    def __init__(self, compiler: rddl2tf.compiler.Compiler, config: Dict) -> None:
        self.compiler = compiler
        self.config = config
        self.saver = None

    @abc.abstractproperty
    def name(self) -> str:
        '''Returns the canonical DRP name.'''
        raise NotImplementedError

    @property
    def graph(self):
        return self.compiler.graph

    @abc.abstractproperty
    def size(self) -> int:
        '''Returns the number of trainable parameters.'''
        raise NotImplementedError

    @abc.abstractproperty
    def trainable_variables(self) -> Sequence[tf.Variable]:
        '''Returns a list of the trainable variables.'''
        raise NotImplementedError

    @abc.abstractmethod
    def build(self) -> None:
        '''Create the DRP layers and trainable weights.'''
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    def save(self, sess: tf.Session, path: str, global_step: Optional[int] = None) -> str:
        '''Serializes all DRP trainable variables into a checkpoint file.

        Args:
            sess (:obj:`tf.Session`): A running session.
            path (str): The path to a checkpoint directory.

        Returns:
            str: The path prefix of the newly created checkpoint file.
        '''
        if self.saver is None:
            self.saver = tf.train.Saver(self.trainable_variables, max_to_keep=1000)

        self._checkpoint = self.saver.save(sess, os.path.join(path, 'drp.ckpt'), global_step=global_step)
        return self._checkpoint

    def restore(self, sess: tf.Session, path: Optional[str] = None) -> None:
        '''Restores previously saved DRP trainable variables.

        If path is not provided, restores from last saved checkpoint.

        Args:
            sess (:obj:`tf.Session`): A running session.
            path (Optional[str]): An optional path to a checkpoint directory.
        '''
        if self.saver is None:
            self.saver = tf.train.Saver()

        if path is None:
            path = self._checkpoint

        self.saver.restore(sess, path)

    def save_config(self, path: str) -> None:
        '''Serializes policy configuration json file to `path` directory.'''
        filename = os.path.join(path, 'drp.config.json')
        with open(filename, 'w') as file:
            file.write(self.to_json())

    def to_json(self) -> str:
        '''Returns the policy configuration parameters serialized in JSON format.'''
        json_config = {
            'class': self.__class__.__name__,
            'params': { **self.config }
        }
        if self.saver:
            json_config['checkpoints'] = self.saver.last_checkpoints
        return json.dumps(json_config, sort_keys=True, indent=4)

    @classmethod
    def from_json(cls, compiler: rddl2tf.compiler.Compiler,
                       json_config: str) -> 'DeepReactivePolicy':
        '''Instantiates a DRP from a `json_config` string.

        Args:
            compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler.
            json_config (str): A DRP configuration encoded in JSON format.

        Returns:
            :obj:`tfmdp.policy.drp.DeepReactivePolicy`: A DRP object.
        '''
        config = json.loads(json_config)
        return cls(compiler, config['params'])

    def summary(self) -> None:
        '''Prints a string summary of the DRP.'''
        print(self.to_json())
