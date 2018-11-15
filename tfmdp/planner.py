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
from tfmdp.train.valuefn import Value
from tfmdp.train.mrm import MarkovRecurrentModel, ReparameterizationType
from tfmdp.train.optimizer import PolicyOptimizer

import tensorflow as tf


class PolicyOptimizationPlanner(object):

    _loss_fn ={
        'linear': lambda c: tf.abs(0 - c),
        'mse': lambda c: tf.square(0 - c),
        # 'huber': tf.losses.huber_loss
    }

    _non_linearities = {
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

    _optimizers = {
        'Adadelta': tf.train.AdadeltaOptimizer,
        'Adagrad': tf.train.AdagradOptimizer,
        'Adam': tf.train.AdamOptimizer,
        'GradientDescent': tf.train.GradientDescentOptimizer,
        'ProximalGradientDescent': tf.train.ProximalGradientDescentOptimizer,
        'ProximalAdagrad': tf.train.ProximalAdagradOptimizer,
        'RMSProp': tf.train.RMSPropOptimizer
    }

    def __init__(self,
            compiler,
            layers, activation, input_layer_norm, hidden_layer_norm,
            logdir=None):
        self._compiler = compiler
        self._logdir = logdir

        self._policy = DeepReactivePolicy(
            self._compiler,
            layers, self._non_linearities[activation],
            input_layer_norm, hidden_layer_norm)

    def build(self,
            learning_rate, batch_size, horizon,
            optimizer='RMSProp',
            loss='linear',
            kernel_l1_regularizer=None, kernel_l2_regularizer=None,
            bias_l1_regularizer=None, bias_l2_regularizer=None,
            reparameterization_type=None,
            baseline_flag=False):

        self._baseline_flag = baseline_flag
        self._valuefn = None
        if baseline_flag:
            self._valuefn = Value(self._compiler, self._policy)
            self._valuefn.build(horizon, 128)

        self._model = MarkovRecurrentModel(self._compiler, self._policy, batch_size)
        if reparameterization_type is None:
            reparameterization_type = ReparameterizationType.FULLY_REPARAMETERIZED

        self._model.build(horizon, self._loss_fn[loss], reparameterization_type, self._valuefn)

        self._optimizer = PolicyOptimizer(self._model, self._logdir, debug=False)
        self._optimizer.build(
            learning_rate, batch_size, horizon,
            self._optimizers[optimizer],
            self._get_regularizer(kernel_l1_regularizer, kernel_l2_regularizer),
            self._get_regularizer(bias_l1_regularizer, bias_l2_regularizer))

    def run(self, epochs, show_progress=True):
        losses, rewards = self._optimizer.run(epochs, show_progress=show_progress, baseline_flag=self._baseline_flag)
        logdir = self._optimizer._train_writer.get_logdir()
        return rewards, self._policy, logdir

    def _get_regularizer(self, l1_regularizer, l2_regularizer):
        regularizer = []
        if l1_regularizer != 0.0:
            regularizer.append(tf.contrib.layers.l1_regularizer(l1_regularizer))
        if l2_regularizer != 0.0:
            regularizer.append(tf.contrib.layers.l2_regularizer(l2_regularizer))
        if regularizer:
            return tf.contrib.layers.sum_regularizer(regularizer)
