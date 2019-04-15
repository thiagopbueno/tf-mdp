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
from tfmdp.policy.drp import DeepReactivePolicy
from tfrddlsim.simulation.policy_simulator import PolicySimulator

import numpy as np
import tensorflow as tf


class Value():

    def __init__(self, compiler, policy):
        self._compiler = compiler
        self._policy = policy

    @property
    def graph(self):
        return self._compiler.graph

    def build(self, horizon, batch_size, learning_rate=0.001):
        self._horizon = horizon
        self._batch_size = batch_size

        with self.graph.as_default():
            with tf.name_scope('value_fn'):
                self._build_trajectory_ops(horizon, batch_size)
                self._build_value_estimates_ops()
                self._build_regression_ops()
                self._build_prediction_ops(horizon)
                self._build_loss_ops()
                self._build_optimization_ops(learning_rate)
                self._build_initialization_ops()

    def fit(self, sess, batch_size, epochs, show_progress=True):

        with self.graph.as_default():
            sess.run(self._init_op)

            dataset = self._regression_dataset(sess)

            losses = []

            for step in range(epochs):

                for state, timestep, target in self._training_batch_generator(batch_size, dataset):

                    feed_dict = {
                        self._step: timestep,
                        self._targets: target,
                        self._training: True
                    }

                    for fluent, feature in zip(state, self._inputs):
                        feed_dict[feature] = fluent

                    loss_, _ = sess.run([self._loss, self._train_op], feed_dict=feed_dict)
                    losses.append(loss_)

                if show_progress:
                    print('Epoch {0:5}: loss = {1:3.6f}\r'.format(step, loss_), end='')

            return losses

    def save(self, sess, save_path=None):
        if self._saver is None:
            self._saver = tf.train.Saver(self._trainable_variables())
        if save_path is None:
            save_path = '/tmp/{}/model.ckpt'.format(self.name)
        self._checkpoint = self._saver.save(sess, save_path)
        return self._checkpoint

    def restore(self, sess, save_path=None):
        if self._saver is None:
            self._saver = tf.train.Saver(self._trainable_variables())
        if save_path is None:
            save_path = self._checkpoint
        self._saver.restore(sess, save_path)

    def __call__(self, state, timestep):
        state_fluents = self._compiler.rddl.domain.state_fluent_ordering
        state_size = self._compiler.rddl.state_size

        with self.graph.as_default():

            t = tf.one_hot(timestep, self._horizon, name='time_one_hot_encoding')

            outputs = []

            for name, fluent, size in zip(state_fluents, state, state_size):

                variable_name_scope = 'valuefn/prediction/{}'.format(name.replace('/', '-'))

                with tf.variable_scope(variable_name_scope, reuse=tf.AUTO_REUSE):
                    size = np.prod(size, dtype=np.int32)

                    fluent = tf.reshape(fluent, [-1, size], name='fluent_input')
                    fluent_norm = tf.layers.batch_normalization(fluent, training=self._training)

                    weights = tf.layers.dense(t, size, name='weigths')
                    logits = tf.multiply(weights, fluent_norm, name='logits')
                    logits = tf.layers.dense(logits, 128, activation=tf.nn.elu)

                    outputs.append(logits)

            logits = tf.concat(outputs, axis=1, name='logits')

            prediction = tf.nn.relu(tf.reduce_sum(logits, axis=1), name='value_estimate')
            return prediction

    def _build_trajectory_ops(self, horizon, batch_size):
        self._simulator = PolicySimulator(self._compiler, self._policy, batch_size)
        trajectories = self._simulator.trajectory(horizon)
        self._states = trajectories[1]
        self._rewards = tf.squeeze(-trajectories[4], name='rewards')
        self._timesteps = tf.squeeze(tf.cast(self._simulator.inputs, tf.int32), name='timesteps')

    def _build_value_estimates_ops(self):
        self._estimates = tf.cumsum(self._rewards, axis=1, exclusive=False, reverse=True)

    def _build_regression_ops(self):
        self._features = []
        for fluent in self._states:
            feature = tf.unstack(fluent, axis=1)
            feature = tf.concat(feature, axis=0)
            self._features.append(feature)
        timesteps = tf.concat(tf.unstack(self._timesteps, axis=1), axis=0)
        self._dataset_features = (timesteps, tuple(self._features))
        self._dataset_targets = tf.squeeze(tf.concat(tf.unstack(self._estimates, axis=1), axis=0))

    def _build_prediction_ops(self, horizon):
        state_fluents = self._compiler.rddl.domain.state_fluent_ordering
        state_size = self._compiler.rddl.state_size

        with tf.name_scope('prediction'):
            self._inputs = []
            for name, size in zip(state_fluents, state_size):
                shape = [None] + list(size)
                fluent = tf.placeholder(tf.float32, shape=shape, name='fluent_input')
                self._inputs.append(fluent)
            state = tuple(self._inputs)

            self._step = tf.placeholder(tf.int32, shape=(None,), name='step_input')

            self._training = tf.placeholder(tf.bool, shape=(), name='training_flag')
            self._predictions = self.__call__(state, self._step)

    def _build_loss_ops(self):
        with tf.name_scope('loss'):
            self._targets = tf.placeholder(tf.float32, shape=[None], name='targets')
            self._loss = tf.reduce_mean(tf.square(self._predictions - self._targets), name='mse')

    def _build_optimization_ops(self, learning_rate):
        self._optimizer = tf.train.RMSPropOptimizer(learning_rate)
        # self._optimizer = tf.train.AdamOptimizer(learning_rate)
        self._grad_and_vars = self._optimizer.compute_gradients(self._loss)
        self._grad_and_vars = [(grad, var) for grad, var in self._grad_and_vars if grad is not None]
        self._train_op = self._optimizer.apply_gradients(self._grad_and_vars)

    def _build_initialization_ops(self):
        valuefn_vars = self._trainable_variables()
        self._init_op = tf.variables_initializer(valuefn_vars)

    def _regression_dataset(self, sess):
        features, targets = sess.run([self._dataset_features, self._dataset_targets])
        timesteps, states = features
        return timesteps, states, targets

    def _training_batch_generator(self, batch_size, dataset):
        timesteps, states, targets = dataset

        dataset_size = self._batch_size * self._horizon
        perm = np.random.permutation(dataset_size)

        timesteps = timesteps[perm]
        states = tuple(fluent[perm] for fluent in states)
        targets = targets[perm]

        i = 0
        while i < dataset_size:
            state = tuple(fluent[i:i+batch_size, :] for fluent in states)
            t = timesteps[i:i+batch_size]
            target = targets[i:i+batch_size]
            yield (state, t, target)
            i += batch_size

    def _trainable_variables(self):
        return tf.trainable_variables(r'valuefn/prediction')
