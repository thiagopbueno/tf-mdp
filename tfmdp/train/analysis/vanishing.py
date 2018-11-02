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


import matplotlib.pyplot as plt
import numpy as np


class VanishingGradientAnalysis():

    def __init__(self, filepath):
        self.filepath = filepath

    def setup(self, optimizer, model):
        self._optimizer = optimizer
        self._model = model

        horizon = self._model.horizon

        horizons = [1, horizon/2, horizon-1]
        losses = []
        self._vanishing_grad = []
        self._vanishing_grad_stats = []
        for t in horizons:
            costs = self._model.surrogate_batch_cost[:, int(t), 0]
            loss = tf.reduce_mean(costs)
            self._vanishing_grad.append(tf.gradients(ys=loss, xs=self._model.initial_state)[0])
            self._vanishing_grad_stats.append([int(t), { 'var': [], 'mean': [] }])

    def __call__(self, sess, step):
        grads = []
        for _ in range(30):
            grad_ = sess.run(self._vanishing_grad)
            grads.append([np.linalg.norm(g) for g in grad_])

        for i, stats in enumerate(self._vanishing_grad_stats):
            i_grads = [grad[i] for grad in grads]
            # stats[1]['var'].append(np.var(i_grads))
            stats[1]['mean'].append(np.mean(i_grads))

    def teardown(self):
        for stats in self._vanishing_grad_stats:
            t = stats[0]
            # variance = stats[1]['var']
            # plt.plot(variance, label='var[t={}]'.format(t))
            mean = stats[1]['mean']
            plt.plot(mean, label='mean[t={}]'.format(t))

        plt.title(TITLE)
        plt.legend()
        plt.grid()
        plt.savefig(self.filepath, format='pdf')