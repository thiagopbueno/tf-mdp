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


class GradientVarianceAnalysis():

    def __init__(self, filepath):
        self.filepath = filepath

    def setup(self, optimizer, model):
        self._optimizer = optimizer
        self._grad_variance = { var.name: [] for _, var in self._optimizer._grad_and_vars }

    def __call__(self, sess, step):
        grads = []
        var_names = []
        estimates = []
        for i, (grad, var)  in enumerate(self._optimizer._grad_and_vars):
            grads.append(grad)
            estimates.append([])
            var_names.append(var.name)

        for _ in range(30):
            grads_ = sess.run(grads)
            for i, grad_ in enumerate(grads_):
                estimates[i].append(grad_)

        for name, gs in zip(var_names, estimates):
            grad_norms = [np.linalg.norm(grad) for grad in gs]
            grad_norm_variance = np.var(grad_norms)
            self._grad_variance[name].append(grad_norm_variance)

    def teardown(self):
        for name, variance in self._grad_variance.items():
            plt.plot(variance, label=name)
        plt.grid()
        plt.legend(loc='upper right')
        plt.title(TITLE)
        plt.savefig(self.filepath, format='pdf')
