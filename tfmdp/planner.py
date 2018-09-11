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
from tfmdp.train.optimizer import PolicyOptimizer


class PolicyOptimizationPlanner(object):

    def __init__(self, compiler, layers, layer_norm, logdir=None):
        self._compiler = compiler
        self._policy = DeepReactivePolicy(self._compiler, layers, layer_norm)
        self._optimizer = PolicyOptimizer(self._compiler, self._policy, logdir)

    def build(self, learning_rate, batch_size, horizon):
        self._optimizer.build(learning_rate, batch_size, horizon)

    def run(self, epochs, show_progress=True):
        losses, rewards = self._optimizer.run(epochs, show_progress=show_progress)
        logdir = self._optimizer._train_writer.get_logdir()
        return rewards, self._policy, logdir
