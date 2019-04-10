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


import tensorflow as tf


optimizers = {
    'Adadelta': tf.train.AdadeltaOptimizer,
    'Adagrad': tf.train.AdagradOptimizer,
    'Adam': tf.train.AdamOptimizer,
    'GradientDescent': tf.train.GradientDescentOptimizer,
    'ProximalGradientDescent': tf.train.ProximalGradientDescentOptimizer,
    'ProximalAdagrad': tf.train.ProximalAdagradOptimizer,
    'RMSProp': tf.train.RMSPropOptimizer
}
