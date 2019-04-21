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

initializers = {
    'zeros': tf.initializers.zeros,
    'ones': tf.initializers.ones,
    'random_normal': tf.initializers.random_normal,
    'random_uniform': tf.initializers.random_uniform,
    'truncated_normal': tf.initializers.truncated_normal,
    'glorot': tf.glorot_uniform_initializer
}
