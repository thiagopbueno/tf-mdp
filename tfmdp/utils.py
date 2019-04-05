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

from typing import Dict


def get_params_string(config: Dict) -> str:
    '''Returns a canonical configuration string by concatenating its parameters.'''
    params = sorted(config)
    params_string = ['{}={}'.format(p, str(config[p])) for p in params]
    params_string = '&'.join(params_string)
    return params_string
