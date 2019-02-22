# Copyright 2018 Google LLC
#
# Author: Giovanni Campagna <gcampagn@cs.stanford.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
Created on Aug 6, 2018

@author: gcampagn
'''


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='genie-parser',
    version='0.1dev',
    
    # python 3.6 makes dictionary ordered by default
    # and, while I try not to rely on it, in the past
    # that has slipped through in some places; better
    # to just use the latest version
    # also, no python 2 - that's obsolete 
    python_requires='>=3.6',
    install_requires=[
        'tensorflow>=1.12.0',
        'tensor2tensor',
        'orderedset',
        'numpy'
        'tornado'
        'SQLAlchemy'
        'semver'
    ],
    
    packages=setuptools.find_packages(exclude=['scripts', 'tests', 'tests.*']),
    scripts=['genie-trainer', 'genie-datagen',
             'genie-evaluator', 'genie-decoder',
             'genie-server', 'genie-print-metrics'],
    
    license='GPL-3.0+',
    author="Stanford Mobisocial Lab",
    author_email="mobisocial@lists.stanford.edu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Stanford-Mobisocial-IoT-Lab/genie-parser",
)
