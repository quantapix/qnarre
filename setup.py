# Copyright 2019 Quantapix Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import re

from os import path
from codecs import open
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

with open(path.join(here, 'qnarre', '__init__.py')) as f:
    m = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if m:
        version = m.group(1)
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name='qnarre',
    version=version,
    description="Qnarre project",
    long_description=long_description,
    url='https://github.com/quantapix/qnarre.git',
    author='Quantapix, Inc.',
    author_email='quantapix@gmail.com',
    license='Apache-2.0',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache-2.0 License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.7',
    ],

    # What does your project relate to?
    keywords='quantapix development',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['tests']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['networkx', 'markdown'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'web': ['django'],
        # 'dev': ['check-manifest'],
        # 'test': ['coverage'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        # 'qnarre': ['def_junks.txt'],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[],  # ('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'qfilt-mbox=qnarre.command:filt_mbox',
            'qmerge-mbox=qnarre.command:merge_mbox',
            'qstrip-mbox=qnarre.command:strip_mbox',
            'qimp-main=qnarre.command:import_main',
            'qimp-blog=qnarre.command:import_blog',
            'qimp-priv=qnarre.command:import_priv',
            'qimp-docs=qnarre.command:import_docs',
            'qimp-sbox=qnarre.command:import_sbox',
            'qimp-mbox=qnarre.command:import_mbox',
            'qimp-tbox=qnarre.command:import_tbox',
            'qimp-bbox=qnarre.command:import_bbox',
            'qimp-pics=qnarre.command:import_pics',
            'qprotect=qnarre.command:protect',
            'qredact=qnarre.command:redact',
            'qobfuscate=qnarre.command:obfuscate',
            'qcheck-recs=qnarre.command:check_recs',
            'qgraph-recs=qnarre.command:graph_recs',
            'qnn-setup=qnarre.command:qnn_setup',
            'qnn-learn=qnarre.command:qnn_learn',
            'qnn-guess=qnarre.command:qnn_guess',
            'qexp-mbox=qnarre.command:export_mbox',
            'qexp-blog=qnarre.command:export_blog',
            'qexp-pngs=qnarre.command:export_pngs',
            'qexp-jpgs=qnarre.command:export_jpgs',
            'qexp-orgs=qnarre.command:export_orgs',
        ],
    },

    # test_suite='nose.collector',
    # tests_require=['nose'],
)
