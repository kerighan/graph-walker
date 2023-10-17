# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['walker']

package_data = \
{'': ['*']}

install_requires = \
['networkx>=3.1,<4.0',
 'numpy>=1.26.1,<2.0.0',
 'pybind11>=2.8.0,<3.0.0',
 'scikit-learn>=1.3.1,<2.0.0',
 'scipy>=1.11.3,<2.0.0']

setup_kwargs = {
    'name': 'walker',
    'version': '1.0.6',
    'description': '',
    'long_description': 'None',
    'author': 'Octavi Font',
    'author_email': 'octavi.fs@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.11,<3.12',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
