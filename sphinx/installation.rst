====================
Installing modrover
====================

Python version
--------------

The package :code:`modrover` is written in Python
and requires Python 3.10 or later.

Install modrover
----------------

Regmod package is distributed at
`PyPI <https://pypi.org/project/modrover/>`_.
To install the package:

.. code::

   pip install modrover

For developers
--------------

For developers, you can clone the repository and install the package in the
development mode.

.. code::

    git clone https://github.com/ihmeuw-msca/modrover.git
    cd modrover
    pip install -e ".[test,docs]"