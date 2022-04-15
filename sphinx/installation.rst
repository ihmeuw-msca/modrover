====================
Installing rover
====================

Python version
--------------

The package :code:`rover` is written in Python
and requires Python 3.10 or later.

Install rover
----------------

Regmod package is distributed at
`PyPI <https://pypi.org/project/rover/>`_.
To install the package:

.. code::

   pip install rover

For developers
--------------

For developers, you can clone the repository and install the package in the
development mode.

.. code::

    git clone https://github.com/ihmeuw-msca/rover.git
    cd rover
    pip install -e ".[test,docs]"