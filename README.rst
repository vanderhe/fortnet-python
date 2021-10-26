*************************************************************
fortnet-python: Python Tools for the Fortnet Software Package
*************************************************************

|license|
|latest version|
|doi|
|issues|

fortnet-python provides tools to generate compatible datasets as well as extract
results obtained by the neural network implementation
`Fortnet <https://github.com/vanderhe/fortnet>`_.

|logo|

Installation
============

Please note, that this package has been tested for Python 3.X support. Its usage
additionally requires

- `numerical Python <https://numpy.org/doc/stable/reference/>`_ (`numpy`)
- `pythonic HDF5 <http://www.h5py.org/>`_ (`h5py`)
- `Atomic Simulation Environment <https://wiki.fysik.dtu.dk/ase/>`_ (`ase`)

as well as the `pytest` framework in order to run the regression tests.

Via the Python Package Index
----------------------------

First, make sure you have an up-to-date version of pip installed::

  python -m pip install --upgrade pip

The package can be downloaded and installed via pip into the active Python
interpreter (preferably using a virtual python environment) by ::

  pip install fortnet-python

or into the user space issueing::

  pip install --user fortnet-python

Locally from Source
-------------------

Alternatively, you can install it locally from source, i.e. from the root folder
of the project::

  python -m pip install .

Testing
=======

The regression testsuite utilizes the `pytest` framework and may be executed by
::

  python -m pytest --basetemp=Testing

Documentation
=============

|docs status|

Consult following resources for documentation:

* `Step-by-step instructions with selected examples (Fortnet Recipes)
  <https://fortnet.readthedocs.io/en/latest/fortformat/index.html>`_

Contributing
============

New features, bug fixes, documentation, tutorial examples and code testing is
welcome during the ongoing fortnet-python development!

The project is
`hosted on github <https://github.com/vanderhe/fortnet-python/>`_.
Please check `CONTRIBUTING.rst <CONTRIBUTING.rst>`_ for guide lines.

I am looking forward to your pull request!

License
=======

fortnet-python is released under the BSD 2-clause license. See the included
`LICENSE <LICENSE>`_ file for the detailed licensing conditions.

.. |logo| image:: ./utils/art/logo.svg
    :alt: Fortnet logo
    :width: 90
    :target: https://github.com/vanderhe/fortnet/

.. |license| image:: https://img.shields.io/github/license/vanderhe/fortnet-python
    :alt: BSD-2-Clause
    :scale: 100%
    :target: https://opensource.org/licenses/BSD-2-Clause

.. |latest version| image:: https://img.shields.io/github/v/release/vanderhe/fortnet-python
    :target: https://github.com/vanderhe/fortnet-python/releases/latest

.. |doi| image:: https://zenodo.org/badge/356394988.svg
   :target: https://zenodo.org/badge/latestdoi/356394988

.. |docs status| image:: https://readthedocs.org/projects/fortnet/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://fortnet-python.readthedocs.io/en/latest/

.. |issues| image:: https://img.shields.io/github/issues/vanderhe/fortnet-python.svg
    :target: https://github.com/vanderhe/fortnet-python/issues/

.. |build status| image:: https://img.shields.io/github/workflow/status/vanderhe/fortnet-python/CI
    :target: https://github.com/vanderhe/fortnet-python/actions/
