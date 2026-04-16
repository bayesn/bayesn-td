.. _installation:

Installation
=============

Requirements
-------------

bayesn-td requires Python 3.11 or later. It depends on the following
packages:

- `jax <https://jax.readthedocs.io/>`_ and jaxlib
- `numpyro <https://num.pyro.ai/>`_
- numpy
- scipy
- pandas
- matplotlib
- astropy
- h5py
- extinction
- arviz
- ruamel.yaml
- tqdm

GPU support
------------

bayesn-td uses JAX as its numerical backend, so GPU acceleration is available
wherever JAX supports it. If you want to run on GPU, you should install the
GPU version of JAX *before* installing bayesn-td — the default pip install is
CPU only. GPU wheels are Linux-only. Full installation instructions can be
found in the
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_.
For example:

.. code-block:: bash

    pip install --upgrade "jax[cuda13]"

If you don't have a GPU, bayesn-td will run on CPU without any additional
configuration.

Installing bayesn-td
----------------------

bayesn-td is not yet available on PyPI. To install from source, clone the
repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/bayesn/bayesn-td.git
    cd bayesn-td
    pip install -e .

This will install bayesn-td along with all of its dependencies, and make the
``run_bayesn_td`` command available on your path.

