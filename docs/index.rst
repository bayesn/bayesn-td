.. _bayesn_td:

BayeSN-TD
===========

About
------

This is the documentation for **bayesn-td**, an extension of
`BayeSN <https://github.com/bayesn/bayesn>`_ for fitting strongly-lensed
Type Ia supernovae. It jointly fits all images of a lensed SN Ia — sharing
the intrinsic SN parameters across images while inferring per-image time
delays, magnifications, and microlensing.

Like BayeSN, bayesn-td is built on `numpyro <https://num.pyro.ai/>`_ and
`JAX <https://jax.readthedocs.io/>`_, so inference can run on GPU when JAX is installed with GPU support.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   intro
   installation
   fitting
   running
   output
   simulation
   models
   filters
   api
   citing

Support
-------

If you find something unclear, come across any bugs, or think of any
functionality you would like, please raise a GitHub issue here:
`<https://github.com/bayesn/bayesn-td/issues>`_.
