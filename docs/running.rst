.. _running:

Running from the command line
================================

bayesn-td can also be run from the command line via the ``run_bayesn_td``
command. This reads a YAML input file specifying the data and configuration,
runs the MCMC fitting, and saves the results to disk. For using bayesn-td
from within Python or a Jupyter notebook, see :ref:`fitting`.

Basic usage
------------

.. code-block:: bash

    run_bayesn_td input.yaml

YAML input file format
-----------------------

A typical input file looks like:

.. code-block:: yaml

    # BayeSN model to use
    load_model: G26x_model

    # MCMC configuration
    num_samples: 500
    num_warmup: 500
    num_chains: 4
    chain_method: parallel
    init_strategy: median
    include_eps: true
    include_ml: true

    # Output directory
    outputdir: results/my_fit

    # Data specification
    # Column names are auto-detected from the file header.
    # peak_mjds, z, and ebv_mw are read from the ECSV metadata if present.
    data:
      photometry: path/to/photometry.ecsv
      peak_mjds: [60000.0, 60050.0, 60080.0]
      z: 1.5
      ebv_mw: 0.05
      band_map:
        f150w: F150W
        f200w: F200W
      error_floor:
        F150W: 0.06
      sn_name: my_sn

Each key is described in the following sections.

Top-level keys
~~~~~~~~~~~~~~~

- ``load_model`` — BayeSN model name or path to a BAYESN.YAML file. Built-in
  options include ``G26x_model``, ``T21_model``, ``M20_model``, ``W22_model``,
  ``W22x_model``. Default: ``G26x_model``.
- ``num_samples`` — number of posterior samples per chain. Default: 500.
- ``num_warmup`` — number of warmup steps per chain. Default: 500.
- ``num_chains`` — number of MCMC chains. Default: 4.
- ``chain_method`` — ``parallel``, ``sequential``, or ``vectorized``.
  Default: ``parallel``.
- ``init_strategy`` — NUTS initialisation. ``median`` starts at the prior
  median; ``sample`` draws a random prior sample. Default: ``median``.
- ``include_eps`` — whether to include :math:`\epsilon` in the model.
  Default: ``true``.
- ``include_ml`` — whether to include the microlensing GP. Default: ``true``.
- ``outputdir`` — directory for output files. Default: ``results/`` in the
  current working directory.

Data keys
~~~~~~~~~~

The ``data`` block specifies the photometry and metadata. Column names are
auto-detected from the file header; you only need to specify them if your
file uses non-standard names.

- ``photometry`` — path to the photometry file (whitespace-delimited,
  CSV, or ECSV).
- ``image_col``, ``time_col``, ``band_col`` — column name overrides
  (optional; auto-detected by default).
- ``mag_col``, ``magerr_col`` — magnitude column name overrides (optional).
- ``flux_col``, ``fluxerr_col`` — flux column name overrides (optional;
  must be FLUXCAL system, zeropoint 27.5).
- ``peak_mjds`` — estimated peak MJD for each image, ordered by
  sorted unique values of the image column. Optional if the photometry file
  is ECSV with ``peak_mjds`` in its header.
- ``z`` — source redshift. Optional if read from ECSV header.
- ``ebv_mw`` — Milky Way :math:`E(B-V)`. Optional if read from ECSV header.
- ``band_map`` — mapping from data band names to BayeSN filter names
  (optional).
- ``error_floor`` — per-band error floor in magnitudes (optional).
- ``time_format`` — ``mjd`` (default) or ``phase``.
- ``sn_name`` — a name for the SN, used in output files (optional).
- ``true_delta_t`` — true time delays, if known (e.g. for simulated data).
  Recorded in the output for comparison but not used in fitting.

Command-line overrides
-----------------------

Top-level YAML keys can be overridden from the command line. Note that nested
keys under ``data`` can't be overridden this way — to change those, edit the
YAML file directly.

.. code-block:: bash

    run_bayesn_td input.yaml --load_model T21_model --num_chains 4

The following command-line arguments are available:

- ``--filters`` — path to a custom filter YAML file.
- ``--outputdir`` — output directory.
- ``--load_model`` — BayeSN model name or path.
- ``--num_chains`` — number of chains.
- ``--num_warmup`` — number of warmup steps.
- ``--num_samples`` — number of posterior samples.
- ``--chain_method`` — chain distribution method.
- ``--init_strategy`` — initialisation strategy.
- ``--include_eps`` — include epsilon (``true`` or ``false``).
- ``--include_ml`` — include microlensing GP (``true`` or ``false``).

Example
--------

A complete working example with simulated LSST photometry of a 3-image lensed
SN Ia is bundled in ``examples/sim_lensed_sn/``. To run it from the repository
root:

.. code-block:: bash

    run_bayesn_td examples/sim_lensed_sn/input.yaml

See ``examples/sim_lensed_sn/README.md`` for the true input parameters.
