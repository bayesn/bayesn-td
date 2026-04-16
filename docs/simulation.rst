.. _simulation:

Simulating lensed light curves
================================

bayesn-td can simulate multi-image lensed SN Ia light curves using the
``simulate_light_curve_ml`` method. Note that the simulator does not currently
include microlensing — simulated light curves have ``beta_t = 0``.

Basic usage
------------

.. code-block:: python

    import numpy as np
    from bayesn_td import SEDmodel

    model = SEDmodel(load_model='G26x_model')

    # Observation MJDs and bands
    mjds = np.arange(59990, 60101, 3.0)
    bands = ['F115W', 'F150W', 'F200W']

    # Per-image peak MJDs for a triply-imaged SN (N=1 SN, 3 images)
    peak_mjds = np.array([[60000.0, 60015.0, 60040.0]])  # shape (N, n_images)

    # Simulate
    data, yerr, params = model.simulate_light_curve_ml(
        mjds=mjds,
        N=1,
        bands=bands,
        peak_mjds=peak_mjds,
        z=1.5,
        yerr=0.05,
        mu=np.array([[44.0, 43.5, 43.0]]),
        ebv_mw=0.01,
    )

This returns three objects: ``data`` (the simulated magnitudes or fluxes, with
noise applied, shape ``(n_total_obs, n_images, N)``), ``yerr`` (the
corresponding uncertainties), and ``params`` (a dictionary of the true
parameter values used in the simulation, including ``del_M``, ``AV``,
``theta``, ``eps``, ``z``, ``mu``, ``ebv_mw``, and ``RV``). The simulation
parameter is called ``del_M`` (with underscore), while the fitting output uses
``delM`` (without underscore) for the derived grey offset :math:`D_s - \mu`.
Time delays are encoded by the differences between ``peak_mjds`` entries.

Flat vs grid mode
~~~~~~~~~~~~~~~~~~

When ``len(mjds) != len(bands)`` (grid mode, as above), each MJD is observed
in every band. Observations in the returned array are stored in band-major
order: the first ``len(mjds)`` rows are ``bands[0]``, the next ``len(mjds)``
rows are ``bands[1]``, and so on.

When ``len(mjds) == len(bands)`` (flat mode), each observation has its own
band — ``mjds[i]`` is taken in ``bands[i]``.

Arguments
----------

The main arguments to ``simulate_light_curve_ml`` are described below. For the
full API, see :ref:`api`.

- ``mjds`` — observer-frame observation MJDs (1D array). See *Flat vs grid
  mode* above for how this interacts with ``bands``.
- ``N`` — number of objects to simulate (int).
- ``bands`` — list of filter names (must be present in the loaded filter set;
  see :ref:`filters`).
- ``peak_mjds`` — observer-frame peak MJD per image, shape ``(N, n_images)``.
  Time delays between images are encoded by the differences between these
  values.
- ``z`` — source redshift. Scalar or array of length ``N``.
- ``yerr`` — photometric uncertainty. Can be a scalar (same for all
  observations), a 1D array (per-observation), or 0 for noiseless
  simulations. Default: 0.
- ``err_type`` — ``'mag'`` (default) or ``'flux'``. If ``'mag'`` and
  generating flux, magnitude errors are propagated through the
  magnitude-to-flux conversion.
- ``mu`` — distance modulus per image. Shape should be ``(N, n_images)`` for
  multi-image simulations, or a scalar for single-image. Pass ``'z'`` to
  compute from the fiducial cosmology. Default: 0.
- ``ebv_mw`` — Milky Way :math:`E(B-V)`. Default: 0.
- ``RV`` — host :math:`R_V`. Default uses the model value (fixed or sampled
  from the population distribution).
- ``del_M`` — grey offset. Default sampled from the model prior.
- ``AV`` — host extinction. Default sampled from the model prior.
- ``theta`` — light-curve shape. Default sampled from the model prior.
- ``eps`` — residual colour variation. Default sampled from the model prior.
  Pass ``0`` to disable.
- ``mag`` — if ``True`` (default), returns magnitudes. If ``False``, returns
  fluxes.
- ``zerr`` — redshift error written to the ECSV output file. Default: ``1e-4``.
- ``save_to`` — if given, write an ECSV file of the simulated light curve(s)
  with truth parameters in the YAML header. For ``N=1`` writes to the given
  path; for ``N>1`` writes one file per SN with an index suffix. Default:
  ``None``.

Simulating in magnitudes vs flux
----------------------------------

By default, ``simulate_light_curve_ml`` returns magnitudes. To get flux
instead, just set ``mag=False``:

.. code-block:: python

    data, yerr, params = model.simulate_light_curve_ml(
        mjds=mjds, N=1, bands=bands,
        peak_mjds=peak_mjds, z=1.5,
        mu=np.array([[44.0, 43.5, 43.0]]),
        mag=False,
    )

Fixed vs sampled parameters
-----------------------------

By default, the intrinsic SN parameters (``theta``, ``AV``, ``del_M``,
``eps``) are drawn from the model priors. You can fix any of these to
specific values:

.. code-block:: python

    data, yerr, params = model.simulate_light_curve_ml(
        mjds=mjds, N=1, bands=bands,
        peak_mjds=peak_mjds, z=1.5,
        mu=np.array([[44.0, 43.5, 43.0]]),
        theta=0.5,     # fix theta
        AV=0.1,        # fix AV
        del_M=0.0,     # fix grey offset
        eps=0,          # disable epsilon
    )
