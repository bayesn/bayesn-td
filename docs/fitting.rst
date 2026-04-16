.. _fitting:

Fitting lensed SNe within Python
==================================

For using a YAML input file and the command line instead, see :ref:`running`.

Fitting with ``fit_lensed_sn``
--------------------------------

``fit_lensed_sn`` loads the photometry and runs MCMC in a single call.
Column names are auto-detected from the file header, and metadata (``z``,
``ebv_mw``, ``peak_mjds``) can be read from ECSV headers. For an ECSV file
produced by the simulator:

.. code-block:: python

    from bayesn_td import SEDmodel

    model = SEDmodel(num_devices=4)
    samples = model.fit_lensed_sn(
        photometry='examples/sim_lensed_sn/photometry.ecsv',
        output='results/my_fit',
    )

For a plain text file without ECSV metadata, pass the metadata explicitly:

.. code-block:: python

    samples = model.fit_lensed_sn(
        photometry='path/to/photometry.txt',
        peak_mjds=[60000.0, 60050.0, 60080.0],
        z=1.5,
        ebv_mw=0.05,
        band_map={'f150w': 'F150W', 'f200w': 'F200W'},
        error_floor={'F150W': 0.06},
        num_samples=500,
        num_warmup=500,
        num_chains=4,
        output='results/my_fit',
    )

Preparing your data
--------------------

bayesn-td reads photometry from a text file (whitespace-delimited, CSV, or
ECSV). The file should contain columns for the observation time, filter/band,
magnitude or flux, uncertainty, and an image identifier. For example:

.. code-block:: text

    mjd filter mag magerr image
    60000.18 F150W 26.50 0.03 A
    60000.25 F200W 25.10 0.05 A
    60050.30 F150W 26.73 0.04 B
    ...

Column names are auto-detected from the header (case-insensitive). The
standard names are ``mjd``/``time`` for time, ``filter``/``band`` for bands,
``image``/``img`` for images, and ``mag``/``magerr`` or ``flux``/``fluxerr``
for photometry. You can override any column name explicitly if your file uses
different names. If the file has no header, columns are assumed to be in the
order: ``mjd, filter, mag, magerr, image``.

If the band names in your data don't match the BayeSN filter names, use
the ``band_map`` argument — for example,
``band_map={'f150w': 'F150W'}``. See :ref:`filters` for the full list of
built-in filter names. You can also specify a per-band error floor in
magnitudes via ``error_floor``, which is added in quadrature to the
magnitude errors.

Images are identified by the unique values of the image column, sorted in
ascending order. ``peak_mjds`` is a list of estimated peak MJDs
(observer-frame) in that same sorted order. These don't need to be precise —
the model fits a rest-frame correction ``tmax`` with a uniform prior over
:math:`\pm 10` rest-frame days around them. See :ref:`time_conventions` for
full details.

If the photometry file is ECSV with ``z``, ``ebv_mw``, and ``peak_mjds``
in its YAML header (as produced by ``simulate_light_curve_ml`` with
``save_to``), those arguments can be omitted.

MCMC configuration
-------------------

The following parameters control the MCMC sampling:

- ``num_samples`` — number of posterior samples per chain (default 500).
- ``num_warmup`` — number of warmup (burn-in) steps per chain (default 500).
- ``num_chains`` — number of independent chains (default 4).
- ``chain_method`` — how to distribute chains. ``'parallel'`` (default)
  runs chains in parallel across CPU cores or GPU devices, ``'sequential'``
  runs one chain at a time, and ``'vectorized'`` vectorises chains on a
  single device.
- ``init_strategy`` — initialisation strategy for the NUTS sampler.
  ``'median'`` (default) initialises parameters at the prior median;
  ``'sample'`` draws a random sample from the prior.

Model options
--------------

- ``include_eps`` — whether to include the residual colour variation
  :math:`\epsilon` in the model (default ``True``).
- ``include_ml`` — whether to include the microlensing GP
  (default ``True``).
- ``load_model`` — which pre-trained BayeSN model to use. See :ref:`models`.

The ``SEDmodel`` constructor also accepts:

- ``num_devices`` — number of CPU cores to use for parallel chains
  (default 4). This is passed to ``numpyro.set_host_device_count()``.
- ``fiducial_cosmology`` — dictionary of cosmological parameters passed to
  ``astropy.cosmology.FlatLambdaCDM``. Default:
  ``{'H0': 73.24, 'Om0': 0.28}``. This is used to compute the fiducial
  distance modulus :math:`\hat\mu` from the source redshift.

Working with the output
------------------------

The ``fit`` and ``fit_lensed_sn`` methods return a dictionary of MCMC samples.
The main keys are:

- ``theta`` — light-curve shape parameter, shape ``(chains, samples, SNe)``
- ``AV`` — host extinction, shape ``(chains, samples, SNe)``
- ``tmax`` — rest-frame time-of-maximum correction per image,
  shape ``(chains, samples, images, SNe)``
- ``Ds`` — effective distance modulus per image,
  shape ``(chains, samples, images, SNe)``
- ``delta_t`` — time delays between image 0 and each subsequent image,
  shape ``(chains, samples, N_images-1, SNe)``.
  :math:`\Delta t_{0i} = \mathrm{peak\_mjd}^{(0)} - \mathrm{peak\_mjd}^{(i)}`
  in **observer-frame days**. Negative values mean image :math:`i` arrives
  later than image 0 (i.e. has a larger peak MJD).
- ``mu`` — distance modulus per image, drawn in post-processing from a
  Normal whose mean is the precision-weighted average of the sampled
  :math:`D_s` and the fiducial distance modulus :math:`\hat\mu`, with
  variance set by the BayeSN intrinsic scatter :math:`\sigma_0`.
  Shape: ``(chains, samples, images, SNe)``
- ``peak_mjd`` — observer-frame peak MJD per image,
  shape ``(chains, samples, images, SNe)``

To extract the time-delay posterior between the first and second images:

.. code-block:: python

    import numpy as np

    delta_t = samples['delta_t']
    dt_01 = delta_t[:, :, 0, 0].flatten()
    print(f'Delta_t (image 0 - image 1): '
          f'{np.mean(dt_01):.2f} +/- {np.std(dt_01):.2f} days')

If ``include_ml`` was set to ``True``, the samples will also contain the
microlensing GP hyperparameters (``A``, ``tscale``, ``tau_ml``, ``p``,
``eta``) and the realised GP function values (``beta_t``).

The results are also saved to disk — see :ref:`output` for details on the
output files.
