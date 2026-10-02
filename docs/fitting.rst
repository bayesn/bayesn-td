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
        outputdir='results/my_fit',
    )

For a plain text file without ECSV metadata, pass the metadata explicitly:

.. code-block:: python

    model = SEDmodel(num_devices=4)
    samples = model.fit_lensed_sn(
        photometry='path/to/photometry.txt',
        peak_mjds=[60000.0, 60050.0, 60080.0],
        z=1.5,
        ebv_mw=0.05,
        filt_map={'f150w': 'F150W', 'f200w': 'F200W'},
        error_floor={'F150W': 0.06},
        num_samples=500,
        num_warmup=500,
        num_chains=4,
        outputdir='results/my_fit',
    )

``fit_lensed_sn`` passes the MCMC options (see `MCMC configuration`_) to
``fit`` and everything else to ``process_dataset``. The two steps can also be
run separately:

.. code-block:: python

    model = SEDmodel(num_devices=4)
    model.process_dataset('path/to/photometry.txt', peak_mjds=[...], z=1.5, ebv_mw=0.05)
    samples = model.fit(num_samples=500, outputdir='results/my_fit')

Each ``SEDmodel`` holds one dataset, so create a new one for each dataset you
load.

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
standard names are ``mjd``/``time`` (or ``phase``) for time, ``filter``/``band`` for bands,
``image``/``img`` for images, and ``mag``/``magerr`` or ``flux``/``fluxerr``
for photometry. You can override any column name explicitly (``time_col``,
``band_col``, ``image_col``, ``mag_col``, ``magerr_col``, ``flux_col``,
``fluxerr_col``) if your file uses different names. The delimiter is detected
from the file content, so whitespace-delimited files work whatever their
extension.

Observations with a missing (NaN) or infinite magnitude, flux or error are
removed. Fluxes are expected on the
FLUXCAL system (zero point 27.5). If the file has a
``zp`` column, fluxes are rescaled from that zero point. If the data has both
magnitudes and fluxes, magnitudes are used.

If the band names in your data don't match the BayeSN filter names, use
the ``filt_map`` argument (``map`` in a YAML input file) — for example,
``filt_map={'f150w': 'F150W'}``. See :ref:`filters` for the full list of
built-in filter names. A band that is not a BayeSN filter after mapping raises
an error; bands you don't want to fit can be left out with ``drop_bands``,
named as in the data. You can also specify a per-band error floor in magnitudes via
``error_floor``, keyed by BayeSN filter name, and a PSF uncertainty via
``sigma_psf``; both are added in quadrature to the magnitude errors, and apply
to magnitude photometry only.

Images are identified by the unique values of the image column, sorted in
ascending order. ``peak_mjds`` is a list of estimated peak MJDs
(observer-frame) in that same sorted order. These don't need to be precise —
the model fits a rest-frame correction ``tmax`` with a uniform prior over
:math:`\pm 10` rest-frame days around them. See :ref:`time_conventions` for
full details. If the times are rest-frame phases relative to each image's
peak, pass ``time_format='phase'``; ``peak_mjds`` then only places the fitted
peaks on the observer clock, and without it the fitted peaks and time delays
are relative to each image's phase zero.

If the photometry file is ECSV with ``z``, ``ebv_mw``, and ``peak_mjds``
in its YAML header (as produced by ``simulate_light_curve_ml`` with
``save_to``), those arguments can be omitted.

Fitting many SNe at once
-------------------------

``process_dataset`` and ``fit_lensed_sn`` accept many SNe in one call. The
``photometry`` argument can be:

- a single file;
- a directory, in which case every file in it is read, in sorted order
  (subdirectories are ignored);
- a glob pattern, e.g. ``'sims/sim_*.ecsv'``;
- a DataFrame, named ``SN``;
- a list of files, directories and glob patterns.

Each file holds one SN, named after the file, except LSST lensed-SN catalogues
(see `Supported formats`_). Names must be unique. Each SN is fitted
independently, and SNe with the same number of images are fitted together in
one vectorised run.

.. code-block:: python

    model = SEDmodel(num_devices=4)
    model.process_dataset('sims/sim_*.ecsv')
    samples = model.fit(outputdir='results/sims')

Per-SN values (``z``, ``ebv_mw``, ``peak_mjds``) are taken from, in order of
precedence:

1. the arguments to ``process_dataset``, which apply to every SN;
2. a ``metadata`` table (a file or DataFrame) with an ``SNID`` column and any of
   the columns ``z``, ``z_cmb``, ``ebv_mw`` and ``peak_mjd_<image label>``
   (e.g. ``peak_mjd_A``);
3. the photometry file itself (ECSV or FITS header).

Any other columns of the ``metadata`` table, such as true parameter values for
simulations, are carried through to the output. The table may list more SNe
than are being fitted, but an error is raised if none of its ``SNID`` values match.

.. code-block:: python

    model = SEDmodel(num_devices=4)
    model.process_dataset(
        'sims/',
        z=1.783, ebv_mw=0.0157,               # shared by every SN
        peak_mjds=[59908.0, 60038.0, 59983.0],
        metadata='sims_truths.csv',           # e.g. true parameters to keep alongside
    )

Supported formats
~~~~~~~~~~~~~~~~~~

The format is detected from the file content, or can be given with
``format=``:

- ``'table'`` — text (CSV, whitespace-delimited or ECSV) or a DataFrame, as
  described above.
- ``'fits'`` — FITS files. A single table holding all images is read with
  ``z`` and ``ebv_mw`` taken from the ``REDSHIFT`` and ``EBV_MW`` header
  keys. A file with one table per image (SNANA format) is
  read with ``z``, ``z_cmb``, ``ebv_mw`` and each image's peak taken from the
  observed header keys ``ZHEL``, ``ZFIN``, ``MWEBV`` and ``PEAKMJD``; pass
  ``true_values=True`` to use the simulated true values ``SZHEL``, ``SZCMB``,
  ``SMWEBV`` and ``SPEAKMJD`` instead.
- ``'lsst_lensed_pickle'`` — a pickled catalogue of simulated LSST lensed SNe,
  one row per system, with photometry in ``obs_times``, ``obs_bands``,
  ``obs_mag_micro`` and ``mag_micro_error``. Each row is one SN, named by its
  row index. The catalogue has no Milky Way E(B-V), so pass ``ebv_mw``. To fit
  a subset, pass the selected rows as a DataFrame with
  ``format='lsst_lensed_pickle'``.

The readers are also available directly as ``bayesn_td.io.read_photometry``,
which returns one dictionary per SN.

Observations outside the model's phase range (rest-frame, relative to the
estimated peak) are left out, as are bands outside the model's wavelength
coverage at an SN's redshift. An SN with an image left with no observations is
dropped; dropped SNe and the reasons are listed in ``model.dropped``.

The loaded data
~~~~~~~~~~~~~~~~

``process_dataset`` stores the prepared data in ``model.data``, a dictionary
keyed by number of images (e.g. ``model.data[2]`` holds the arrays for every
double). ``model.sn_list`` is a table with one row per SN to be fitted, giving its
name, number of images, redshifts, E(B-V), :math:`\hat\mu`, image labels
(``image_<i>``), estimated peaks (``peak_mjd_<i>``) and any extra metadata
columns. Its rows are in the same order as the SN axis of the fit output, and
it is written to ``sn_list.txt`` in the output directory.

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
- ``outputdir`` — output directory (default ``results``).

Model options
--------------

- ``include_eps`` — whether to include the residual colour variation
  :math:`\epsilon` in the model (default ``True``).
- ``include_ml`` — whether to include the microlensing GP
  (default ``True``). When fitting many SNe at once, light curves are padded
  to the longest with the same number of images. The padding does not change
  each SN's posterior, but it gives the GP extra latent values, so the MCMC
  draws for an SN depend on the padded length.
- ``load_model`` — which pre-trained BayeSN model to use. See :ref:`models`.

The ``SEDmodel`` constructor also accepts:

- ``num_devices`` — number of CPU cores to use for parallel chains
  (default 4). This is passed to ``numpyro.set_host_device_count()``.
- ``fiducial_cosmology`` — dictionary of cosmological parameters passed to
  ``astropy.cosmology.FlatLambdaCDM``. Default:
  ``{'H0': 73.24, 'Om0': 0.28}``.

Working with the output
------------------------

The ``fit`` and ``fit_lensed_sn`` methods return a dictionary of MCMC samples.
SNe are on the last axis, in the order of ``model.sn_list``. The main
keys are:

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

When SNe with different numbers of images are fitted together, the
``images`` axis has the size of the largest, and entries for images an SN
does not have are ``NaN``.

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
