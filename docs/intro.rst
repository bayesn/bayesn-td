.. _intro:

Overview
=========

What bayesn-td does
--------------------

bayesn-td extends the `BayeSN <https://github.com/bayesn/bayesn>`_ optical-NIR
SED model to strongly-lensed Type Ia supernovae. It jointly fits the
light curves of all images of a lensed SN Ia, sharing the intrinsic SN
parameters across images and inferring per-image time of maximum, magnification,
and microlensing. bayesn-td provides a joint posterior over time delays and
magnifications between images along with SN parameters.

Relationship to BayeSN
-----------------------

bayesn-td uses the same pre-trained SED surfaces, Hsiao template, dust law,
and filter infrastructure as BayeSN. The main extension is handling :math:`N`
lensed images of the same SN in a single fit: intrinsic parameters are shared
while per-image :math:`t_{\max}^{(i)}`, magnification, and (optionally) a
microlensing Gaussian process are fit independently.

Any pre-trained BayeSN model can be used; see :ref:`models`.

In addition to fitting, bayesn-td includes a simple simulator for
multi-image lensed SN Ia light curves with configurable time delays and
magnifications. It is intended for quick end-to-end tests of the fitting
pipeline rather than realistic survey simulations. See :ref:`simulation`.

The probabilistic model
------------------------

The intrinsic SN parameters — light-curve shape :math:`\theta`, host-galaxy
dust extinction :math:`A_V`, residual colour variation :math:`\epsilon`, and
dust law slope :math:`R_V` — are shared across all images. Each image has
its own time of peak brightness :math:`t_{\max}^{(i)}` and effective distance
modulus :math:`D_s^{(i)}`, which absorbs the absolute magnification from
lensing. When enabled, each image also has its own microlensing GP
hyperparameters. See Grayling et al. (2026) for full details.

Inference is performed using the No-U-Turn Sampler (NUTS) via numpyro.

.. _time_conventions:

Time conventions
-----------------

Observation times (``time_col``) should be provided in observer-frame MJD,
along with an estimated peak MJD (``peak_mjds``) for each image. These
estimates don't need to be precise — the model fits for a rest-frame
correction ``tmax`` around each initial estimate, with a uniform prior over
:math:`\pm 10` rest-frame days.

The observer-frame peak MJD for each image is then computed as
:math:`\mathrm{peak\_mjd}^{(i)} = \mathrm{peak\_mjds}^{(i)} + t_{\max}^{(i)} \times (1 + z)`,
and the time delays ``delta_t`` are defined as
:math:`\Delta t_{0i} = \mathrm{peak\_mjd}^{(0)} - \mathrm{peak\_mjd}^{(i)}`
in observer-frame days. Negative values mean image :math:`i` arrives
later than image 0 (i.e. has a larger peak MJD). Alternatively, if your times
are already rest-frame phases relative to each image's peak, you can set
``time_format='phase'``.

GPU acceleration
-----------------

bayesn-td is built on numpyro and JAX, so all model evaluations and MCMC
sampling can run on GPU. See :ref:`installation` for details on setting up
JAX with GPU support.
