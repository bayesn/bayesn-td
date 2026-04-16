.. _models:

Built-in models
================

bayesn-td ships with several pre-trained BayeSN models. These are the same
models available in the main BayeSN package, and define the spectral energy
distribution surfaces, dust parameters, and intrinsic scatter that underpin
the light-curve model.

Summary
--------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 15

   * - Model
     - Wavelength range
     - Phase range
     - :math:`R_V` type
   * - ``G26x_model``
     - 2800--18500 Å
     - −10 to +85 days
     - population
   * - ``W22x_model``
     - 2750--18500 Å
     - −10 to +50 days
     - fixed
   * - ``W22_model``
     - 3000--18500 Å
     - −10 to +40 days
     - fixed
   * - ``T21_model``
     - 3500--9500 Å
     - −10 to +40 days
     - fixed
   * - ``M20_model``
     - 3000--18500 Å
     - −10 to +40 days
     - fixed

Available models
-----------------

G26x_model
~~~~~~~~~~~

The Grayling et al. (2026, *MNRAS*, 548) extended model, covering
2800--18500 Å and −10 to +85 rest-frame days. This is the default model.

W22x_model
~~~~~~~~~~~~

A variant of the W22 model, covering 2750--18500 Å and −10 to +50 rest-frame
days.

W22_model
~~~~~~~~~~

The Ward et al. (2023, *ApJ*, 956, 111) model, trained on combined Foundation
and Avelino datasets with *BgVrizYJH* photometry spanning 3000--18500 Å and
−10 to +40 rest-frame days.

T21_model
~~~~~~~~~~

The Thorp et al. (2021, *MNRAS*, 508, 4310) model, trained on 157 SNe Ia
from the Foundation survey using *griz* photometry, covering 3500--9500 Å and
−10 to +40 rest-frame days.

M20_model
~~~~~~~~~~

The Mandel et al. (2022, *MNRAS*, 510, 3939) model, trained on 86 SNe Ia
with *BVRIYJH* photometry spanning 3000--18500 Å and −10 to +40 rest-frame
days.

Choosing a model
-----------------

The model's wavelength range must cover the rest-frame wavelengths probed by
your filters at the source redshift.

Using a custom model
---------------------

You can use any BayeSN model by pointing ``load_model`` to a ``BAYESN.YAML``
file:

.. code-block:: python

    model = SEDmodel(load_model='/path/to/my_model/BAYESN.YAML')

The YAML file should contain the model parameters (``W0``, ``W1``,
``L_SIGMA_EPSILON``, ``M0``, ``SIGMA0``, ``TAUA``, ``L_KNOTS``,
``TAU_KNOTS``, and either ``RV`` for a fixed-:math:`R_V` model or ``MUR``
and ``SIGMAR`` for a population-:math:`R_V` model).
