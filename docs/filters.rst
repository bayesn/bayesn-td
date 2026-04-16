.. _filters:

Defining filters
===================

bayesn-td includes a selection of filters and standards for convenience.
Alternatively, you can use your own through a simple YAML file, allowing
new or updated filters to be easily implemented independently of updates
to the package.

Built-in filters
-----------------

bayesn-td includes the following filters, grouped by instrument, survey, or
system. The name shown is the string to use when referring to the filter
within bayesn-td (e.g. ``g_PS1`` for Pan-STARRS 1 *g*-band). If the names in
your data files do not match these, you can provide a ``band_map`` to match
them up, as discussed in :ref:`fitting`.

- Standard

    - Stritzinger *UBVRI* filters

        - Names: ``U``, ``B``, ``V``, ``R``, ``I``
        - Source: Stritzinger+11, 2005PASP..117..810S

    - 2MASS Peters Automated Infrared Imaging Telescope *JHK* filters

        - Names: ``J``, ``H``, ``K``
        - Source: Cohen+03, 2003AJ....126.1090C

    - Persson *YJHK* filters

        - Names: ``Y_P``, ``J_P``, ``H_P``, ``K_P``
        - Source: 1998AJ....116.2475P

- CSP

    - Carnegie Supernova Project *BVgri* Swope filters

        - Names: ``B_CSP``, ``V_CSP``, ``V_CSP_3009``, ``V_CSP_3014``, ``g_CSP``, ``r_CSP``, ``i_CSP``
        - Source: Krisciunas+17, 2017AJ....154..211K, https://csp.obs.carnegiescience.edu/data/filters

    - Carnegie Supernova Project II *BVgri* Swope filters

        - Names: ``B_CSP2``, ``V_CSP2``, ``g_CSP2``, ``r_CSP2``, ``i_CSP2``
        - Source: https://csp.obs.carnegiescience.edu/data/filters

    - Swope RetroCam *YJH* filters

        - Names: ``Y_RC``, ``J_RC1``, ``J_RC2``, ``H_RC``
        - Source: Krisciunas+17, 2017AJ....154..211K, https://csp.obs.carnegiescience.edu/data/filters

    - Dupont WIRC *YJH* filters

        - Names: ``Y_WIRC``, ``J_WIRC``, ``H_WIRC``
        - Source: Krisciunas+17, 2017AJ....154..211K, https://csp.obs.carnegiescience.edu/data/filters

    - Dupont RetroCam *YJH* filters

        - Names: ``Y_RCDP``, ``J_RCDP``, ``H_RCDP``
        - Source: https://csp.obs.carnegiescience.edu/data/filters

- DECam

    - Dark Energy Camera at Cerro Tololo Inter-American Observatory *griz* filters

        - Names: ``g_DES``, ``r_DES``, ``i_DES``, ``z_DES``
        - Source: https://noirlab.edu/science/programs/ctio/filters/Dark-Energy-Camera

- HST

    - Hubble Space Telescope WFC3 IR/UVIS2 filters

        - Names: ``F105W``, ``F125W``, ``F140W``, ``F160W``, ``F225W``, ``F275W``, ``F300X``, ``F336W``, ``F390W``, ``F438W``, ``F475W``, ``F555W``, ``F625W``, ``F814W``
        - Source: https://www.stsci.edu/hst/instrumentation/wfc3/performance/throughputs

- JWST

    - NIRCam broad-band filters

        - Names: ``F090W``, ``F115W``, ``F150W``, ``F200W``, ``F277W``, ``F356W``, ``F444W``
        - Source: STScI

- LSST

    - Legacy Survey of Space and Time at Vera Rubin Observatory *ugrizy* filters

        - Names: ``u_LSST``, ``g_LSST``, ``r_LSST``, ``i_LSST``, ``z_LSST``, ``y_LSST``
        - Source: https://github.com/lsst/throughputs

- PS1

    - Pan-STARRS 1 *grizy* filters

        - Names: ``g_PS1``, ``r_PS1``, ``i_PS1``, ``z_PS1``, ``y_PS1``
        - Source: Tonry+12, 2012ApJ...750...99T

- Roman

    - Wide Field Instrument (WFI) filters

        - Names: ``roman::Z``, ``roman::Y``, ``roman::J``, ``roman::H``, ``roman::F``, ``roman::R``

- SWIFT UVOT

    - UVOT *UBV* and UV filters

        - Names: ``U_SWIFT``, ``B_SWIFT``, ``V_SWIFT``, ``UVW1``, ``UVW2``, ``UVM2``
        - Source: Poole+08, 2008MNRAS.383..627P

- USNO

    - United States Naval Observatory 40-inch telescope *u'g'r'i'z'* filters

        - Names: ``u_prime``, ``g_prime``, ``r_prime``, ``i_prime``, ``z_prime``
        - Source: Fukugita+96, 1996AJ....111.1748F; Smith+02, 2002AJ....123.2121S

- ZTF

    - Zwicky Transient Facility *gri* filters

        - Names: ``p48g``, ``p48r``, ``p48i``
        - Source: Bellm+19, 2019PASP..131a8002B

- ANDICAM

    - ANDICAM at Cerro Tololo Inter-American Observatory *YJHK* filters

        - Names: ``Y_AND``, ``J_AND``, ``H_AND``, ``K_AND``

For a complete listing of all available filters, inspect the bundled
``filters.yaml`` file in the bayesn-td package directory.

Specifying custom filters
---------------------------

You can add custom filters beyond those built in by providing a filter YAML
file via the ``filter_yaml`` argument when initialising the model, or the
``--filters`` command-line argument. Custom filters are included alongside
the built-in ones, so you can mix and match. If you give a custom filter the
same name as a built-in one, your custom filter will take precedence.

The filter YAML file should have the following structure:

.. code-block:: yaml

    standards_root: /path/to/standards
    standards:
      vega:
        path: vega_standard.fits
      bd17:
        path: bd17_standard.fits
    filters_root: /path/to/filters
    filters:
      my_band_1:
        magsys: ab
        magzero: 0
        path: my_band_1_response.dat
        lam_unit: nm
      my_band_2:
        magsys: vega
        magzero: 0
        path: my_band_2_response.dat

- ``standards_root`` — root directory for standard spectra (optional; paths
  can be absolute).
- ``standards`` — defines reference spectra for magnitude systems. Each entry
  needs a ``path`` to a FITS or text file with WAVELENGTH and FLUX columns.
- ``filters_root`` — root directory for filter response files (optional).
- ``filters`` — defines the filters. Each entry requires:

  - ``magsys`` — magnitude system (``ab``, ``vega``, ``bd17``, or a custom
    standard name).
  - ``magzero`` — magnitude offset.
  - ``path`` — path to the filter response file.
  - ``lam_unit`` — wavelength unit in the response file. Default is Angstroms;
    set to ``nm`` or ``micron`` if needed.

Automatic filter dropping
--------------------------

The wavelength range covered by the model depends on which pre-trained model
you use (see :ref:`models`). Filters are automatically dropped for individual
SNe when they fall outside the rest-frame wavelength range at the source
redshift. The cut-off wavelengths for each filter are defined as the
wavelength where the filter response first drops below 1 per cent of its
maximum value.
