"""
BayeSN SED Model. Defines a class which allows you to fit or simulate from the
BayeSN Optical+NIR SED model for strongly-lensed Type Ia supernovae.
"""

import inspect
import os
import re
import timeit
import pickle
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import h5py
import pandas as pd
import jax
from jax import device_put
import jax.numpy as jnp
from jax.random import PRNGKey
from jax.scipy.stats import norm
from jax.scipy.special import ndtri
from astropy.cosmology import FlatLambdaCDM
import astropy.constants as const
from astropy.io import fits
from ruamel.yaml import YAML
import scipy
import extinction
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median, init_to_sample
import arviz

from .spline_utils import invKD_irr, spline_coeffs_irr
from .io import ZPT, read_photometry, read_text_table, write_ecsv

yaml = YAML(typ='safe')
yaml.default_flow_style = False

jax.config.update('jax_enable_x64', True)

# fluxerr given to padded (masked) observations: with flux and model both 0 there, their Normal log-density is
# exactly 0
PAD_FLUXERR = 1 / np.sqrt(2 * np.pi)

# Width (mag) of the Normal prior on each image's distance modulus Ds, centred on the fiducial value
DS_PRIOR_WIDTH = 5

# Random seed for the MCMC and for post-processing draws
SEED = 123

# Keys of the padded photometry arrays passed to td_model, each with shape (n_obs, n_images, n_sn) except muhat,
# which has shape (n_images, n_sn): rest-frame phase relative to the estimated peak, FLUXCAL flux and error, band
# index into the bands in use, mask (False for padding) and fiducial distance modulus
OBS_KEYS = ('phase', 'flux', 'fluxerr', 'band', 'mask', 'muhat')


class SEDmodel:
    """
    BayeSN-SED Model

    Class which imports a BayeSN model, and allows one to fit or simulate
    Type Ia supernovae based on this model.

    Parameters
    ----------
    num_devices : int, optional
        Number of CPU cores for numpyro to use. Defaults to 4.
    load_model : str, optional
        A pre-defined BayeSN model name or a path to a BAYESN.YAML file.
        Built-in models: 'G26x_model', 'T21_model', 'M20_model', 'W22_model', etc.
        Default is 'G26x_model'.
    filter_yaml : str, optional
        Path to filter YAML file. Defaults to the bundled filters.yaml.
    fiducial_cosmology : dict, optional
        Kwargs ``{H0, Om0}`` for FlatLambdaCDM. Defaults to ``{H0: 73.24, Om0: 0.28}``.
    """

    def __init__(self, num_devices=4, load_model='G26x_model', filter_yaml=None,
                 fiducial_cosmology={"H0": 73.24, "Om0": 0.28}):
        import numpyro
        numpyro.set_host_device_count(num_devices)

        self.__root_dir__ = os.path.dirname(os.path.abspath(__file__))

        if filter_yaml is None:
            filter_yaml = os.path.join(self.__root_dir__, 'filters', 'filters.yaml')

        self.cosmo = FlatLambdaCDM(**fiducial_cosmology)
        self.data = None
        self.sn_list = None
        self.dropped = None
        self.hsiao_interp = None
        self.RV_MW = device_put(jnp.array(3.1))
        self.sigma_pec = device_put(jnp.array(150 / 3e5))
        self.filter_yaml = filter_yaml
        built_in_models = [d for d in next(os.walk(os.path.join(self.__root_dir__, 'model_files')))[1]
                           if os.path.exists(os.path.join(self.__root_dir__, 'model_files', d, 'BAYESN.YAML'))]

        if os.path.exists(load_model):
            with open(load_model, 'r') as file:
                params = yaml.load(file)
        elif load_model in built_in_models:
            with open(os.path.join(self.__root_dir__, 'model_files', load_model, 'BAYESN.YAML'), 'r') as file:
                params = yaml.load(file)
        else:
            raise FileNotFoundError(f'Specified model {load_model} does not exist and does not correspond to one '
                                    f'of the built-in models {built_in_models}')

        self.l_knots = jnp.array(params['L_KNOTS'])
        self.tau_knots = jnp.array(params['TAU_KNOTS'])
        self.W0 = jnp.array(params['W0'])
        self.W1 = jnp.array(params['W1'])
        self.L_Sigma = jnp.array(params['L_SIGMA_EPSILON'])
        self.M0 = jnp.array(params['M0'])
        self.sigma0 = jnp.array(params['SIGMA0'])
        self.tauA = jnp.array(params['TAUA'])
        if 'RV' in params.keys():
            self.model_type = 'fixed_RV'
            self.RV = jnp.array(params['RV'])
        elif 'MUR' in params.keys():
            self.model_type = 'pop_RV'
            self.mu_R = jnp.array(params['MUR'])
            self.sigma_R = jnp.array(params['SIGMAR'])
        if 'TRUNCRV' in params.keys():
            self.truncate_RV = True
            self.trunc_val = jnp.array(params['TRUNCRV'])
        else:
            self.truncate_RV = False

        self.used_band_inds = None
        self._setup_band_weights()

        KD_l = invKD_irr(self.l_knots)
        self.J_l_T = device_put(spline_coeffs_irr(self.model_wave, self.l_knots, KD_l))
        self.KD_t = device_put(invKD_irr(self.tau_knots))
        self._load_hsiao_template()

        self.ZPT = ZPT
        self.J_l_T = device_put(self.J_l_T)
        self.hsiao_flux = device_put(self.hsiao_flux)
        self.J_l_T_hsiao = device_put(self.J_l_T_hsiao)
        self.xk = jnp.array(
            [0.0, 1e4 / 26500., 1e4 / 12200., 1e4 / 6000., 1e4 / 5470., 1e4 / 4670., 1e4 / 4110., 1e4 / 2700.,
             1e4 / 2600.])
        KD_x = invKD_irr(self.xk)
        self.M_fitz_block = device_put(spline_coeffs_irr(1e4 / self.model_wave, self.xk, KD_x))

        self.J_t_map = jax.jit(jax.vmap(self.spline_coeffs_irr_step, in_axes=(0, None, None)))

    def _load_hsiao_template(self):
        """Loads the Hsiao template from the internal HDF5 file."""
        with h5py.File(os.path.join(self.__root_dir__, 'hsiao.h5'), 'r') as file:
            data = file['default']
            hsiao_phase = data['phase'][()].astype('float64')
            hsiao_wave = data['wave'][()].astype('float64')
            hsiao_flux = data['flux'][()].astype('float64')

        KD_l_hsiao = invKD_irr(hsiao_wave)
        self.KD_t_hsiao = device_put(invKD_irr(hsiao_phase))
        self.J_l_T_hsiao = device_put(spline_coeffs_irr(self.model_wave, hsiao_wave, KD_l_hsiao))

        self.hsiao_t = device_put(hsiao_phase)
        self.hsiao_l = device_put(hsiao_wave)
        self.hsiao_flux = device_put(hsiao_flux.T)
        self.hsiao_flux = jnp.matmul(self.J_l_T_hsiao, self.hsiao_flux)

    def _setup_band_weights(self):
        """
        Sets up the interpolation for the band weights used for photometry as well as calculating the zero points for
        each band. Based on ParSNiP from Boone+21.
        """
        self.min_wave = self.l_knots[0]
        self.max_wave = self.l_knots[-1]
        self.spectrum_bins = 300
        self.band_oversampling = 51
        self.max_redshift = 4

        model_log_wave = np.linspace(np.log10(self.min_wave),
                                     np.log10(self.max_wave),
                                     self.spectrum_bins)

        model_spacing = model_log_wave[1] - model_log_wave[0]
        band_spacing = model_spacing / self.band_oversampling
        band_max_log_wave = (
                np.log10(self.max_wave * (1 + self.max_redshift))
                + band_spacing
        )

        assert self.band_oversampling % 2 == 1
        pad = (self.band_oversampling - 1) // 2
        band_log_wave = np.arange(np.log10(self.min_wave),
                                  band_max_log_wave, band_spacing)
        band_wave = 10 ** band_log_wave

        if not os.path.exists(self.filter_yaml):
            raise FileNotFoundError(f'Specified filter yaml {self.filter_yaml} does not exist')
        with open(self.filter_yaml, 'r') as file:
            filter_dict = yaml.load(file)

        # Load standard spectra if necessary
        if 'standards' in filter_dict.keys():
            if 'standards_root' in filter_dict.keys():
                standards_root = filter_dict['standards_root']
            else:
                standards_root = ''
            for key, val in filter_dict['standards'].items():
                path = os.path.join(standards_root, val['path'])
                split_path = os.path.normpath(path).split(os.path.sep)
                root = split_path[0]
                if root[:1] == '$':
                    env = os.getenv(root[1:])
                    if env is None:
                        raise FileNotFoundError(f'The environment variable {root} was not found')
                    path = os.path.join(env, *split_path[1:])
                elif not os.path.isabs(path):
                    path = os.path.join(os.path.split(self.filter_yaml)[0], path)
                if '.fits' in path:
                    with fits.open(path) as hdu:
                        standard_df = pd.DataFrame.from_records(hdu[1].data)
                    standard_lam, standard_f = standard_df.WAVELENGTH.values, standard_df.FLUX.values
                else:
                    standard_txt = np.loadtxt(path)
                    standard_lam, standard_f = standard_txt[:, 0], standard_txt[:, 1]
                filter_dict['standards'][key]['lam'] = standard_lam
                filter_dict['standards'][key]['f_lam'] = standard_f

        def ab_standard_flam(l):
            f = (const.c.to('AA/s').value / 1e23) * (l ** -2) * 10 ** (-48.6 / 2.5) * 1e23
            return f

        # Load filters
        if 'filters_root' in filter_dict.keys():
            filters_root = filter_dict['filters_root']
        else:
            filters_root = ''

        band_weights, zps, offsets = [], [], []
        self.band_dict, self.zp_dict, self.band_lim_dict = {}, {}, {}

        # Prepare NULL band for padded data points
        self.band_dict['NULL_BAND'] = 0
        self.zp_dict['NULL_BAND'] = 10
        self.band_lim_dict['NULL_BAND'] = band_wave[0], band_wave[-1]
        band_weights.append(np.ones_like(band_wave))
        zps.append(10)
        offsets.append(0)

        band_ind = 1
        for key, val in filter_dict['filters'].items():
            path = os.path.join(filters_root, val['path'])
            split_path = os.path.normpath(path).split(os.path.sep)
            root = split_path[0]
            if root[:1] == '$':
                env = os.getenv(root[1:])
                if env is None:
                    raise FileNotFoundError(f'The environment variable {root} was not found')
                path = os.path.join(env, *split_path[1:])
            elif not os.path.isabs(path):
                path = os.path.join(os.path.split(self.filter_yaml)[0], path)
            band, magsys, offset = key, val['magsys'], val['magzero']
            try:
                R = np.loadtxt(path)
            except Exception as e:
                raise FileNotFoundError(f'Filter response not found for {key}: {path}') from e

            units = val.get("lam_unit", "AA")
            if units.lower() == "nm":
                R[:, 0] = R[:, 0] * 10
            elif units.lower() == "micron":
                R[:, 0] = R[:, 0] * 1e4

            band_low_lim = R[np.where(R[:, 1] > 0.01 * R[:, 1].max())[0][0], 0]
            band_up_lim = R[np.where(R[:, 1] > 0.01 * R[:, 1].max())[0][-1], 0]

            band_conv_transmission = scipy.interpolate.interp1d(R[:, 0], R[:, 1], kind='cubic',
                                                                fill_value=0, bounds_error=False)(band_wave)

            dlamba = jnp.diff(band_wave)
            dlamba = jnp.r_[dlamba, dlamba[-1]]

            num = band_wave * band_conv_transmission * dlamba
            denom = jnp.sum(num)
            band_weight = num / denom

            band_weights.append(band_weight)

            lam = R[:, 0]
            if magsys == 'ab':
                zp = ab_standard_flam(lam)
            else:
                standard = filter_dict['standards'][magsys]
                zp = interp1d(standard['lam'], standard['f_lam'], kind='cubic')(lam)

            int1 = simpson(lam * zp * R[:, 1], lam)
            int2 = simpson(lam * R[:, 1], lam)
            zp = 2.5 * np.log10(int1 / int2)
            self.band_dict[band] = band_ind
            self.band_lim_dict[band] = [band_low_lim, band_up_lim]
            self.zp_dict[band] = zp
            zps.append(zp)
            offsets.append(offset)
            band_ind += 1

        self.used_band_inds = np.array(list(self.band_dict.values()))
        self.zps = jnp.array(zps)
        self.offsets = jnp.array(offsets)
        self.inv_band_dict = {val: key for key, val in self.band_dict.items()}

        band_interpolate_locations = jnp.arange(
            0,
            self.spectrum_bins * self.band_oversampling,
            self.band_oversampling
        )

        self.band_interpolate_locations = device_put(band_interpolate_locations)
        self.band_interpolate_spacing = band_spacing
        self.band_interpolate_weights = jnp.array(band_weights)
        self.model_wave = 10 ** model_log_wave
        self.used_band_dict = {val: val for val in self.band_dict.values()}

    def _calculate_band_weights(self, redshifts, ebv):
        """
        Calculates the observer-frame band weights, including the effect of Milky Way extinction, for each SN.

        Parameters
        ----------
        redshifts : array-like
            Array of redshifts for each SN.
        ebv : array-like
            Array of Milky Way E(B-V) values for each SN.

        Returns
        -------
        weights : array-like
            Array containing observer-frame band weights.
        """
        locs = (
                self.band_interpolate_locations
                + jnp.log10(1 + redshifts)[:, None] / self.band_interpolate_spacing
        )

        flat_locs = locs.flatten()

        int_locs = flat_locs.astype(jnp.int32)
        remainders = flat_locs - int_locs

        filtered_weights = self.band_interpolate_weights[self.used_band_inds, ...]

        start = filtered_weights[..., int_locs]
        end = filtered_weights[..., int_locs + 1]

        flat_result = remainders * end + (1 - remainders) * start
        weights = flat_result.reshape((-1,) + locs.shape).transpose(1, 2, 0)
        sum = jnp.sum(weights, axis=1)
        weights /= sum[:, None, :]

        av = self.RV_MW * ebv
        all_lam = np.array(self.model_wave[None, :] * (1 + redshifts[:, None]))
        all_lam = all_lam.flatten(order='F')
        mw_ext = extinction.fitzpatrick99(all_lam, 1, self.RV_MW)
        mw_ext = mw_ext.reshape((weights.shape[0], weights.shape[1]), order='F')
        mw_ext = mw_ext * av[:, None]
        mw_ext = jnp.power(10, -0.4 * mw_ext)

        weights = weights * mw_ext[..., None]
        weights /= (1 + redshifts)[:, None, None]

        return weights

    def get_spectra(self, theta, Av, W0, W1, eps, Rv, J_t, hsiao_interp):
        """
        Calculates rest-frame spectra for given parameter values (multi-image).

        Parameters
        ----------
        theta, Av, W0, W1, eps, Rv, J_t, hsiao_interp : array-like
            Model parameters and interpolation matrices.

        Returns
        -------
        model_spectra : array-like
            Model spectra for all SNe at all time-steps.
        """
        num_batch = theta.shape[0]

        W = W0[None, ...] + theta[..., None, None] * W1[None, ...] + eps
        W = W[:, None, ...]

        WJt = jnp.matmul(W, J_t)
        W_grid = jnp.matmul(self.J_l_T, WJt)
        low_hsiao = self.hsiao_flux[:, hsiao_interp[0, ...].astype(int)]
        up_hsiao = self.hsiao_flux[:, hsiao_interp[1, ...].astype(int)]
        H_grid = ((1 - hsiao_interp[2, :]) * low_hsiao + hsiao_interp[2, :] * up_hsiao).transpose(3, 2, 0, 1)

        model_spectra = H_grid * 10 ** (-0.4 * W_grid)

        # Fitzpatrick99 host extinction
        f99_x0 = 4.596
        f99_gamma = 0.99
        f99_c2 = -0.824 + 4.717 / Rv
        f99_c1 = 2.030 - 3.007 * f99_c2
        f99_c3 = 3.23
        f99_c4 = 0.41
        f99_c5 = 5.9
        f99_d1 = self.xk[7] ** 2 / ((self.xk[7] ** 2 - f99_x0 ** 2) ** 2 + (f99_gamma * self.xk[7]) ** 2)
        f99_d2 = self.xk[8] ** 2 / ((self.xk[8] ** 2 - f99_x0 ** 2) ** 2 + (f99_gamma * self.xk[8]) ** 2)
        yk = jnp.zeros((num_batch, 9))
        yk = yk.at[:, 0].set(-Rv)
        yk = yk.at[:, 1].set(0.26469 * Rv / 3.1 - Rv)
        yk = yk.at[:, 2].set(0.82925 * Rv / 3.1 - Rv)
        yk = yk.at[:, 3].set(-0.422809 + 1.00270 * Rv + 2.13572e-4 * Rv ** 2 - Rv)
        yk = yk.at[:, 4].set(-5.13540e-2 + 1.00216 * Rv - 7.35778e-5 * Rv ** 2 - Rv)
        yk = yk.at[:, 5].set(0.700127 + 1.00184 * Rv - 3.32598e-5 * Rv ** 2 - Rv)
        yk = yk.at[:, 6].set(
            1.19456 + 1.01707 * Rv - 5.46959e-3 * Rv ** 2 + 7.97809e-4 * Rv ** 3 - 4.45636e-5 * Rv ** 4 - Rv)
        yk = yk.at[:, 7].set(f99_c1 + f99_c2 * self.xk[7] + f99_c3 * f99_d1)
        yk = yk.at[:, 8].set(f99_c1 + f99_c2 * self.xk[8] + f99_c3 * f99_d2)

        A = Av[..., None] * (1 + (self.M_fitz_block @ yk.T).T / Rv[..., None])
        f_A = 10 ** (-0.4 * A)
        model_spectra = model_spectra * f_A[:, None, :, None]

        return model_spectra

    def get_flux_batch(self, theta, Av, W0, W1, eps, Ds, Rv, band_indices, mask, J_t, hsiao_interp, weights, beta_t):
        """Calculates observer-frame fluxes for multi-image (lensed) data."""
        num_batch = theta.shape[0]
        num_observations = band_indices.shape[0]
        num_images = band_indices.shape[1]

        model_spectra = self.get_spectra(theta, Av, W0, W1, eps, Rv, J_t, hsiao_interp)

        batch_indices = (
            jnp.arange(num_batch)
            .repeat(num_images).repeat(num_observations)
        ).astype(int)

        obs_band_weights = (
            weights[batch_indices, :, band_indices.flatten(order='F')]
            .reshape((num_batch, num_images, num_observations, -1))
            .transpose(0, 1, 3, 2)
        )

        model_flux = jnp.sum(model_spectra * obs_band_weights, axis=2).T
        model_flux = model_flux * 10 ** (-0.4 * (self.M0 + Ds))
        zps = self.zps[band_indices]
        offsets = self.offsets[band_indices]
        zp_flux = 10 ** (zps / 2.5)
        model_flux = (model_flux / zp_flux) * 10 ** (0.4 * (self.ZPT - offsets))
        model_flux = model_flux * 10 ** (-0.4 * beta_t)  # Apply microlensing

        model_flux *= mask
        return model_flux

    def get_mag_batch(self, theta, Av, W0, W1, eps, Ds, Rv, band_indices, mask, J_t, hsiao_interp, weights, beta_t):
        """Calculates observer-frame magnitudes for multi-image (lensed) data."""
        model_flux = self.get_flux_batch(theta, Av, W0, W1, eps, Ds, Rv, band_indices, mask, J_t, hsiao_interp, weights, beta_t)
        model_flux = model_flux + (1 - mask) * 0.01
        model_mag = - 2.5 * jnp.log10(model_flux) + self.ZPT
        model_mag *= mask
        return model_mag

    @staticmethod
    def spline_coeffs_irr_step(x_now, x, invkd):
        """Vectorized version of cubic spline coefficient calculator for JAX vmap."""
        X = jnp.zeros_like(x)
        up_extrap = x_now > x[-1]
        down_extrap = x_now < x[0]
        interp = 1 - up_extrap - down_extrap

        h = x[-1] - x[-2]
        a = (x[-1] - x_now) / h
        b = 1 - a
        f = (x_now - x[-1]) * h / 6.0

        X = X.at[-2].set(X[-2] + a * up_extrap)
        X = X.at[-1].set(X[-1] + b * up_extrap)
        X = X.at[:].set(X[:] + f * invkd[-2, :] * up_extrap)

        h = x[1] - x[0]
        b = (x_now - x[0]) / h
        a = 1 - b
        f = (x_now - x[0]) * h / 6.0

        X = X.at[0].set(X[0] + a * down_extrap)
        X = X.at[1].set(X[1] + b * down_extrap)
        X = X.at[:].set(X[:] - f * invkd[1, :] * down_extrap)

        q = jnp.argmax(x_now < x) - 1
        h = x[q + 1] - x[q]
        a = (x[q + 1] - x_now) / h
        b = 1 - a
        c = ((a ** 3 - a) / 6) * h ** 2
        d = ((b ** 3 - b) / 6) * h ** 2

        X = X.at[q].set(X[q] + a * interp)
        X = X.at[q + 1].set(X[q + 1] + b * interp)
        X = X.at[:].set(X[:] + c * invkd[q, :] * interp + d * invkd[q + 1, :] * interp)

        return X

    # --- Prior sampling methods ---

    def sample_del_M(self, N):
        """Samples grey offset del_M from model prior."""
        return np.random.normal(0, self.sigma0, N)

    def sample_AV(self, N):
        """Samples AV from model prior."""
        return np.random.exponential(self.tauA, N)

    def sample_theta(self, N):
        """Samples theta from model prior."""
        return np.random.normal(0, 1, N)

    def sample_epsilon(self, N):
        """Samples epsilon from model prior."""
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]
        eps_mu = np.zeros(N_knots_sig)
        eps_tform = np.random.multivariate_normal(eps_mu, np.eye(N_knots_sig), N)
        eps_tform = eps_tform.T
        eps = np.matmul(self.L_Sigma, eps_tform)
        eps = eps.T
        eps = np.reshape(eps, (N, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
        eps_full = np.zeros((N, self.l_knots.shape[0], self.tau_knots.shape[0]))
        eps_full[:, 1:-1, :] = eps
        return eps_full

    # --- Numpyro probabilistic model ---

    def td_model(self, data, weights, include_eps=True, include_ml=True):
        """
        Unified numpyro probabilistic model for time-delay fitting of lensed SNe Ia.

        Parameters
        ----------
        data : dict
            Padded photometry arrays (keys ``OBS_KEYS``) for SNe with the same number of images, from
            ``process_dataset``.
        weights : array-like
            Band weights from _calculate_band_weights.
        include_eps : bool, optional
            Whether to include residual colour variation (epsilon). Default True.
        include_ml : bool, optional
            Whether to include microlensing GP model. Default True.
        """
        sample_size = data['flux'].shape[-1]
        n_img = data['flux'].shape[-2]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]

        if hasattr(self, 'mu_R'):
            mu_R, sigma_R = self.mu_R, self.sigma_R
            phi_alpha_R = norm.cdf((1.2 - mu_R) / sigma_R)
            pop_RV = True
        else:
            pop_RV = False

        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample('theta', dist.Normal(0, 1.0))
            Av = numpyro.sample('AV', dist.Exponential(1 / self.tauA))
            if pop_RV:
                RV_tform = numpyro.sample('RV_tform', dist.Uniform(0, 1))
                RV = numpyro.deterministic('RV', mu_R + sigma_R * ndtri(phi_alpha_R + RV_tform * (1 - phi_alpha_R)))
            else:
                RV = self.RV

            if include_eps:
                eps_mu = jnp.zeros(N_knots_sig)
                eps_tform = numpyro.sample('eps_tform', dist.MultivariateNormal(eps_mu, jnp.eye(N_knots_sig)))
                eps_tform = eps_tform.T
                eps = numpyro.deterministic('eps', jnp.matmul(self.L_Sigma, eps_tform))
                eps = eps.T
                eps = jnp.reshape(eps, (sample_size, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
                eps_full = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
                eps = eps_full.at[:, 1:-1, :].set(eps)
            else:
                eps = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))

            with numpyro.plate('img', n_img) as img_index:
                tmax = numpyro.sample('tmax', dist.Uniform(-10, 10))
                Ds = numpyro.sample('Ds', dist.Normal(data['muhat'], DS_PRIOR_WIDTH))

                if include_ml:
                    A = numpyro.sample('A', dist.HalfNormal(0.1))
                    lam = numpyro.sample('tscale', dist.Uniform(10, 150))
                    tau_ml = numpyro.sample('tau_ml', dist.Uniform(-10, 85))
                    p = numpyro.sample('p', dist.Uniform(0, 1))
                    eta = numpyro.sample('eta', dist.Uniform(1, 40))
                    beta_t_tform = numpyro.sample('beta_t_tform',
                                                  dist.MultivariateNormal(0, jnp.eye(data['phase'].shape[0])))

            t = data['phase'] - tmax[None, ...]

            if include_ml:
                tt = t[:, None, ...] - t[None, ...]
                # Gibbs kernel
                l0 = lam * (1 - p * jnp.exp(-(t - tau_ml) ** 2 / (2 * eta ** 2)))
                l, l_prime = l0[:, None, ...], l0[None, ...]
                K = (A ** 2 * jnp.sqrt((2 * l * l_prime) / (l * l + l_prime * l_prime)) *
                     jnp.exp(-tt ** 2 / (l * l + l_prime * l_prime)) +
                     1e-15 * jnp.eye(t.shape[0])[..., None, None])
                L_K = jnp.linalg.cholesky(K.transpose(2, 3, 0, 1))
                beta_t = numpyro.deterministic('beta_t', jnp.matmul(L_K, beta_t_tform[..., None]))
                beta_t = beta_t[..., 0].transpose(2, 0, 1)
            else:
                beta_t = jnp.zeros(data['flux'].shape)

            hsiao_interp = jnp.array([19 + jnp.floor(t), 19 + jnp.ceil(t), jnp.remainder(t, 1)])
            keep_shape = t.shape
            t = t.flatten(order='F')
            J_t = self.J_t_map(t, self.tau_knots, self.KD_t).reshape(
                (*keep_shape, self.tau_knots.shape[0]), order='F').transpose(2, 1, 3, 0)
            flux = self.get_flux_batch(theta, Av, self.W0, self.W1, eps, Ds, RV,
                                       data['band'], data['mask'], J_t, hsiao_interp, weights, beta_t)

            with numpyro.handlers.mask(mask=data['mask']):
                numpyro.sample('obs', dist.Normal(flux, data['fluxerr']), obs=data['flux'])

    # --- Data loading ---

    def process_dataset(self, photometry, *, metadata=None, z=None, ebv_mw=None, peak_mjds=None, filt_map=None,
                        drop_bands=None, error_floor=None, sigma_psf=0, **read_kwargs):
        """
        Load one or more lensed SNe for fitting. Each SEDmodel holds one dataset: create a new SEDmodel for each
        dataset you load.

        Per-SN values (``z``, ``ebv_mw``, ``peak_mjds``) are taken from, in order of precedence: the arguments here
        (applied to every SN), the ``metadata`` table, then the photometry source itself. Observations with
        non-finite photometry, in ``drop_bands``, outside the model's phase range or in bands outside the model's
        wavelength coverage at each SN's redshift are left out, and SNe with an image left with no observations are
        dropped.

        The prepared data are stored in:

        - ``self.data``: a dictionary keyed by number of images, e.g. ``self.data[2]`` for all doubles. Each entry is
          a dictionary of padded arrays (keys ``OBS_KEYS``) plus ``band_weights`` and ``sn_index``, the rows of
          ``self.sn_list`` it holds.
        - ``self.sn_list``: one row per SN to be fitted, in the order of the SN axis of the fit output: ``SNID``,
          ``n_images``, ``z``, ``z_cmb``, ``ebv_mw``, ``muhat``, ``image_<i>``, ``peak_mjd_<i>`` and any extra
          ``metadata`` columns.
        - ``self.dropped``: SNe that were dropped, with the reason.
        - ``self.used_band_inds``: indices into the model's band table of the bands in use, which the ``band``
          arrays index.
        - ``self.used_band_dict``: the position in ``self.used_band_inds`` of each band in use, keyed by its index
          in the model's band table.
        - ``self.zps``, ``self.offsets``: reduced to the bands in use, as in BayeSN.

        Parameters
        ----------
        photometry : str, pd.DataFrame or list
            Anything accepted by ``bayesn_td.io.read_photometry``: a file, a directory, a glob pattern, a DataFrame,
            or a list of these.
        metadata : str or pd.DataFrame, optional
            Table keyed by an ``SNID`` column, with any of the columns ``z``, ``z_cmb``, ``ebv_mw`` and
            ``peak_mjd_<image label>``. Other columns (e.g. true values) are carried through to the output.
        z : float, optional
            Source redshift, applied to every SN.
        ebv_mw : float, optional
            Milky Way E(B-V), applied to every SN.
        peak_mjds : list of float, optional
            Estimated observer-frame peak of each image, ordered by sorted image label, applied to every SN (all must
            have this many images). For ``time_format='phase'`` data, the peaks only place the fitted peaks and time
            delays on the observer clock, and default to 0.
        filt_map : dict, optional
            Mapping from band names in the data to BayeSN filter names (``map`` in a YAML input file). Bands not in
            the filter set after mapping raise an error.
        drop_bands : list of str, optional
            Bands to leave out, named as in the data.
        error_floor : dict, optional
            Per-band error floor in magnitudes, keyed by BayeSN filter name (after ``filt_map``), added in quadrature.
            Magnitude photometry only.
        sigma_psf : float, optional
            PSF uncertainty in magnitudes, added in quadrature. Magnitude photometry only. Default 0.
        **read_kwargs
            Passed to ``bayesn_td.io.read_photometry`` (e.g. ``format``, ``time_format``, column names,
            ``true_values``).
        """
        if self.data is not None:
            raise ValueError('This SEDmodel already holds a dataset; create a new SEDmodel for each dataset')
        sne = read_photometry(photometry, **read_kwargs)
        counts = pd.Series([sn['name'] for sn in sne]).value_counts()
        if (counts > 1).any():
            raise ValueError(f'SN names must be unique; duplicated: {list(counts.index[counts > 1])}')

        metadata = self._read_metadata(metadata)
        if metadata is not None and not metadata.index.isin([sn['name'] for sn in sne]).any():
            raise ValueError(f'No SN names match the metadata SNID column (e.g. {metadata.index[0]!r} vs '
                             f'{sne[0]["name"]!r})')
        values_for_all_sne = {'z': z, 'ebv_mw': ebv_mw, 'peak_mjds': peak_mjds}

        rows, curves, dropped, n_nonfinite = [], [], [], []
        for sn in sne:
            info = self._resolve_metadata(sn, metadata, values_for_all_sne)
            lc, reason, n_removed = self._prepare_sn(sn, info, filt_map or {}, set(drop_bands or ()),
                                                     error_floor or {}, sigma_psf)
            if n_removed:
                n_nonfinite.append(n_removed)
            if reason is not None:
                dropped.append((sn['name'], reason))
                continue
            rows.append(info)
            curves.append(lc)
        if n_nonfinite:
            print(f'process_dataset: removed {sum(n_nonfinite)} observations with non-finite photometry from '
                  f'{len(n_nonfinite)} SNe')
        if dropped:
            print(f'process_dataset: dropped {len(dropped)} of {len(sne)} SNe; see self.dropped')
        if not rows:
            raise ValueError('No SNe left to fit')

        self.sn_list = pd.DataFrame(rows)
        self.dropped = pd.DataFrame(dropped, columns=['SNID', 'reason'])
        bands = sorted({b for lc in curves for b in lc['band']}, key=self.band_dict.get)
        self.used_band_inds = np.array([self.band_dict['NULL_BAND']] + [self.band_dict[b] for b in bands])
        self.used_band_dict = {int(band_ind): i for i, band_ind in enumerate(self.used_band_inds)}
        self.zps = self.zps[self.used_band_inds]
        self.offsets = self.offsets[self.used_band_inds]
        self.data = self._pad_by_image_count(curves)

    @staticmethod
    def _read_metadata(metadata):
        """Load a per-SN metadata table, indexed by SNID."""
        if metadata is None:
            return None
        if isinstance(metadata, pd.DataFrame):
            table = metadata.copy()
        else:
            table = read_text_table(metadata, dtype={'SNID': str})[0]
        if 'SNID' not in table.columns:
            raise ValueError(f'metadata table needs an SNID column; has {list(table.columns)}')
        table['SNID'] = table['SNID'].astype(str)
        if table['SNID'].duplicated().any():
            raise ValueError('metadata table has duplicate SNID entries')
        clashing_columns = [c for c in table.columns if c in ('n_images', 'muhat') or re.fullmatch(r'image_\d+', c)]
        if clashing_columns:
            raise ValueError(f'metadata columns {clashing_columns} clash with values computed by process_dataset; '
                             f'rename them')
        return table.set_index('SNID')

    def _resolve_metadata(self, sn, metadata, values_for_all_sne):
        """
        Resolve the per-SN values used for fitting: process_dataset argument, then metadata table, then the
        photometry source itself.
        """
        name = sn['name']
        labels = sorted(set(sn['phot']['image']) | set(sn['peak_mjds'] or {}))
        row = metadata.loc[name] if metadata is not None and name in metadata.index else None

        def pick(key):
            """Return the value for key and its source: 0 argument, 1 metadata table, 2 photometry source."""
            if values_for_all_sne.get(key) is not None:
                return values_for_all_sne[key], 0
            if row is not None and key in row.index and pd.notna(row[key]):
                return row[key], 1
            return sn[key], 2

        (z, z_source), (z_cmb, z_cmb_source), (ebv_mw, _) = pick('z'), pick('z_cmb'), pick('ebv_mw')
        if z_cmb_source > z_source:  # a CMB-frame redshift from a less authoritative source than z is not used
            z_cmb = None

        if values_for_all_sne['peak_mjds'] is not None:
            if len(values_for_all_sne['peak_mjds']) != len(labels):
                raise ValueError(f'{name}: peak_mjds has {len(values_for_all_sne["peak_mjds"])} entries but the SN '
                                 f'has images {labels}')
            peaks = dict(zip(labels, values_for_all_sne['peak_mjds']))
        elif row is not None and all(pd.notna(row.get(f'peak_mjd_{label}')) for label in labels):
            peaks = {label: row[f'peak_mjd_{label}'] for label in labels}
        elif sn['peak_mjds'] is not None:
            peaks = sn['peak_mjds']
        elif sn['time_format'] == 'phase':  # without peaks, fitted peaks are relative to each image's phase zero
            peaks = {label: 0.0 for label in labels}
        else:
            peaks = None

        missing = [key for key, val in [('z', z), ('ebv_mw', ebv_mw), ('peak_mjds', peaks)] if val is None]
        if peaks is not None and not set(labels) <= set(peaks):
            missing.append(f'peak_mjds for images {sorted(set(labels) - set(peaks))}')
        if missing:
            raise ValueError(f'{name}: no value found for {missing}. Pass them as arguments, in a metadata table, or '
                             f'in the photometry file header.')

        info = {'SNID': name, 'n_images': len(labels), 'z': float(z),
                'z_cmb': np.nan if z_cmb is None else float(z_cmb), 'ebv_mw': float(ebv_mw),
                'muhat': float(self.cosmo.distmod(z_cmb if z_cmb is not None else z).value)}
        for i, label in enumerate(labels):
            info[f'image_{i}'] = label
            info[f'peak_mjd_{i}'] = float(peaks[label])
        if row is not None:
            info.update({key: val for key, val in row.items()
                         if key not in ('z', 'z_cmb', 'ebv_mw') and not key.startswith('peak_mjd_')})
        return info

    def _prepare_sn(self, sn, info, filt_map, drop_bands, error_floor, sigma_psf):
        """
        Remove observations with non-finite photometry or in dropped bands, map bands, compute rest-frame phases,
        remove observations outside the model's phase range and bands outside its wavelength coverage for one SN,
        then convert its photometry to flux.

        Returns
        -------
        lc : pd.DataFrame or None
            Columns ``image``, ``phase``, ``band``, ``flux``, ``fluxerr``.
        reason : str or None
            Why the SN was dropped, or None if it is kept.
        n_removed : int
            Number of observations removed for non-finite photometry.
        """
        labels = [info[f'image_{i}'] for i in range(info['n_images'])]
        if not labels:
            return None, 'no images', 0
        peaks = {label: info[f'peak_mjd_{i}'] for i, label in enumerate(labels)}
        z = info['z']

        phot = sn['phot']
        photometry = ['mag', 'magerr'] if 'mag' in phot.columns else ['flux', 'fluxerr']
        finite = np.isfinite(phot[photometry]).all(axis=1)
        n_removed = int((~finite).sum())
        lc = phot[finite & ~phot['band'].isin(drop_bands)].copy()
        lc['band'] = lc['band'].map(lambda b: filt_map.get(b, b))
        unknown = sorted(set(lc['band']) - set(self.band_dict))
        if unknown:
            raise KeyError(f'{sn["name"]}: bands {unknown} are not in the filter set; map them with filt_map or '
                           f'leave them out with drop_bands')

        if sn['time_format'] == 'mjd':
            lc['phase'] = (lc['time'] - lc['image'].map(peaks)) / (1 + z)
        elif sn['time_format'] == 'phase':
            lc['phase'] = lc['time']
        else:
            raise ValueError(f"{sn['name']}: time_format must be 'mjd' or 'phase', got {sn['time_format']!r}")
        lc = lc[(lc['phase'] > self.tau_knots[0]) & (lc['phase'] < self.tau_knots[-1])]

        low, high = self.l_knots[0], self.l_knots[-1]
        covered = [b for b in lc['band'].unique()
                   if self.band_lim_dict[b][0] / low - 1 >= z >= self.band_lim_dict[b][1] / high - 1]
        lc = lc[lc['band'].isin(covered)]
        empty = [label for label in labels if not (lc['image'] == label).any()]
        if empty:
            return None, f'no usable observations in images {empty}', n_removed

        if 'mag' in lc.columns:
            floor = lc['band'].map(lambda b: error_floor.get(b, 0))
            magerr = np.sqrt(lc['magerr'] ** 2 + floor ** 2 + sigma_psf ** 2)
            lc['flux'] = np.power(10, (ZPT - lc['mag']) / 2.5)
            lc['fluxerr'] = (np.log(10) / 2.5) * magerr * lc['flux']
        elif error_floor or sigma_psf:
            raise ValueError('error_floor and sigma_psf are magnitude errors and cannot be applied to flux '
                             'photometry')
        return lc[['image', 'phase', 'band', 'flux', 'fluxerr']], None, n_removed

    def _pad_by_image_count(self, curves):
        """
        Pad the prepared light curves into arrays, one set per number of images.

        Parameters
        ----------
        curves : list of pd.DataFrame
            Prepared light curves, in the order of ``self.sn_list``.

        Returns
        -------
        data : dict
            Keyed by number of images; see ``process_dataset``.
        """
        table = self.sn_list
        data = {}
        for n_img in sorted(table['n_images'].unique()):
            sn_index = np.flatnonzero(table['n_images'].values == n_img)
            n_sn = len(sn_index)
            n_obs = max(curves[i]['image'].value_counts().max() for i in sn_index)
            phase = np.zeros((n_obs, n_img, n_sn))
            flux = np.zeros((n_obs, n_img, n_sn))
            fluxerr = np.full((n_obs, n_img, n_sn), PAD_FLUXERR)
            band = np.zeros((n_obs, n_img, n_sn), dtype=int)
            mask = np.zeros((n_obs, n_img, n_sn), dtype=bool)
            for j, i in enumerate(sn_index):
                for k in range(n_img):
                    img = curves[i][curves[i]['image'] == table.at[i, f'image_{k}']]
                    n = len(img)
                    phase[:n, k, j] = img['phase']
                    flux[:n, k, j] = img['flux']
                    fluxerr[:n, k, j] = img['fluxerr']
                    band[:n, k, j] = img['band'].map(self.band_dict).map(self.used_band_dict)
                    mask[:n, k, j] = True
            muhat = np.tile(table['muhat'].values[sn_index], (n_img, 1))
            arrays = dict(zip(OBS_KEYS, (phase, flux, fluxerr, band, mask, muhat)))
            data[n_img] = {key: device_put(jnp.asarray(val)) for key, val in arrays.items()}
            data[n_img]['band_weights'] = self._calculate_band_weights(table['z'].values[sn_index],
                                                                       table['ebv_mw'].values[sn_index])
            data[n_img]['sn_index'] = sn_index
        return data

    # --- Fitting ---

    def fit(self, num_samples=500, num_warmup=500, num_chains=4, outputdir='results', chain_method='parallel',
            init_strategy='median', include_eps=True, include_ml=True):
        """
        Run MCMC fitting for lensed SN time delays.

        Data must be loaded first via ``process_dataset``. Each SN is fitted independently; SNe with the same number
        of images are fitted together in one vectorised run.

        Parameters
        ----------
        num_samples : int, optional
            Number of posterior samples per chain. Default 500.
        num_warmup : int, optional
            Number of warmup steps. Default 500.
        num_chains : int, optional
            Number of chains. Default 4.
        outputdir : str, optional
            Output directory path for results. Default 'results'.
        chain_method : str, optional
            'parallel', 'sequential', or 'vectorized'. Default 'parallel'.
        init_strategy : str, optional
            'median' or 'sample'. Default 'median'.
        include_eps : bool, optional
            Include epsilon in model. Default True.
        include_ml : bool, optional
            Include microlensing GP in model. Default True.

        Returns
        -------
        samples : dict
            Processed MCMC samples including time delays, with SNe on the last axis in the order of
            ``self.sn_list``.
        """
        if self.data is None:
            raise ValueError('No data loaded; call process_dataset first')
        init_strategies = {'median': init_to_median, 'sample': init_to_sample}
        if init_strategy not in init_strategies:
            raise ValueError(f'Invalid init strategy {init_strategy!r}, must be one of {list(init_strategies)}')

        def numpyro_model(data, weights):
            self.td_model(data, weights, include_eps=include_eps, include_ml=include_ml)

        def do_mcmc(data, weights):
            nuts_kernel = NUTS(numpyro_model, adapt_step_size=True,
                               init_strategy=init_strategies[init_strategy](), max_tree_depth=10)
            mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup,
                        num_chains=num_chains, chain_method=chain_method, progress_bar=False)
            data = jax.tree_util.tree_map(lambda x: x[..., None], data)
            mcmc.run(PRNGKey(SEED), data, weights[None, ...], extra_fields=['diverging'])
            return mcmc.get_samples(group_by_chain=True), mcmc.get_extra_fields(group_by_chain=True)

        start = timeit.default_timer()
        samples_by_n_img, extras_by_n_img = {}, {}
        for n_img, data in self.data.items():
            obs = {key: data[key] for key in OBS_KEYS}
            sn_axes = self._sn_axes(numpyro_model, obs, data['band_weights'])
            samples, extras = jax.vmap(do_mcmc, in_axes=(-1, 0))(obs, data['band_weights'])
            samples_by_n_img[n_img] = {key: self._sn_last(val, sn_axes[key]) for key, val in samples.items()}
            extras_by_n_img[n_img] = {key: self._sn_last(val, None) for key, val in extras.items()}
        end = timeit.default_timer()
        print(f'Inference time: {end - start:.2f} seconds')

        return self.fit_postprocess(self._merge_image_counts(samples_by_n_img),
                                    self._merge_image_counts(extras_by_n_img), outputdir)

    @staticmethod
    def _sn_axes(model, obs, band_weights):
        """
        Find the axis of the SN plate in every sample and deterministic site of ``model``, by tracing it with one SN
        and with two: the SN axis is the one axis whose size changes.
        """
        one = {key: val[..., :1] for key, val in obs.items()}
        two = {key: jnp.concatenate([val, val], axis=-1) for key, val in one.items()}
        weights = band_weights[:1]
        shapes = []
        for data, w in [(one, weights), (two, jnp.concatenate([weights, weights]))]:
            model_trace = numpyro.handlers.trace(numpyro.handlers.seed(model, 0)).get_trace(data, w)
            shapes.append({name: jnp.shape(site['value']) for name, site in model_trace.items()
                           if site['type'] in ('sample', 'deterministic') and not site.get('is_observed')})
        axes = {}
        for name, shape in shapes[0].items():
            changed = [i for i, (a, b) in enumerate(zip(shape, shapes[1][name])) if a != b]
            if len(changed) != 1:
                raise RuntimeError(f'Cannot locate the SN axis of site {name!r}: shapes {shape} and '
                                   f'{shapes[1][name]}')
            axes[name] = changed[0]
        return axes

    @staticmethod
    def _sn_last(values, sn_axis):
        """
        Convert a vmapped MCMC output of shape ``(n_sn, chains, samples, *site)`` to
        ``(chains, samples, *site without the SN plate axis, n_sn)``. Extra fields such as ``diverging`` have no
        plate axis (``sn_axis=None``).
        """
        values = np.asarray(values)
        if sn_axis is not None:
            values = np.take(values, 0, axis=3 + sn_axis)
        return np.moveaxis(values, 0, -1)

    def _merge_image_counts(self, samples_by_n_img):
        """
        Merge samples fitted separately for each number of images into arrays covering every SN, in the order of
        ``self.sn_list``. Axes that differ in size between image counts (e.g. the number of images) are padded with
        NaN, or False for booleans.
        """
        n_sn = len(self.sn_list)
        merged = {}
        for key in next(iter(samples_by_n_img.values())):
            parts = [(self.data[n_img]['sn_index'], samples[key]) for n_img, samples in samples_by_n_img.items()]
            shape = np.max([part.shape[:-1] for _, part in parts], axis=0)
            dtype = parts[0][1].dtype
            out = np.full((*shape, n_sn), False if dtype == bool else np.nan, dtype=dtype)
            for sn_index, part in parts:
                out[tuple(slice(0, n) for n in part.shape[:-1]) + (sn_index,)] = part
            merged[key] = out
        return merged

    def fit_lensed_sn(self, photometry, **kwargs):
        """
        Load data and fit one or more lensed SNe.

        Keyword arguments accepted by ``fit`` (``num_samples``, ``num_warmup``, ``num_chains``, ``outputdir``,
        ``chain_method``, ``init_strategy``, ``include_eps``, ``include_ml``) are passed to it; all others are passed
        to ``process_dataset``.

        Parameters
        ----------
        photometry : str, pd.DataFrame or list
            See ``process_dataset``.
        **kwargs
            Arguments for ``fit`` and ``process_dataset``.

        Returns
        -------
        samples : dict
            Processed MCMC samples including time delays.
        """
        fit_params = inspect.signature(self.fit).parameters
        fit_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in fit_params}
        self.process_dataset(photometry, **kwargs)
        return self.fit(**fit_kwargs)

    def fit_postprocess(self, samples, extras, outputdir):
        """
        Process MCMC output: compute time delays, save chains and summary statistics.

        Parameters
        ----------
        samples : dict
            MCMC samples, with SNe on the last axis.
        extras : dict
            Extra MCMC fields (divergences, etc.).
        outputdir : str
            Output directory path.

        Returns
        -------
        samples : dict
            Processed samples with time delays and distance moduli added.
        """
        os.makedirs(outputdir, exist_ok=True)
        with open(os.path.join(outputdir, 'initial_chains.pkl'), 'wb') as file:
            pickle.dump({**samples, **extras}, file)

        table = self.sn_list
        n_img = samples['tmax'].shape[-2]
        peak_guesses = table[[f'peak_mjd_{i}' for i in range(n_img)]].values.T
        z = table['z'].values
        muhat = table['muhat'].values

        samples['peak_mjd'] = peak_guesses + samples['tmax'] * (1 + z)
        samples['delta_t'] = samples['peak_mjd'][:, :, :1] - samples['peak_mjd'][:, :, 1:]

        sigma0, width = float(self.sigma0), DS_PRIOR_WIDTH
        Ds_var = width ** 2 + sigma0 ** 2
        mu_mean = (samples['Ds'] * width ** 2 + muhat * sigma0 ** 2) / Ds_var
        mu_sd = np.sqrt(sigma0 ** 2 * width ** 2 / Ds_var)
        samples['mu'] = np.random.default_rng(SEED).normal(mu_mean, mu_sd)
        samples['delM'] = samples['Ds'] - samples['mu']
        samples['delta'] = samples['mu'] - muhat

        with open(os.path.join(outputdir, 'chains.pkl'), 'wb') as file:
            pickle.dump({**samples, **extras}, file)
        arviz.summary(samples).to_csv(os.path.join(outputdir, 'fit_summary.csv'))
        table.to_csv(os.path.join(outputdir, 'sn_list.txt'), index=False)
        return samples

    # --- Simulation ---

    def simulate_light_curve_ml(self, mjds, N, bands, peak_mjds, z,
                                yerr=0, err_type='mag', zerr=1e-4, mu=0,
                                ebv_mw=0, RV=None, del_M=None, AV=None, theta=None,
                                eps=None, mag=True, save_to=None):
        """
        Simulates multi-image lensed light curves from the BayeSN model.

        Parameters
        ----------
        mjds : array-like
            Observer-frame observation MJDs (1D). If ``len(mjds) == len(bands)``,
            each observation uses the corresponding band (flat mode). Otherwise,
            every MJD is observed in every band (grid mode, band-major order).
        N : int
            Number of objects to simulate.
        bands : array-like
            List of bands. See ``mjds`` for flat vs grid semantics.
        peak_mjds : array-like
            Observer-frame peak MJD per image. Shape ``(N, n_images)``. Time
            delays between images are encoded by the differences between these
            values.
        z : float or array-like
            Source redshift. Scalar or shape ``(N,)``.
        yerr : float or array-like, optional
            Uncertainties. Default 0.
        err_type : str, optional
            'mag' or 'flux'. Default 'mag'.
        zerr : float, optional
            Redshift error. Default 1e-4.
        mu : float, array-like, or 'z', optional
            Distance modulus. Default 0. Shape should be (N, n_images).
        ebv_mw : float or array-like, optional
            MW E(B-V). Default 0.
        RV : float or array-like, optional
            Host R_V. Default uses model value.
        del_M : float or array-like, optional
            Grey offset. Default sampled from prior.
        AV : float or array-like, optional
            Host extinction. Default sampled from prior.
        theta : float or array-like, optional
            Theta. Default sampled from prior.
        eps : array-like or int, optional
            Epsilon. Default sampled from prior. Pass 0 to disable.
        mag : bool, optional
            Return magnitudes if True. Default True.
        save_to : str, optional
            If given, write the simulated light curve(s) to ECSV file(s) with
            true parameters in the YAML header. Images are labelled ``'A'``,
            ``'B'``, ... in the output file's ``image`` column. For ``N=1``,
            writes directly to ``save_to``; for ``N>1``, writes one file per
            SN with an index suffix (e.g. ``sim_0.ecsv``).

        Returns
        -------
        data : array-like
            Simulated flux or mag values. Shape ``(n_total_obs, n_images, N)``.
        yerr : array-like
            Corresponding errors.
        param_dict : dict
            Parameter values for each simulated object.
        """
        if del_M is None:
            del_M = self.sample_del_M(N)
        else:
            del_M = np.atleast_1d(np.array(del_M))
            if del_M.shape[0] == 1:
                del_M = del_M.repeat(N)
        if AV is None:
            AV = self.sample_AV(N)
        else:
            AV = np.atleast_1d(np.array(AV))
            if AV.shape[0] == 1:
                AV = AV.repeat(N)
        if theta is None:
            theta = self.sample_theta(N)
        else:
            theta = np.atleast_1d(np.array(theta))
            if theta.shape[0] == 1:
                theta = theta.repeat(N)
        if eps is None:
            eps = self.sample_epsilon(N)
        elif len(np.array(eps).shape) == 0:
            eps = np.array(eps)
            if eps == 0:
                eps = np.zeros((N, self.l_knots.shape[0], self.tau_knots.shape[0]))
            else:
                raise ValueError('For epsilon, pass an array or 0 to disable')
        ebv_mw = np.atleast_1d(np.array(ebv_mw))
        if ebv_mw.shape[0] == 1:
            ebv_mw = ebv_mw.repeat(N)
        peak_mjds = np.asarray(peak_mjds, dtype=float)
        if peak_mjds.ndim != 2 or peak_mjds.shape[0] != N:
            raise ValueError(f'peak_mjds must have shape (N={N}, n_images), got {peak_mjds.shape}')
        if RV is None:
            if self.model_type == 'fixed_RV':
                RV = self.RV
            else:
                # Sample from pop_RV distribution
                RV = np.random.normal(float(self.mu_R), float(self.sigma_R), N)
                if self.truncate_RV:
                    RV = np.clip(RV, float(self.trunc_val), None)
        RV = np.atleast_1d(np.array(RV))
        if RV.shape[0] == 1:
            RV = RV.repeat(N)
        z = np.atleast_1d(np.array(z))
        if z.shape[0] == 1:
            z = z.repeat(N)
        if type(mu) == str and mu == 'z':
            mu_scalar = self.cosmo.distmod(z).value
            mu = np.tile(np.atleast_1d(mu_scalar)[:, None], (1, peak_mjds.shape[1]))
        else:
            mu = np.atleast_1d(np.array(mu))
            if mu.shape[0] == 1:
                mu = mu.repeat(N, axis=0)

        param_dict = {
            'del_M': del_M, 'AV': AV, 'theta': theta, 'eps': eps,
            'z': z, 'mu': mu, 'ebv_mw': ebv_mw, 'RV': RV,
        }

        num_image = peak_mjds.shape[1]
        mjds = np.asarray(mjds, dtype=float)
        if mjds.ndim != 1:
            raise ValueError(f'mjds must be 1D, got shape {mjds.shape}')
        bands_arr = np.asarray(bands)
        num_bands = bands_arr.shape[0]
        for band in bands_arr:
            if band not in self.band_dict:
                raise ValueError(f'{band} not present in filters yaml file')
            if self.band_dict[band] not in self.used_band_dict:
                loaded = [self.inv_band_dict[i] for i in self.used_band_inds[1:]]
                raise ValueError(f'{band} is not one of the bands loaded by process_dataset ({loaded}); simulate it '
                                 f'with a new SEDmodel')

        if mjds.shape[0] == num_bands:
            # Flat mode: one band per observation
            mjds_flat = mjds
            per_obs_bands = bands_arr.copy()
            band_indices = np.array([self.used_band_dict[self.band_dict[b]] for b in bands_arr], dtype=int)
        else:
            # Grid mode: each MJD observed in every band (band-major order)
            num_per_band = mjds.shape[0]
            mjds_flat = np.tile(mjds, num_bands)
            per_obs_bands = np.repeat(bands_arr, num_per_band)
            band_indices = np.zeros(num_bands * num_per_band, dtype=int)
            for i, band in enumerate(bands_arr):
                band_indices[i * num_per_band: (i + 1) * num_per_band] = self.used_band_dict[self.band_dict[band]]
        band_indices = band_indices[:, None, None].repeat(num_image, axis=1).repeat(N, axis=2)

        mask = np.ones_like(band_indices)
        band_weights = self._calculate_band_weights(z, ebv_mw)

        # Rest-frame phase per (obs, N, image) from observer-frame MJDs and per-image peaks
        t = (jnp.asarray(mjds_flat)[:, None, None]
             - jnp.asarray(peak_mjds)[None, :, :]) / (1 + jnp.asarray(z)[None, :, None])
        hsiao_interp = jnp.array(
            [19 + jnp.floor(t), 19 + jnp.ceil(t), jnp.remainder(t, 1)]
        ).transpose(0, 1, 3, 2)
        keep_shape = t.shape
        t = t.flatten(order='F')
        vmap = jax.vmap(self.spline_coeffs_irr_step, in_axes=(0, None, None))
        J_t = vmap(t, self.tau_knots, self.KD_t).reshape(
            (*keep_shape, self.tau_knots.shape[0]), order='F').transpose(1, 2, 3, 0)
        t = t.reshape(keep_shape, order='F')
        beta_t = 0

        if mag:
            data = self.get_mag_batch(
                theta, AV, self.W0, self.W1, eps, (mu + del_M[:, None]).T, RV,
                band_indices, mask, J_t, hsiao_interp, band_weights, beta_t)
        else:
            data = self.get_flux_batch(
                theta, AV, self.W0, self.W1, eps, (mu + del_M[:, None]).T, RV,
                band_indices, mask, J_t, hsiao_interp, band_weights, beta_t)

        # Apply error
        yerr = jnp.array(yerr)
        if err_type == 'mag' and not mag:
            yerr = yerr * (np.log(10) / 2.5) * data
        if len(yerr.shape) == 0:
            yerr = np.ones_like(data) * yerr
        elif len(yerr.shape) == 1:
            yerr = np.repeat(yerr[..., None], N, axis=1)
        data = np.random.normal(data, yerr)

        if save_to is not None:
            write_ecsv(save_to, mjds_flat, per_obs_bands, peak_mjds, z, ebv_mw,
                       np.asarray(data), np.asarray(yerr), mag, param_dict)

        return data, yerr, param_dict
