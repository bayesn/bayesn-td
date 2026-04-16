"""Regenerate the simulated lensed SN Ia photometry file.

Run from the repository root:
    python examples/sim_lensed_sn/make_sim.py
"""
import os
import numpy as np
from bayesn_td import SEDmodel

Z = 0.5
EBV_MW = 0.02
THETA = 0.0
AV = 0.1
PEAK_MJDS = np.array([[60000.0, 60030.0, 60060.0]])  # (N=1, 3 images)
MAGNIFICATIONS = np.array([5.0, 3.0, 2.0])
BANDS = ['g_LSST', 'r_LSST', 'i_LSST', 'z_LSST']
MJDS = np.arange(59980, 60101, 5.0)
MAG_ERR = 0.03

np.random.seed(42)

model = SEDmodel(load_model='G26x_model')

# Distance modulus per image = cosmological distmod - 2.5 log10(magnification)
mu_z = float(model.cosmo.distmod(Z).value)
mu_per_image = mu_z - 2.5 * np.log10(MAGNIFICATIONS)
mu_arr = mu_per_image[None, :]

out_path = os.path.join(os.path.dirname(__file__), 'photometry.ecsv')

model.simulate_light_curve_ml(
    mjds=MJDS, N=1, bands=BANDS,
    peak_mjds=PEAK_MJDS, z=Z,
    yerr=MAG_ERR, err_type='mag',
    mu=mu_arr, ebv_mw=EBV_MW,
    theta=THETA, AV=AV, eps=0, mag=True,
    save_to=out_path,
)

print(f'Wrote simulated light curve to {out_path}')
