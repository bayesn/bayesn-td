# bayesn-td

BayeSN for strongly-lensed Type Ia supernovae time-delay cosmography.

`bayesn-td` extends the [BayeSN](https://github.com/bayesn/bayesn) hierarchical SED model to jointly fit multiply-imaged Type Ia supernovae, inferring time delays, magnification ratios, and (optionally) achromatic microlensing effects using JAX/NumPyro.

The method is described in [Grayling et al. (2026), MNRAS, 548, 2](https://doi.org/10.1093/mnras/stag340) ([arXiv:2510.11719](https://arxiv.org/abs/2510.11719)).

## Documentation

Full documentation is hosted at [bayesn-td.readthedocs.io](https://bayesn-td.readthedocs.io).

## Installation

```bash
pip install .
```

For development:

```bash
pip install -e .
```

### Requirements

- Python >= 3.11
- JAX, NumPyro, NumPy, SciPy, pandas, Matplotlib, Astropy, h5py, extinction, ArviZ, ruamel.yaml, tqdm

## Usage

A complete working example with simulated LSST photometry of a 3-image lensed SN Ia is bundled in `examples/sim_lensed_sn/`.

### Command line

```bash
run_bayesn_td examples/sim_lensed_sn/input.yaml
```

### Python API

```python
from bayesn_td import SEDmodel

model = SEDmodel()
samples = model.fit_lensed_sn(
    photometry='examples/sim_lensed_sn/photometry.ecsv',
    output='results/sim_lensed_sn',
)
```

See `examples/sim_lensed_sn/README.md` for the true input parameters used to generate the simulation.

## Citation

If you use `bayesn-td` in your research, please cite Grayling et al. (2026):

```bibtex
@article{Grayling2026,
    title        = {BayeSN-TD: Time Delay and $H_0$ Estimation for Lensed SN H0pe},
    author       = {Grayling, M. and Thorp, S. and Mandel, K. S. and Pascale, M. and Pierel, J. D. R. and Hayes, E. E. and Larison, C. and Agrawal, A. and Narayan, G.},
    journal      = {Monthly Notices of the Royal Astronomical Society},
    volume       = {548},
    number       = {2},
    year         = {2026},
    doi          = {10.1093/mnras/stag340},
    eprint       = {2510.11719},
    archivePrefix = {arXiv},
    primaryClass = {astro-ph.CO},
}
```

## License

MIT
