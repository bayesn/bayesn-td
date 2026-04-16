# Simulated lensed SN Ia example

A toy example: a 3-image strongly-lensed Type Ia supernova at z=0.5, observed
in LSST g/r/i/z every 5 days from MJD 59980 to 60100. Generated using the
G26 extended model with no microlensing.

## True parameters

| Parameter | Value |
|---|---|
| z | 0.5 |
| ebv_mw | 0.02 |
| theta | 0.0 |
| AV | 0.1 |
| Image A peak MJD | 60000.0 |
| Image B peak MJD | 60030.0 |
| Image C peak MJD | 60060.0 |
| delta_t (A - B) | -30.0 days |
| delta_t (A - C) | -60.0 days |
| Image magnifications | [5.0, 3.0, 2.0] |
| Per-observation mag error | 0.03 |

## Running the fit

From the repository root:

```bash
run_bayesn_td examples/sim_lensed_sn/input.yaml
```

Or in Python — see `docs/fitting.rst` or `fitting_tutorial.ipynb`.

## Files

- `photometry.ecsv` — simulated photometry in astropy ECSV format, with true input parameters (z, ebv_mw, peak_mjds, true_theta, true_AV, etc.) in the YAML header
- `input.yaml` — config for the CLI
