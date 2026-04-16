"""
Comprehensive test suite for bayesn-td refactor.
Tests + mutable state verification.
"""
import numpy as np
import os
import sys
import shutil
import time
import tempfile
import traceback

PYTHON = sys.executable
PHOTOMETRY = 'examples/sim_lensed_sn/photometry.ecsv'
SIM_KWARGS = dict(
    photometry=PHOTOMETRY,
    image_col='image', time_col='mjd', band_col='filter',
    flux_col='flux', fluxerr_col='fluxerr',
    mag_col='mag', magerr_col='magerr',
    peak_mjds=[60000.0, 60030.0, 60060.0],
    z=0.5, ebv_mw=0.02,
)

passed = 0
failed = 0
errors = []

def run_test(name, func):
    global passed, failed, errors
    try:
        func()
        print(f'  PASSED: {name}')
        passed += 1
    except Exception as e:
        print(f'  FAILED: {name} -- {e}')
        traceback.print_exc()
        failed += 1
        errors.append(name)


# ══════════════════════════════════════════════════════════════
# Group A: Imports & Model Loading (Tests 1-5)
# ══════════════════════════════════════════════════════════════
print('\n=== Group A: Imports & Model Loading ===')

def test_1():
    from bayesn_td import SEDmodel
    import inspect
    assert hasattr(SEDmodel, 'process_dataset')
    assert hasattr(SEDmodel, 'fit')
    assert hasattr(SEDmodel, 'simulate_light_curve_ml')
    assert not hasattr(SEDmodel, 'simulate_spectrum'), 'simulate_spectrum should be removed'
    assert not hasattr(SEDmodel, 'simulate_light_curve'), 'simulate_light_curve should be removed'
    sig = inspect.signature(SEDmodel.process_dataset)
    assert 'data_mode' not in sig.parameters, 'data_mode should be removed'
run_test('Test 1: Import and package structure', test_1)

def test_2():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='T21_model')
    assert model.model_type == 'fixed_RV'
    assert hasattr(model, 'RV')
    assert not hasattr(model, 'mu_R')
    assert not hasattr(model, 'sigma_R')
    for attr in ['W0', 'W1', 'l_knots', 'tau_knots', 'M0', 'sigma0', 'tauA', 'L_Sigma', 'band_dict', 'hsiao_flux']:
        assert hasattr(model, attr), f'Missing {attr}'
run_test('Test 2: Model loading - fixed_RV (T21)', test_2)

def test_3():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    assert model.model_type == 'pop_RV'
    assert hasattr(model, 'mu_R')
    assert hasattr(model, 'sigma_R')
    assert float(model.mu_R) > 0
    assert float(model.sigma_R) > 0
    for attr in ['W0', 'W1', 'l_knots', 'tau_knots', 'M0', 'sigma0', 'tauA', 'L_Sigma', 'band_dict', 'hsiao_flux']:
        assert hasattr(model, attr), f'Missing {attr}'
run_test('Test 3: Model loading - pop_RV (G26)', test_3)

def test_4():
    from bayesn_td import SEDmodel
    try:
        model = SEDmodel(load_model='nonexistent_model')
        assert False, 'Should have raised error'
    except Exception:
        pass
run_test('Test 4: Invalid model name', test_4)

def test_5():
    from bayesn_td import SEDmodel
    for name in ['T21_model', 'G26x_model']:
        m = SEDmodel(load_model=name)
        assert m.model_type in ('fixed_RV', 'pop_RV')
run_test('Test 5: Load all built-in models', test_5)


# ══════════════════════════════════════════════════════════════
# Group B: Data Loading (Tests 6-10)
# ══════════════════════════════════════════════════════════════
print('\n=== Group B: Data Loading ===')

def test_6():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)
    assert model.data.shape[0] == 9
    assert model.data.shape[2] == 3  # 3 images
    assert model.data.shape[3] == 1  # 1 SN
    assert model.band_weights is not None
    assert model.peak_mjds.shape == (3, 1)
run_test('Test 6: process_dataset - file path (mag-based)', test_6)

def test_7():
    import pandas as pd
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    df = pd.read_csv(PHOTOMETRY, sep=r'\s+', comment='#')
    kwargs = dict(SIM_KWARGS)
    kwargs['photometry'] = df
    model.process_dataset(**kwargs)
    assert model.data.shape[0] == 9
    assert model.data.shape[2] == 3
run_test('Test 7: process_dataset - DataFrame', test_7)

def test_8():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    kwargs = dict(SIM_KWARGS)
    kwargs['mag_col'] = None
    kwargs['magerr_col'] = None
    model.process_dataset(**kwargs)
    assert model.data.shape[0] == 9
run_test('Test 8: process_dataset - flux-based', test_8)

def test_9():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)
    n_bands_low = len(model.used_band_inds)

    model2 = SEDmodel(load_model='G26x_model')
    kwargs2 = dict(SIM_KWARGS)
    kwargs2['z'] = 1.5
    model2.process_dataset(**kwargs2)
    n_bands_high = len(model2.used_band_inds)
    assert n_bands_high < n_bands_low
run_test('Test 9: process_dataset - band filtering', test_9)

def test_10():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    kwargs = dict(SIM_KWARGS)
    kwargs['band_map'] = {'g_LSST': 'g_LSST', 'i_LSST': 'i_LSST'}
    model.process_dataset(**kwargs)
    assert model.data.shape[0] == 9
run_test('Test 10: process_dataset - band_map', test_10)


# ══════════════════════════════════════════════════════════════
# Group C: CLI (Tests 11-12)
# ══════════════════════════════════════════════════════════════
print('\n=== Group C: CLI ===')

def test_11():
    import subprocess
    result = subprocess.run([sys.executable, '-m', 'bayesn_td.cli', '--help'],
                          capture_output=True, text=True)
    assert result.returncode == 0
    for flag in ['--filters', '--outputdir', '--load_model', '--num_chains', '--num_warmup', '--num_samples']:
        assert flag in result.stdout, f'Missing flag {flag}'
run_test('Test 11: CLI entry point', test_11)

def test_12():
    import yaml, argparse
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='T21_model')
    config = {
        'load_model': 'T21_model', 'num_samples': 100, 'num_warmup': 100,
        'num_chains': 2, 'chain_method': 'sequential', 'init_strategy': 'median',
        'include_eps': True, 'include_ml': False, 'outputdir': '/tmp/test_cli',
        'data': {
            'photometry': PHOTOMETRY, 'image_col': 'image', 'time_col': 'mjd',
            'band_col': 'filter', 'flux_col': 'flux', 'fluxerr_col': 'fluxerr',
            'peak_mjds': [60000.0, 60030.0, 60060.0],
            'z': 0.5, 'ebv_mw': 0.02,
        }
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        tmp_path = f.name
    cmd_args = argparse.Namespace(
        load_model='G26x_model', num_chains=None, num_warmup=None, num_samples=None,
        chain_method=None, init_strategy=None, include_eps=None, include_ml=None,
        outputdir=None, filters=None,
    )
    with open(tmp_path) as f:
        args = yaml.safe_load(f)
    result = model.parse_yaml_input(args, cmd_args)
    assert result['num_samples'] == 100
    assert result['num_chains'] == 2
    os.unlink(tmp_path)
run_test('Test 12: CLI override logic', test_12)


# ══════════════════════════════════════════════════════════════
# Group D: MCMC Fitting (Tests 13-19)
# ══════════════════════════════════════════════════════════════
print('\n=== Group D: MCMC Fitting ===')

def test_13():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='T21_model')
    model.process_dataset(**SIM_KWARGS)
    out = '/tmp/bayesn_td_test_13'
    if os.path.exists(out): shutil.rmtree(out)
    samples = model.fit(num_samples=10, num_warmup=10, num_chains=1, output=out,
                        chain_method='sequential', init_strategy='median',
                        include_eps=False, include_ml=False)
    for key in ['theta', 'AV', 'tmax', 'Ds', 'delta_t', 'peak_mjd', 'mu', 'delM']:
        assert key in samples, f'Missing key {key}'
    assert 'RV' not in samples
    assert 'RV_tform' not in samples
    for f in ['chains.pkl', 'initial_chains.pkl', 'fit_summary.csv', 'sn_list.txt']:
        assert os.path.exists(os.path.join(out, f)), f'Missing output file {f}'
run_test('Test 13: Fit - fixed_RV, no eps, no ml', test_13)

def test_14():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)
    out = '/tmp/bayesn_td_test_14'
    if os.path.exists(out): shutil.rmtree(out)
    samples = model.fit(num_samples=10, num_warmup=10, num_chains=1, output=out,
                        chain_method='sequential', init_strategy='median',
                        include_eps=True, include_ml=False)
    assert 'RV' in samples
    assert 'RV_tform' in samples
    assert np.all(np.array(samples['RV']) > 1.2)
run_test('Test 14: Fit - pop_RV, with eps, no ml', test_14)

def test_15():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)
    out = '/tmp/bayesn_td_test_15'
    if os.path.exists(out): shutil.rmtree(out)
    samples = model.fit(num_samples=10, num_warmup=10, num_chains=1, output=out,
                        chain_method='sequential', init_strategy='median',
                        include_eps=True, include_ml=True)
    for key in ['A', 'beta_t']:
        assert key in samples, f'Missing ML key {key}'
    dt = samples['delta_t']
    assert dt.shape[2] == 2  # 3 images - 1
run_test('Test 15: Fit - pop_RV, with eps, with ml', test_15)

def test_16():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)
    out = '/tmp/bayesn_td_test_16'
    if os.path.exists(out): shutil.rmtree(out)
    samples = model.fit(num_samples=10, num_warmup=10, num_chains=1, output=out,
                        chain_method='sequential', init_strategy='median',
                        include_eps=False, include_ml=False)
    assert 'RV' in samples
    assert 'theta' in samples
    assert 'delta_t' in samples
run_test('Test 16: Fit - pop_RV, no eps, no ml', test_16)

def test_17():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='T21_model')
    model.process_dataset(**SIM_KWARGS)
    out = '/tmp/bayesn_td_test_17'
    if os.path.exists(out): shutil.rmtree(out)
    samples = model.fit(num_samples=10, num_warmup=10, num_chains=2, output=out,
                        chain_method='parallel', init_strategy='median',
                        include_eps=False, include_ml=False)
    theta = np.array(samples['theta'])
    assert theta.shape[0] == 2, f'Expected 2 chains, got {theta.shape[0]}'
run_test('Test 17: Fit - multi-chain parallel', test_17)

def test_18():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='T21_model')
    out = '/tmp/bayesn_td_test_18'
    if os.path.exists(out): shutil.rmtree(out)
    samples = model.fit_lensed_sn(
        **SIM_KWARGS,
        num_samples=10, num_warmup=10, num_chains=1,
        chain_method='sequential', init_strategy='median',
        include_eps=False, include_ml=False, output=out,
    )
    for key in ['theta', 'AV', 'delta_t']:
        assert key in samples, f'Missing key {key}'
run_test('Test 18: fit_lensed_sn convenience', test_18)

def test_19():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)
    out = '/tmp/bayesn_td_test_19'
    if os.path.exists(out): shutil.rmtree(out)
    samples = model.fit(num_samples=10, num_warmup=10, num_chains=1, output=out,
                        chain_method='sequential', init_strategy='sample',
                        include_eps=False, include_ml=False)
    assert 'theta' in samples
    assert 'delta_t' in samples
run_test('Test 19: Fit - init_strategy=sample', test_19)


# ══════════════════════════════════════════════════════════════
# Group E: Simulation & File Output (Tests 20-24)
# ══════════════════════════════════════════════════════════════
print('\n=== Group E: Simulation & File Output ===')

def test_20():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    mjds = np.linspace(59990, 60040, 10)
    data, yerr, params = model.simulate_light_curve_ml(
        mjds=mjds, N=2, bands=['F115W', 'F150W', 'F200W'],
        peak_mjds=np.zeros((2, 2)) + 60010, z=0.5,
        ebv_mw=0.01, mu=np.zeros((2, 2)), mag=True,
    )
    assert data.shape[1] == 2  # images
    assert data.shape[2] == 2  # SNe
    for key in ['del_M', 'AV', 'theta', 'eps', 'z', 'mu', 'ebv_mw', 'RV']:
        assert key in params, f'Missing param {key}'
    assert np.all(np.isfinite(data))
run_test('Test 20: simulate_light_curve_ml - basic', test_20)

def test_21():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    mjds = np.linspace(59990, 60040, 10)
    data, yerr, params = model.simulate_light_curve_ml(
        mjds=mjds, N=2, bands=['F115W', 'F150W'],
        peak_mjds=np.zeros((2, 2)) + 60010, z=0.5,
        ebv_mw=0.01, mu=np.zeros((2, 2)), mag=False,
    )
    assert np.all(data > 0)
run_test('Test 21: simulate_light_curve_ml - flux', test_21)

def test_22():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    mjds = np.linspace(59990, 60040, 10)
    data, yerr, params = model.simulate_light_curve_ml(
        mjds=mjds, N=2, bands=['F115W', 'F150W'],
        peak_mjds=np.zeros((2, 1)) + 60010, z=0.5,
        ebv_mw=0.01, eps=0, mu=np.zeros((2, 1)), mag=True,
    )
    assert np.all(params['eps'] == 0)
run_test('Test 22: simulate_light_curve_ml - eps=0', test_22)

# ══════════════════════════════════════════════════════════════
# Group F: Output Validation (Tests 25-26)
# ══════════════════════════════════════════════════════════════
print('\n=== Group F: Output Validation ===')

def test_25():
    import pickle, pandas as pd
    out = '/tmp/bayesn_td_test_13'
    assert os.path.exists(out), 'Test 13 output missing'
    with open(os.path.join(out, 'chains.pkl'), 'rb') as f:
        chains = pickle.load(f)
    for key in ['theta', 'AV', 'tmax', 'Ds', 'delta_t', 'peak_mjd', 'mu', 'delM']:
        assert key in chains, f'Missing key {key} in chains'
    summary = pd.read_csv(os.path.join(out, 'fit_summary.csv'))
    assert len(summary) > 0
    sn_list = pd.read_csv(os.path.join(out, 'sn_list.txt'))
    assert 'sn' in sn_list.columns
run_test('Test 25: Verify output file contents', test_25)

def test_26():
    import pickle
    out = '/tmp/bayesn_td_test_13'
    with open(os.path.join(out, 'chains.pkl'), 'rb') as f:
        chains = pickle.load(f)
    with open(os.path.join(out, 'initial_chains.pkl'), 'rb') as f:
        initial = pickle.load(f)
    for key in ['delta_t', 'peak_mjd', 'mu', 'delM']:
        assert key in chains, f'{key} missing from chains'
        assert key not in initial, f'{key} should not be in initial_chains'
run_test('Test 26: Verify initial_chains vs chains', test_26)


# ══════════════════════════════════════════════════════════════
# Group G: Mutable State Fix Verification
# ══════════════════════════════════════════════════════════════
print('\n=== Group G: Mutable State Fix ===')

def test_repeat_process_dataset():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)
    data1 = np.array(model.data)
    bw1 = np.array(model.band_weights)
    zps1 = np.array(model.zps)
    offsets1 = np.array(model.offsets)
    ubi1 = np.array(model.used_band_inds)

    model.process_dataset(**SIM_KWARGS)
    data2 = np.array(model.data)
    bw2 = np.array(model.band_weights)
    zps2 = np.array(model.zps)
    offsets2 = np.array(model.offsets)
    ubi2 = np.array(model.used_band_inds)

    assert np.array_equal(data1, data2), 'data tensors differ'
    assert np.allclose(bw1, bw2), 'band_weights differ'
    assert np.array_equal(zps1, zps2), 'zps differ'
    assert np.array_equal(offsets1, offsets2), 'offsets differ'
    assert np.array_equal(ubi1, ubi2), 'used_band_inds differ'
run_test('Mutable state: Repeated process_dataset identical', test_repeat_process_dataset)

def test_redshift_roundtrip():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')

    model.process_dataset(**SIM_KWARGS)
    bw_a = np.array(model.band_weights)
    zps_a = np.array(model.zps)
    offsets_a = np.array(model.offsets)

    kwargs_alt = dict(SIM_KWARGS)
    kwargs_alt['z'] = 1.0
    model.process_dataset(**kwargs_alt)
    bw_b = np.array(model.band_weights)
    assert bw_a.shape != bw_b.shape or not np.allclose(bw_a, bw_b), 'band_weights should differ for different z'

    model.process_dataset(**SIM_KWARGS)
    bw_c = np.array(model.band_weights)
    zps_c = np.array(model.zps)
    offsets_c = np.array(model.offsets)

    assert np.allclose(bw_a, bw_c), 'band_weights differ after round-trip'
    assert np.array_equal(zps_a, zps_c), 'zps differ after round-trip'
    assert np.array_equal(offsets_a, offsets_c), 'offsets differ after round-trip'
run_test('Mutable state: Different redshift round-trip', test_redshift_roundtrip)

def test_simulate_after_process_dataset():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)

    mjds = np.linspace(59990, 60040, 20)
    data_sim, yerr_sim, params = model.simulate_light_curve_ml(
        mjds=mjds, N=1, bands=['F115W', 'F150W'],
        peak_mjds=np.array([[60010.0]]), z=1.0,
        ebv_mw=0.01, mu=np.zeros((1, 1)), mag=True,
    )
    assert np.all(np.isfinite(data_sim)), 'Non-finite values in simulation after process_dataset'
    expected_rows = len(mjds) * 2  # 2 bands
    assert data_sim.shape[0] == expected_rows, f'Wrong shape: {data_sim.shape[0]} != {expected_rows}'
run_test('Mutable state: simulate_light_curve_ml after process_dataset', test_simulate_after_process_dataset)

def test_band_weights_not_mutated():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    shape_before = model.band_interpolate_weights.shape
    model.process_dataset(**SIM_KWARGS)
    shape_after = model.band_interpolate_weights.shape
    assert shape_before == shape_after, f'band_interpolate_weights mutated: {shape_before} -> {shape_after}'
run_test('Mutable state: band_interpolate_weights preserved', test_band_weights_not_mutated)

def test_repeated_fit():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='T21_model')

    model.process_dataset(**SIM_KWARGS)
    data1 = np.array(model.data)
    bw1 = np.array(model.band_weights)
    zps1 = np.array(model.zps)
    offsets1 = np.array(model.offsets)
    ubi1 = np.array(model.used_band_inds)
    out1 = '/tmp/bayesn_td_test_repeat_1'
    if os.path.exists(out1): shutil.rmtree(out1)
    samples1 = model.fit(num_samples=10, num_warmup=10, num_chains=1, output=out1,
                         chain_method='sequential', init_strategy='median',
                         include_eps=False, include_ml=False)
    assert 'delta_t' in samples1

    # Second call on same instance
    model.process_dataset(**SIM_KWARGS)
    data2 = np.array(model.data)
    bw2 = np.array(model.band_weights)
    zps2 = np.array(model.zps)
    offsets2 = np.array(model.offsets)
    ubi2 = np.array(model.used_band_inds)

    # Verify the model sees identical input data both times
    assert np.array_equal(data1, data2), 'data tensors differ on second process_dataset before fit'
    assert np.allclose(bw1, bw2), 'band_weights differ on second process_dataset before fit'
    assert np.array_equal(zps1, zps2), 'zps differ on second process_dataset before fit'
    assert np.array_equal(offsets1, offsets2), 'offsets differ on second process_dataset before fit'
    assert np.array_equal(ubi1, ubi2), 'used_band_inds differ on second process_dataset before fit'

    out2 = '/tmp/bayesn_td_test_repeat_2'
    if os.path.exists(out2): shutil.rmtree(out2)
    samples2 = model.fit(num_samples=10, num_warmup=10, num_chains=1, output=out2,
                         chain_method='sequential', init_strategy='median',
                         include_eps=False, include_ml=False)
    assert 'delta_t' in samples2

    # Both should have same keys and shapes
    for key in ['theta', 'AV', 'delta_t']:
        assert key in samples2, f'Missing key {key} on second fit'
        assert samples1[key].shape == samples2[key].shape, f'Shape mismatch for {key}'
    shutil.rmtree(out1)
    shutil.rmtree(out2)
run_test('Mutable state: Repeated process_dataset + fit', test_repeated_fit)


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print(f'\n{"=" * 50}')
print(f'Results: {passed} passed, {failed} failed out of {passed + failed}')
if failed == 0:
    print('All tests passed!')
else:
    print(f'FAILED tests: {", ".join(errors)}')
    sys.exit(1)
