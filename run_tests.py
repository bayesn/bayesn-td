"""
Comprehensive test suite for bayesn-td.
"""
import numpy as np
import os
import sys
import shutil
import tempfile
import traceback

PYTHON = sys.executable
PHOTOMETRY = 'examples/sim_lensed_sn/photometry.ecsv'
PEAK_MJDS = [60000.0, 60030.0, 60060.0]
SIM_KWARGS = dict(
    photometry=PHOTOMETRY,
    image_col='image', time_col='mjd', band_col='filter',
    flux_col='flux', fluxerr_col='fluxerr',
    mag_col='mag', magerr_col='magerr',
    peak_mjds=PEAK_MJDS,
    z=0.5, ebv_mw=0.02,
)
FIT_KWARGS = dict(num_samples=10, num_warmup=10, num_chains=1, chain_method='sequential',
                  init_strategy='median')
TMP = tempfile.mkdtemp(prefix='bayesn_td_tests_')

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


def tmp_path(*parts):
    path = os.path.join(TMP, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def example_table():
    """The bundled example photometry as a DataFrame."""
    from astropy.table import Table
    return Table.read(PHOTOMETRY, format='ascii.ecsv').to_pandas()


def write_sn(df, name, **meta):
    """Write photometry to an ECSV file named after the SN, with metadata in its header; return the path."""
    from astropy.table import Table
    table = Table.from_pandas(df)
    table.meta = meta
    path = tmp_path('sne', f'{name}.ecsv')
    table.write(path, format='ascii.ecsv', overwrite=True)
    return path


def only_data(model):
    """The padded arrays of a model whose SNe all have the same number of images."""
    assert len(model.data) == 1, f'expected one image count, got {list(model.data)}'
    return next(iter(model.data.values()))


def raises(exc_type, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except exc_type as e:
        return e
    raise AssertionError(f'Expected {exc_type.__name__}')


# ══════════════════════════════════════════════════════════════
# Group A: Imports & Model Loading
# ══════════════════════════════════════════════════════════════
print('\n=== Group A: Imports & Model Loading ===')

def test_1():
    from bayesn_td import SEDmodel
    import inspect
    assert hasattr(SEDmodel, 'process_dataset')
    assert hasattr(SEDmodel, 'fit')
    assert hasattr(SEDmodel, 'simulate_light_curve_ml')
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
# Group B: Data Loading
# ══════════════════════════════════════════════════════════════
print('\n=== Group B: Data Loading ===')

def test_6():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)
    assert list(model.data) == [3]
    data = model.data[3]
    assert data['flux'].shape[1:] == (3, 1)  # 3 images, 1 SN
    assert data['muhat'].shape == (3, 1)
    assert data['band_weights'].shape[0] == 1
    assert list(model.sn_list['n_images']) == [3]
    assert [model.sn_list.at[0, f'peak_mjd_{i}'] for i in range(3)] == PEAK_MJDS
run_test('Test 6: process_dataset - file path (mag-based)', test_6)

def test_7():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**dict(SIM_KWARGS, photometry=example_table()))
    assert list(model.sn_list['SNID']) == ['SN']
    assert only_data(model)['flux'].shape[1] == 3
run_test('Test 7: process_dataset - DataFrame', test_7)

def test_8():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    df = example_table()[['mjd', 'filter', 'flux', 'fluxerr', 'image']]
    model.process_dataset(df, z=0.5, ebv_mw=0.02, peak_mjds=PEAK_MJDS)
    data = only_data(model)
    used = {model.inv_band_dict[i] for i in model.used_band_inds}
    # bands outside the model's wavelength coverage and observations outside its phase range are cut
    phase = (df['mjd'] - df['image'].map(dict(zip('ABC', PEAK_MJDS)))) / 1.5
    in_range = (phase > float(model.tau_knots[0])) & (phase < float(model.tau_knots[-1]))
    expected = df.loc[df['filter'].isin(used) & in_range, 'flux'].values
    assert np.allclose(np.sort(np.asarray(data['flux'])[np.asarray(data['mask'])]), np.sort(expected))
run_test('Test 8: process_dataset - flux-based', test_8)

def test_9():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)
    n_bands_low = len(model.used_band_inds)
    kwargs2 = dict(SIM_KWARGS)
    kwargs2['z'] = 1.5
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**kwargs2)
    n_bands_high = len(model.used_band_inds)
    assert n_bands_high < n_bands_low
run_test('Test 9: process_dataset - band filtering', test_9)

def test_10():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    df = example_table()
    df['filter'] = df['filter'].str[0]  # g_LSST -> g
    kwargs = dict(SIM_KWARGS, photometry=df)
    raises(KeyError, model.process_dataset, **kwargs)  # unmapped bands must raise, not be dropped
    model.process_dataset(**kwargs, filt_map={b: f'{b}_LSST' for b in 'griz'})
    mapped = list(model.used_band_inds)
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)
    assert mapped == list(model.used_band_inds), 'mapped bands must match the original names'
run_test('Test 10: process_dataset - filt_map, unknown bands raise', test_10)

def test_mag_errors():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    df = example_table()
    model.process_dataset(**SIM_KWARGS)
    data = only_data(model)
    rel = np.asarray(data['fluxerr'])[np.asarray(data['mask'])] / np.asarray(data['flux'])[np.asarray(data['mask'])]
    assert np.allclose(rel, np.log(10) / 2.5 * df['magerr'].values[0]), 'sigma_F/F must be (ln10/2.5) sigma_m'
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS, sigma_psf=0.04)
    data = only_data(model)
    rel = np.asarray(data['fluxerr'])[np.asarray(data['mask'])] / np.asarray(data['flux'])[np.asarray(data['mask'])]
    assert np.allclose(rel, np.log(10) / 2.5 * np.hypot(df['magerr'].values[0], 0.04)), \
        'sigma_psf must apply without error_floor'
    flux_only = example_table()[['mjd', 'filter', 'flux', 'fluxerr', 'image']]
    kwargs = dict(z=0.5, ebv_mw=0.02, peak_mjds=PEAK_MJDS)
    raises(ValueError, SEDmodel(load_model='G26x_model').process_dataset, flux_only, **kwargs, sigma_psf=0.04)
    raises(ValueError, SEDmodel(load_model='G26x_model').process_dataset, flux_only, **kwargs,
           error_floor={'g_LSST': 0.02})
run_test('Data: magnitude errors, error floor and sigma_psf', test_mag_errors)

def test_coverage_drop():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    full = example_table()
    # at z=0.5, g_LSST lies outside the model's wavelength coverage, so an image observed only in g has no data
    g_only_b = full[(full['image'] != 'B') | (full['filter'] == 'g_LSST')]
    meta = dict(z=0.5, ebv_mw=0.02, peak_mjds=PEAK_MJDS)
    paths = [write_sn(full, 'full', **meta), write_sn(g_only_b, 'g_only_b', **meta)]
    model.process_dataset(paths)
    assert 'g_LSST' not in {model.inv_band_dict[i] for i in model.used_band_inds}
    assert list(model.sn_list['SNID']) == ['full']
    assert list(model.dropped['SNID']) == ['g_only_b']
    assert "['B']" in model.dropped['reason'].iloc[0]
    raises(ValueError, SEDmodel(load_model='G26x_model').process_dataset, paths[1])  # no SNe left
run_test('Data: bands outside model coverage removed, empty images dropped', test_coverage_drop)

def test_phase_format():
    from bayesn_td import SEDmodel
    df = example_table()
    phases = (df['mjd'] - df['image'].map(dict(zip('ABC', PEAK_MJDS)))) / 1.5
    phase_df = df.drop(columns='mjd').assign(phase=phases)
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(phase_df, time_format='phase', z=0.5, ebv_mw=0.02, peak_mjds=PEAK_MJDS)
    data = only_data(model)
    used = {model.inv_band_dict[i] for i in model.used_band_inds}
    in_range = (phases > float(model.tau_knots[0])) & (phases < float(model.tau_knots[-1]))
    expected = np.sort(phases[df['filter'].isin(used) & in_range].values)
    assert np.allclose(np.sort(np.asarray(data['phase'])[np.asarray(data['mask'])]), expected), \
        'phase data must be used as given'
    assert [model.sn_list.at[0, f'peak_mjd_{i}'] for i in range(3)] == PEAK_MJDS, 'peaks place the fit on the MJD clock'
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(phase_df, time_format='phase', z=0.5, ebv_mw=0.02)
    assert [model.sn_list.at[0, f'peak_mjd_{i}'] for i in range(3)] == [0.0, 0.0, 0.0]
run_test('Data: time_format=phase uses times as given; peaks only place the fit', test_phase_format)

def test_drop_bands():
    from bayesn_td import SEDmodel
    df = example_table()
    df['filter'] = df['filter'].str[0]  # g_LSST -> g
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(df, z=0.5, ebv_mw=0.02, peak_mjds=PEAK_MJDS,
                          filt_map={b: f'{b}_LSST' for b in 'griz'}, drop_bands=['i'])
    used = {model.inv_band_dict[i] for i in model.used_band_inds}
    assert 'i_LSST' not in used and {'r_LSST', 'z_LSST'} <= used, 'drop_bands uses the band names in the data'
run_test('Data: drop_bands', test_drop_bands)

def test_error_floor():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS, error_floor={'r_LSST': 0.1})
    data = only_data(model)
    mask = np.asarray(data['mask'])
    rel = np.asarray(data['fluxerr'])[mask] / np.asarray(data['flux'])[mask]
    bands = np.asarray(data['band'])[mask]
    r = bands == model.used_band_dict[model.band_dict['r_LSST']]
    assert r.any() and (~r).any()
    assert np.allclose(rel[r], np.log(10) / 2.5 * np.hypot(0.03, 0.1)), 'error floor applies to its band'
    assert np.allclose(rel[~r], np.log(10) / 2.5 * 0.03), 'error floor leaves other bands alone'
run_test('Data: error_floor applied per band', test_error_floor)

def test_nonfinite_removed():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)
    n_full = int(np.asarray(only_data(model)['mask']).sum())
    df = example_table()
    used = {model.inv_band_dict[i] for i in model.used_band_inds}
    row = df.index[df['filter'].isin(used) & (df['image'] == 'A') & (df['mjd'] == 60000)][0]
    df.loc[row, 'mag'] = np.nan
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**dict(SIM_KWARGS, photometry=df))
    data = only_data(model)
    assert int(np.asarray(data['mask']).sum()) == n_full - 1, 'exactly the non-finite observation is removed'
    assert np.isfinite(np.asarray(data['flux'])).all()
run_test('Data: non-finite observations removed', test_nonfinite_removed)

def test_one_dataset_per_model():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)
    error = raises(ValueError, model.process_dataset, **SIM_KWARGS)
    assert 'new SEDmodel' in str(error)
    error = raises(ValueError, model.simulate_light_curve_ml, mjds=np.linspace(59990, 60040, 5), N=1,
                   bands=['F150W'], peak_mjds=np.array([[60010.0]]), z=0.5, mu='z')
    assert 'not one of the bands loaded' in str(error), 'simulating an unloaded band gives a clear error'
run_test('Data: one dataset per SEDmodel; simulating unloaded bands raises clearly', test_one_dataset_per_model)


# ══════════════════════════════════════════════════════════════
# Group B2: Readers
# ══════════════════════════════════════════════════════════════
print('\n=== Group B2: Readers ===')

def test_reader_whitespace_csv():
    from bayesn_td.io import read_photometry
    path = tmp_path('readers', 'whitespace.csv')
    example_table().to_csv(path, sep=' ', index=False)
    rec, = read_photometry(path)
    assert rec['name'] == 'whitespace'
    assert sorted(rec['phot']['image'].unique()) == ['A', 'B', 'C']
run_test('Reader: whitespace-delimited file named .csv', test_reader_whitespace_csv)

def test_reader_zp():
    from bayesn_td.io import read_photometry
    df = example_table()[['mjd', 'filter', 'flux', 'fluxerr', 'image']]
    scaled = df.assign(flux=df['flux'] / 10, fluxerr=df['fluxerr'] / 10, zp=25.0)
    rec, = read_photometry(scaled)
    assert np.allclose(rec['phot']['flux'], df['flux']), 'zp=25 flux must be rescaled to zp=27.5'
run_test('Reader: zp column rescales flux', test_reader_zp)

def test_reader_snana():
    from astropy.io import fits
    from astropy.table import Table
    from bayesn_td.io import read_photometry
    df = example_table()
    hdus = [fits.PrimaryHDU()]
    for i, img in enumerate(['A', 'B']):
        part = df[df['image'] == img]
        hdu = fits.BinTableHDU(Table({'time': part['mjd'].values, 'band': part['filter'].values.astype(str),
                                      'flux': part['flux'].values, 'fluxerr': part['fluxerr'].values,
                                      'zp': np.full(len(part), 27.5)}))
        hdu.header.update({'ZHEL': 0.5, 'ZFIN': 0.501, 'MWEBV': 0.03, 'PEAKMJD': PEAK_MJDS[i] + 1,
                           'SZHEL': 0.5, 'SZCMB': 0.5011, 'SMWEBV': 0.02, 'SPEAKMJD': PEAK_MJDS[i]})
        hdus.append(hdu)
    path = tmp_path('readers', 'snana.dat')
    fits.HDUList(hdus).writeto(path, overwrite=True)
    obs, = read_photometry(path)
    true, = read_photometry(path, true_values=True)
    assert (obs['z'], obs['z_cmb'], obs['ebv_mw']) == (0.5, 0.501, 0.03)
    assert obs['peak_mjds'] == {'image_1': 60001.0, 'image_2': 60031.0}
    assert (true['z_cmb'], true['ebv_mw']) == (0.5011, 0.02)
    assert true['peak_mjds'] == {'image_1': 60000.0, 'image_2': 60030.0}
run_test('Reader: SNANA multi-HDU FITS, observed and true values', test_reader_snana)

def test_reader_lsst_pickle():
    import pandas as pd
    from bayesn_td.io import read_photometry
    mags = np.array([[22.0, 23.0], [np.inf, 22.5], [21.5, 22.8]])
    errs = np.array([[0.05, 0.1], [np.nan, 0.1], [0.04, 0.09]])
    cat = pd.DataFrame({'obs_times': [np.array([-5.0, 0.0, 5.0])], 'obs_bands': [np.array(['g', 'r', 'i'])],
                        'obs_mag_micro': [mags], 'mag_micro_error': [errs],
                        'time_delay': [np.array([0.0, 12.0])], 'z_source': [0.8]}, index=[7])
    path = tmp_path('readers', 'catalogue.pkl')
    cat.to_pickle(path)
    for source, kwargs in [(path, {}), (cat, {'format': 'lsst_lensed_pickle'})]:
        rec, = read_photometry(source, **kwargs)
        assert rec['name'] == '7' and rec['z'] == 0.8
        assert rec['peak_mjds'] == {'A': 0.0, 'B': 12.0}
        assert len(rec['phot']) == 6, 'the reader keeps every observation; process_dataset removes non-finite ones'
        assert set(rec['phot']['band']) == {'g_LSST', 'r_LSST', 'i_LSST'}
run_test('Reader: LSST lensed-SN pickle (file and DataFrame)', test_reader_lsst_pickle)


# ══════════════════════════════════════════════════════════════
# Group B3: Batches
# ══════════════════════════════════════════════════════════════
print('\n=== Group B3: Batches ===')

def write_batch(n=3):
    """Write n copies of the example photometry to a directory; return their paths."""
    paths = []
    for i in range(n):
        path = tmp_path('batch', f'sn_{i}.ecsv')
        shutil.copy(PHOTOMETRY, path)
        paths.append(path)
    return paths

def test_batch_sources():
    from bayesn_td import SEDmodel
    paths = write_batch()
    for source in [paths, os.path.dirname(paths[0]), os.path.join(os.path.dirname(paths[0]), 'sn_*.ecsv')]:
        model = SEDmodel(load_model='G26x_model')
        model.process_dataset(source)
        assert list(model.sn_list['SNID']) == ['sn_0', 'sn_1', 'sn_2']
        assert only_data(model)['flux'].shape[1:] == (3, 3)
    raises(ValueError, SEDmodel(load_model='G26x_model').process_dataset, [PHOTOMETRY, PHOTOMETRY])  # duplicate names
run_test('Batch: list, directory and glob sources; duplicate names raise', test_batch_sources)

def test_batch_metadata():
    import pandas as pd
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    df = example_table()[['mjd', 'filter', 'mag', 'magerr', 'image']]
    sources = [write_sn(df, 'a'), write_sn(df, 'b')]
    raises(ValueError, model.process_dataset, sources)  # no z, ebv_mw or peaks anywhere
    meta = pd.DataFrame({'SNID': ['a', 'b'], 'z': [0.5, 0.6], 'ebv_mw': [0.02, 0.03], 'peak_mjd_A': [60000., 60001.],
                         'peak_mjd_B': [60030., 60031.], 'peak_mjd_C': [60060., 60061.], 'true_dt': [1.0, 2.0]})
    model.process_dataset(sources, metadata=meta)
    m = model.sn_list
    assert list(m['z']) == [0.5, 0.6] and list(m['peak_mjd_0']) == [60000., 60001.]
    assert list(m['true_dt']) == [1.0, 2.0], 'extra metadata columns must be carried through'
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(sources, metadata=meta, z=0.7)
    assert list(model.sn_list['z']) == [0.7, 0.7], 'explicit arguments take precedence'
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(PHOTOMETRY, metadata=pd.DataFrame({'SNID': ['photometry'], 'z': [0.9]}))
    assert model.sn_list.at[0, 'z'] == 0.9, 'metadata takes precedence over the file header'
    assert model.sn_list.at[0, 'ebv_mw'] == 0.02, 'file header fills what metadata lacks'
run_test('Batch: metadata precedence and pass-through', test_batch_metadata)

def mixed_paths():
    """A triple (the example) and a double, as files."""
    double = write_sn(example_table().query('image != "C"'), 'double', z=0.5, ebv_mw=0.02,
                      peak_mjds=PEAK_MJDS[:2])
    return [PHOTOMETRY, double]

def test_batch_mixed_images():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(mixed_paths())
    assert list(model.sn_list['n_images']) == [3, 2]
    assert {n_img: list(data['sn_index']) for n_img, data in model.data.items()} == {2: [1], 3: [0]}
run_test('Batch: SNe grouped by image count', test_batch_mixed_images)

def test_batch_likelihood_invariance():
    import numpyro
    from numpyro.infer.util import log_density
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    # a shorter, shifted light curve, so the batch includes padding
    other = write_sn(example_table().iloc[::2].assign(mjd=lambda d: d['mjd'] + 5), 'other', z=0.5, ebv_mw=0.02,
                     peak_mjds=PEAK_MJDS)
    paths = [PHOTOMETRY, other]

    def model_fn():
        data = only_data(model)
        model.td_model(data, data['band_weights'], include_ml=False)

    def take_sn(params, i):
        # eps_tform has the SN axis first; every other site has it last
        return {k: v[i:i + 1] if k == 'eps_tform' else v[..., i:i + 1] for k, v in params.items()}

    model.process_dataset(paths)
    trace = numpyro.handlers.trace(numpyro.handlers.seed(model_fn, 1)).get_trace()
    params = {k: v['value'] for k, v in trace.items() if v['type'] == 'sample' and not v['is_observed']}
    batch_total = float(log_density(model_fn, (), {}, params)[0])
    total = 0
    for i, path in enumerate(paths):
        model = SEDmodel(load_model='G26x_model')  # model_fn uses whichever model is current
        model.process_dataset(path)
        total += float(log_density(model_fn, (), {}, take_sn(params, i))[0])
    assert np.isclose(batch_total, total, rtol=1e-10), f'batch {batch_total} != sum of singles {total}'
run_test('Batch: log density of a batch equals the sum over its SNe', test_batch_likelihood_invariance)


# ══════════════════════════════════════════════════════════════
# Group C: CLI
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
    import yaml
    from bayesn_td.cli import load_config
    config = {
        'load_model': 'T21_model', 'num_samples': 100, 'num_warmup': 100,
        'num_chains': 2, 'chain_method': 'sequential', 'init_strategy': 'median',
        'include_eps': True, 'include_ml': False, 'outputdir': tmp_path('cli'),
        'data': {
            'photometry': PHOTOMETRY, 'image_col': 'image', 'time_col': 'mjd',
            'band_col': 'filter', 'flux_col': 'flux', 'fluxerr_col': 'fluxerr',
            'peak_mjds': PEAK_MJDS, 'z': 0.5, 'ebv_mw': 0.02, 'map': {'g': 'g_LSST'},
        }
    }
    path = tmp_path('cli.yaml')
    with open(path, 'w') as f:
        yaml.dump(config, f)
    model_kwargs, data_kwargs, fit_kwargs = load_config(path, {'load_model': 'G26x_model', 'num_chains': None})
    assert model_kwargs == {'load_model': 'G26x_model'}
    assert data_kwargs['z'] == 0.5
    assert data_kwargs['filt_map'] == {'g': 'g_LSST'} and 'map' not in data_kwargs
    assert fit_kwargs['num_samples'] == 100
    assert fit_kwargs['num_chains'] == 2
    config['output'] = 'not_a_key'
    with open(path, 'w') as f:
        yaml.dump(config, f)
    raises(ValueError, load_config, path)  # unknown keys raise
run_test('Test 12: CLI config parsing and overrides', test_12)


# ══════════════════════════════════════════════════════════════
# Group D: MCMC Fitting
# ══════════════════════════════════════════════════════════════
print('\n=== Group D: MCMC Fitting ===')

def test_13():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='T21_model')
    model.process_dataset(**SIM_KWARGS)
    out = tmp_path('test_13', '')
    samples = model.fit(**FIT_KWARGS, outputdir=out, include_eps=False, include_ml=False)
    for key in ['theta', 'AV', 'tmax', 'Ds', 'delta_t', 'peak_mjd', 'mu', 'delM']:
        assert key in samples, f'Missing key {key}'
    assert 'RV' not in samples
    assert 'RV_tform' not in samples
    assert samples['theta'].shape == (1, 10, 1), f'theta shape {samples["theta"].shape}'
    assert samples['tmax'].shape == (1, 10, 3, 1), f'tmax shape {samples["tmax"].shape}'
    for f in ['chains.pkl', 'initial_chains.pkl', 'fit_summary.csv', 'sn_list.txt']:
        assert os.path.exists(os.path.join(out, f)), f'Missing output file {f}'
run_test('Test 13: Fit - fixed_RV, no eps, no ml, one chain', test_13)

def test_14():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)
    samples = model.fit(**FIT_KWARGS, outputdir=tmp_path('test_14', ''), include_eps=True, include_ml=False)
    assert 'RV' in samples
    assert 'RV_tform' in samples
    assert np.all(np.array(samples['RV']) > 1.2)
    assert samples['eps'].shape[-1] == 1 and samples['eps'].shape[:2] == (1, 10)
run_test('Test 14: Fit - pop_RV, with eps, no ml', test_14)

def test_15():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)
    samples = model.fit(**FIT_KWARGS, outputdir=tmp_path('test_15', ''), include_eps=True, include_ml=True)
    for key in ['A', 'beta_t']:
        assert key in samples, f'Missing ML key {key}'
    dt = samples['delta_t']
    assert dt.shape[2] == 2  # 3 images - 1
run_test('Test 15: Fit - pop_RV, with eps, with ml', test_15)

def test_16():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)
    samples = model.fit(**FIT_KWARGS, outputdir=tmp_path('test_16', ''), include_eps=False, include_ml=False)
    assert 'RV' in samples
    assert 'theta' in samples
    assert 'delta_t' in samples
run_test('Test 16: Fit - pop_RV, no eps, no ml', test_16)

def test_17():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='T21_model')
    model.process_dataset(**SIM_KWARGS)
    samples = model.fit(**{**FIT_KWARGS, 'num_chains': 2, 'chain_method': 'parallel'},
                        outputdir=tmp_path('test_17', ''), include_eps=False, include_ml=False)
    theta = np.array(samples['theta'])
    assert theta.shape == (2, 10, 1), f'Expected (2, 10, 1), got {theta.shape}'
run_test('Test 17: Fit - multi-chain parallel', test_17)

def test_18():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='T21_model')
    samples = model.fit_lensed_sn(**SIM_KWARGS, **FIT_KWARGS, include_eps=False, include_ml=False,
                                  outputdir=tmp_path('test_18', ''))
    for key in ['theta', 'AV', 'delta_t']:
        assert key in samples, f'Missing key {key}'
run_test('Test 18: fit_lensed_sn convenience', test_18)

def test_19():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(**SIM_KWARGS)
    samples = model.fit(**{**FIT_KWARGS, 'init_strategy': 'sample'}, outputdir=tmp_path('test_19', ''),
                        include_eps=False, include_ml=False)
    assert 'theta' in samples
    assert 'delta_t' in samples
run_test('Test 19: Fit - init_strategy=sample', test_19)

def test_fit_mixed_batch():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    model.process_dataset(mixed_paths())
    samples = model.fit(**FIT_KWARGS, outputdir=tmp_path('mixed', ''), include_eps=False, include_ml=False)
    assert samples['tmax'].shape == (1, 10, 3, 2)
    assert samples['delta_t'].shape == (1, 10, 2, 2)
    assert not np.isnan(samples['tmax'][..., 0]).any(), 'triple must have all images'
    assert np.isnan(samples['tmax'][:, :, 2, 1]).all(), 'missing image of the double must be NaN'
    assert np.isnan(samples['delta_t'][:, :, 1, 1]).all()
    assert not np.isnan(samples['delta_t'][:, :, 0, 1]).any()
run_test('Fit: mixed doubles and triples, NaN-padded outputs', test_fit_mixed_batch)

def test_simulate_roundtrip():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='G26x_model')
    peaks = np.array([[60000.0, 60030.0]] * 3)
    model.simulate_light_curve_ml(mjds=np.arange(59980, 60100, 5.0), N=3, bands=['g_LSST', 'r_LSST', 'i_LSST'],
                                  peak_mjds=peaks, z=0.5, yerr=0.03, mu='z', ebv_mw=0.02,
                                  save_to=tmp_path('roundtrip', 'sim.ecsv'))
    model.process_dataset(tmp_path('roundtrip', 'sim_*.ecsv'))
    assert list(model.sn_list['SNID']) == ['sim_0', 'sim_1', 'sim_2']
    samples = model.fit(**FIT_KWARGS, outputdir=tmp_path('roundtrip_fit', ''), include_eps=False, include_ml=False)
    assert samples['delta_t'].shape == (1, 10, 1, 3)
    assert np.all(np.isfinite(samples['delta_t']))
run_test('Fit: simulate N=3 -> glob -> fit round trip', test_simulate_roundtrip)


# ══════════════════════════════════════════════════════════════
# Group E: Simulation
# ══════════════════════════════════════════════════════════════
print('\n=== Group E: Simulation ===')

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
# Group F: Output Validation
# ══════════════════════════════════════════════════════════════
print('\n=== Group F: Output Validation ===')

def test_25():
    import pickle, pandas as pd
    out = tmp_path('test_13', '')
    with open(os.path.join(out, 'chains.pkl'), 'rb') as f:
        chains = pickle.load(f)
    for key in ['theta', 'AV', 'tmax', 'Ds', 'delta_t', 'peak_mjd', 'mu', 'delM', 'diverging']:
        assert key in chains, f'Missing key {key} in chains'
    summary = pd.read_csv(os.path.join(out, 'fit_summary.csv'))
    assert len(summary) > 0
    sn_list = pd.read_csv(os.path.join(out, 'sn_list.txt'))
    for col in ['SNID', 'n_images', 'z', 'peak_mjd_0']:
        assert col in sn_list.columns, f'Missing column {col} in sn_list.txt'
run_test('Test 25: Verify output file contents', test_25)

def test_26():
    import pickle
    out = tmp_path('test_13', '')
    with open(os.path.join(out, 'chains.pkl'), 'rb') as f:
        chains = pickle.load(f)
    with open(os.path.join(out, 'initial_chains.pkl'), 'rb') as f:
        initial = pickle.load(f)
    for key in ['delta_t', 'peak_mjd', 'mu', 'delM']:
        assert key in chains, f'{key} missing from chains'
        assert key not in initial, f'{key} should not be in initial_chains'
run_test('Test 26: Verify initial_chains vs chains', test_26)


# ══════════════════════════════════════════════════════════════
# Group G: Model State
# ══════════════════════════════════════════════════════════════
print('\n=== Group G: Model State ===')

def test_model_state_unchanged():
    from bayesn_td import SEDmodel
    model = SEDmodel(load_model='T21_model')
    data_attributes = {'data', 'sn_list', 'dropped', 'used_band_inds', 'used_band_dict', 'zps', 'offsets'}
    before = {k: v for k, v in vars(model).items() if k not in data_attributes}
    copies = {k: np.array(v) for k, v in before.items() if isinstance(v, np.ndarray)}
    model.process_dataset(**SIM_KWARGS)
    model.fit(**FIT_KWARGS, outputdir=tmp_path('state', ''), include_eps=False, include_ml=False)
    loaded_band = model.inv_band_dict[model.used_band_inds[1]]
    model.simulate_light_curve_ml(mjds=np.linspace(59990, 60040, 5), N=1, bands=[loaded_band],
                                  peak_mjds=np.array([[60010.0]]), z=0.5, mu='z')
    after = {k: v for k, v in vars(model).items() if k not in data_attributes}
    assert after.keys() == before.keys(), f'attributes added/removed: {after.keys() ^ before.keys()}'
    for k in before:
        assert after[k] is before[k], f'attribute {k} was reassigned'
    for k, v in copies.items():
        assert np.array_equal(v, after[k]), f'array attribute {k} was modified in place'
run_test('Model state: only the loaded data and band table change across process_dataset, fit and simulate',
         test_model_state_unchanged)


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
shutil.rmtree(TMP, ignore_errors=True)
print(f'\n{"=" * 50}')
print(f'Results: {passed} passed, {failed} failed out of {passed + failed}')
if failed == 0:
    print('All tests passed!')
else:
    print(f'FAILED tests: {", ".join(errors)}')
    sys.exit(1)
