"""
Photometry I/O for bayesn-td.

Readers turn a photometry source into one dictionary per SN, with keys:

- ``name``: SN name, unique within a batch.
- ``phot``: DataFrame with columns ``image``, ``time``, ``band``, plus
  ``mag``/``magerr`` and/or ``flux``/``fluxerr`` (flux on the ZP=27.5 FLUXCAL
  system).
- ``time_format``: ``'mjd'`` if ``time`` is on an observer-frame clock,
  ``'phase'`` if it is rest-frame phase relative to each image's peak.
- ``z``, ``z_cmb``, ``ebv_mw``: heliocentric redshift, CMB-frame redshift and
  Milky Way E(B-V), or None if the source does not state them.
- ``peak_mjds``: ``{image_label: peak}`` on the same clock as ``time``, or None.

Readers report what the source states and nothing more: they never apply a
cosmology and make no analysis choices (cuts, error floors, band coverage),
which are left to ``SEDmodel.process_dataset``.
"""

import glob
import os

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

ZPT = 27.5  # zero point of the FLUXCAL flux system used throughout bayesn-td


def _sn_dict(name, phot, time_format='mjd', z=None, z_cmb=None, ebv_mw=None, peak_mjds=None):
    """Build the per-SN dictionary returned by the readers."""
    return {'name': name, 'phot': phot, 'time_format': time_format, 'z': z, 'z_cmb': z_cmb, 'ebv_mw': ebv_mw,
            'peak_mjds': peak_mjds}


# Column name patterns for auto-detection (case-insensitive)
_COL_PATTERNS = {
    'time':    ['mjd', 'time', 'jd', 'phase'],
    'band':    ['filter', 'band', 'flt'],
    'image':   ['image', 'img'],
    'mag':     ['mag', 'magnitude'],
    'magerr':  ['magerr', 'mag_err'],
    'flux':    ['flux', 'fluxcal'],
    'fluxerr': ['fluxerr', 'fluxcalerr', 'flux_err'],
    'zp':      ['zp'],
}
_DEFAULT_COLS = ['mjd', 'filter', 'mag', 'magerr', 'image']

# SNANA header keys: observed values by default, simulated truths on request
_SNANA_KEYS = {
    False: {'z': 'ZHEL', 'z_cmb': 'ZFIN', 'ebv_mw': 'MWEBV', 'peak': 'PEAKMJD'},
    True: {'z': 'SZHEL', 'z_cmb': 'SZCMB', 'ebv_mw': 'SMWEBV', 'peak': 'SPEAKMJD'},
}

_LSST_PICKLE_COLS = ['obs_times', 'obs_bands', 'obs_mag_micro', 'mag_micro_error', 'time_delay', 'z_source']


def read_photometry(source, format='auto', **options):
    """
    Read one or more lensed SNe from a photometry source.

    Parameters
    ----------
    source : str, os.PathLike, pd.DataFrame or list
        A file, a directory (all files in it, sorted), a glob pattern, a
        DataFrame (named ``'SN'``), or a list of files, directories and glob
        patterns. SNe read from files are named after the file.
    format : str, optional
        ``'auto'`` (default) detects the format from file content. Otherwise one
        of ``'table'``, ``'fits'`` or ``'lsst_lensed_pickle'``. DataFrames are
        read as ``'table'`` unless a format is given.
    **options
        Passed to the reader for the format; see ``read_table``, ``read_fits``
        and ``read_lsst_lensed_pickle``. Options a reader does not accept raise.

    Returns
    -------
    list of dict
        One dictionary per SN; see the module docstring for its keys.
    """
    readers = {'table': read_table, 'fits': read_fits, 'lsst_lensed_pickle': read_lsst_lensed_pickle}
    if format != 'auto' and format not in readers:
        raise ValueError(f'Unknown format {format!r}; must be "auto" or one of {list(readers)}')

    records = []
    for item in _expand_source(source):
        fmt = format
        if fmt == 'auto':
            fmt = 'table' if isinstance(item, pd.DataFrame) else _detect_format(item)
        if fmt == 'lsst_lensed_pickle':
            records.extend(read_lsst_lensed_pickle(item, **options))
        else:
            records.append(readers[fmt](item, **options))
    return records


def _expand_source(source):
    """Expand a source into a list of file paths or DataFrames."""
    if isinstance(source, pd.DataFrame):
        return [source]
    if isinstance(source, (list, tuple)):
        return [entry for item in source for entry in _expand_source(item)]
    path = os.fspath(source)
    if os.path.isdir(path):
        paths = sorted(os.path.join(path, f) for f in os.listdir(path)
                       if not f.startswith('.') and os.path.isfile(os.path.join(path, f)))
        if not paths:
            raise FileNotFoundError(f'Directory {path} contains no files')
    elif os.path.exists(path):
        paths = [path]
    elif glob.has_magic(path):
        paths = sorted(glob.glob(path))
        if not paths:
            raise FileNotFoundError(f'No files match {path}')
    else:
        raise FileNotFoundError(f'Photometry source {path} does not exist')
    return paths


def _detect_format(path):
    """Detect a file's format from its content."""
    with open(path, 'rb') as f:
        head = f.read(8)
    if head[:1] == b'\x80':
        return 'lsst_lensed_pickle'
    if head.startswith(b'SIMPLE'):
        return 'fits'
    return 'table'


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _detect_column(columns, role, override=None):
    """Return the column name for a given role, or None if not found."""
    if override is not None:
        return override
    lower_map = {c.lower(): c for c in columns}
    for pat in _COL_PATTERNS[role]:
        if pat in lower_map:
            return lower_map[pat]
    return None


def _standardise(df, image_col=None, time_col=None, band_col=None, mag_col=None, magerr_col=None,
                 flux_col=None, fluxerr_col=None):
    """
    Map a raw photometry table onto the standard ``phot`` columns (see the module docstring),
    rescaling flux to ZP=27.5 if the table has a ``zp`` column.
    """
    cols = df.columns
    found = {
        'image': _detect_column(cols, 'image', image_col),
        'time': _detect_column(cols, 'time', time_col),
        'band': _detect_column(cols, 'band', band_col),
        'mag': _detect_column(cols, 'mag', mag_col),
        'magerr': _detect_column(cols, 'magerr', magerr_col),
        'flux': _detect_column(cols, 'flux', flux_col),
        'fluxerr': _detect_column(cols, 'fluxerr', fluxerr_col),
        'zp': _detect_column(cols, 'zp'),
    }
    for role in ['image', 'time', 'band']:
        if found[role] is None:
            raise ValueError(f'Could not detect {role} column. Available columns: {list(cols)}. '
                             f'Pass {role}_col explicitly.')
    has_mag = found['mag'] is not None and found['magerr'] is not None
    has_flux = found['flux'] is not None and found['fluxerr'] is not None
    if not has_mag and not has_flux:
        raise ValueError('Could not detect photometry columns. Need either mag+magerr or flux+fluxerr '
                         f'columns. Available columns: {list(cols)}.')

    phot = pd.DataFrame({
        'image': df[found['image']].astype(str).values,
        'time': df[found['time']].astype(float).values,
        'band': df[found['band']].astype(str).values,
    })
    if has_mag:
        phot['mag'] = df[found['mag']].astype(float).values
        phot['magerr'] = df[found['magerr']].astype(float).values
    if has_flux:
        scale = 1.0
        if found['zp'] is not None:
            scale = np.power(10, 0.4 * (ZPT - df[found['zp']].astype(float).values))
        phot['flux'] = df[found['flux']].astype(float).values * scale
        phot['fluxerr'] = df[found['fluxerr']].astype(float).values * scale
    return phot


def _peaks_by_label(peak_list, phot):
    """Map a list of peaks, ordered by sorted image label, onto those labels."""
    labels = sorted(phot['image'].unique())
    if len(peak_list) != len(labels):
        raise ValueError(f'peak_mjds has length {len(peak_list)} but the data has images {labels}')
    return dict(zip(labels, (float(p) for p in peak_list)))


def read_text_table(path, dtype=None):
    """
    Read a text table (CSV, whitespace-delimited or ECSV), detecting the
    delimiter from the content. Returns the table and its ECSV metadata (empty
    for other formats). ``dtype`` is passed to ``pandas.read_csv`` for non-ECSV
    files.
    """
    with open(path, 'r') as f:
        first = f.readline()
        if first.startswith('# %ECSV'):
            table = Table.read(path, format='ascii.ecsv')
            return table.to_pandas(), dict(table.meta)
        header = first
        while header and (header.startswith('#') or not header.strip()):
            header = f.readline()
    if not header:
        raise ValueError(f'{path} contains no table')
    sep = ',' if ',' in header else r'\s+'
    return pd.read_csv(path, sep=sep, comment='#', dtype=dtype), {}


def read_table(source, time_format='mjd', image_col=None, time_col=None, band_col=None,
               mag_col=None, magerr_col=None, flux_col=None, fluxerr_col=None):
    """
    Read one lensed SN from a text table (CSV, whitespace-delimited or ECSV) or
    a DataFrame.

    The delimiter is detected from the file content. Column names are
    auto-detected (case-insensitive) unless given via ``image_col``,
    ``time_col``, ``band_col``, ``mag_col``, ``magerr_col``, ``flux_col`` or
    ``fluxerr_col``. A DataFrame with integer column labels is treated as
    headerless, with columns ``mjd, filter, mag, magerr, image``. ``z``,
    ``ebv_mw`` and ``peak_mjds`` are read from an ECSV header if present.

    Parameters
    ----------
    source : str or pd.DataFrame
        File path, or a DataFrame (the SN is then named ``'SN'``).
    time_format : str, optional
        ``'mjd'`` (default) or ``'phase'``.
    image_col, time_col, band_col, mag_col, magerr_col, flux_col, fluxerr_col : str, optional
        Column name overrides.
    """
    columns = dict(image_col=image_col, time_col=time_col, band_col=band_col,
                   mag_col=mag_col, magerr_col=magerr_col, flux_col=flux_col, fluxerr_col=fluxerr_col)
    if isinstance(source, pd.DataFrame):
        df, meta, name = source.copy(), {}, 'SN'
    else:
        (df, meta), name = read_text_table(source), _stem(source)

    if all(isinstance(c, (int, np.integer)) for c in df.columns):
        print(f'read_table: no header detected. Assuming column order: {_DEFAULT_COLS}')
        df.columns = _DEFAULT_COLS[:len(df.columns)]

    phot = _standardise(df, **columns)
    peak_mjds = _peaks_by_label(meta['peak_mjds'], phot) if 'peak_mjds' in meta else None
    return _sn_dict(name, phot, time_format, z=meta.get('z'), ebv_mw=meta.get('ebv_mw'), peak_mjds=peak_mjds)


def _header_value(header, key):
    """Return a FITS header value as a float, or None if the key is absent."""
    return float(header[key]) if key in header else None


def read_fits(path, time_format='mjd', true_values=False, image_col=None, time_col=None, band_col=None,
              mag_col=None, magerr_col=None, flux_col=None, fluxerr_col=None):
    """
    Read one lensed SN from a FITS file.

    The layout is chosen by structure:

    - **One table HDU** holding all images: ``z`` and ``ebv_mw`` are read from
      the ``REDSHIFT`` and ``EBV_MW`` header keys.
    - **One table HDU per image** (SNANA): ``z``, ``z_cmb``, ``ebv_mw`` and each
      image's peak are read from the observed header keys ``ZHEL``, ``ZFIN``,
      ``MWEBV`` and ``PEAKMJD``, or from the simulated truths ``SZHEL``,
      ``SZCMB``, ``SMWEBV`` and ``SPEAKMJD`` if ``true_values=True``. Values
      for the whole SN must agree across HDUs.

    Parameters
    ----------
    path : str
        FITS file path. The SN is named after the file.
    time_format : str, optional
        ``'mjd'`` (default) or ``'phase'``.
    true_values : bool, optional
        For SNANA files, read simulated true values instead of observed ones.
    image_col, time_col, band_col, mag_col, magerr_col, flux_col, fluxerr_col : str, optional
        Column name overrides.
    """
    columns = dict(image_col=image_col, time_col=time_col, band_col=band_col,
                   mag_col=mag_col, magerr_col=magerr_col, flux_col=flux_col, fluxerr_col=fluxerr_col)
    name = _stem(path)
    with fits.open(path) as hdul:
        tables = [hdu for hdu in hdul if isinstance(hdu, fits.BinTableHDU)]
        if not tables:
            raise ValueError(f'{path} contains no FITS table')

        if len(tables) == 1:
            if true_values:
                raise ValueError(f'{path} has a single table, not one per image (SNANA); true_values does not '
                                 f'apply')
            header = dict(tables[0].header)
            df = Table(tables[0].data).to_pandas()
            return _sn_dict(name, _standardise(df, **columns), time_format, z=_header_value(header, 'REDSHIFT'),
                            ebv_mw=_header_value(header, 'EBV_MW'))

        keys = _SNANA_KEYS[bool(true_values)]
        frames, peak_mjds, per_sn = [], {}, []
        for i, hdu in enumerate(tables):
            df = Table(hdu.data).to_pandas()
            label = f'image_{i + 1}'
            hdu_image_col = _detect_column(df.columns, 'image', image_col)
            if hdu_image_col is not None:
                labels = df[hdu_image_col].astype(str).unique()
                if len(labels) != 1:
                    raise ValueError(f'{path} HDU {i + 1} holds several images: {list(labels)}')
                label = labels[0]
            if label in peak_mjds:
                raise ValueError(f'{path}: image {label} appears in more than one HDU')
            df = df.assign(**{hdu_image_col or 'image': label})
            frames.append(df)
            for key in keys.values():
                if key not in hdu.header:
                    raise ValueError(f'{path} HDU {i + 1} is missing header key {key}')
            peak_mjds[label] = float(hdu.header[keys['peak']])
            per_sn.append(tuple(float(hdu.header[keys[k]]) for k in ['z', 'z_cmb', 'ebv_mw']))
        if len(set(per_sn)) != 1:
            raise ValueError(f'{path}: {keys["z"]}/{keys["z_cmb"]}/{keys["ebv_mw"]} differ between HDUs')
        z, z_cmb, ebv_mw = per_sn[0]
        phot = _standardise(pd.concat(frames, ignore_index=True), **columns)
        return _sn_dict(name, phot, time_format, z=z, z_cmb=z_cmb, ebv_mw=ebv_mw, peak_mjds=peak_mjds)


def read_lsst_lensed_pickle(source):
    """
    Read lensed SNe from a simulated LSST lensed-SN catalogue (a pickled
    DataFrame with one row per lensed system).

    Each row gives one SN, named by its row index. Photometry is taken from
    ``obs_times``, ``obs_bands`` and the microlensed magnitudes
    ``obs_mag_micro``/``mag_micro_error`` (shape ``(n_obs, n_images)``). Each image's peak is
    ``time_delay[image]`` on the ``obs_times`` clock, the redshift is
    ``z_source``, and single-letter bands map to ``<band>_LSST``.

    Parameters
    ----------
    source : str or pd.DataFrame
        Pickle file path, or the catalogue DataFrame (e.g. a subset of rows).
    """
    catalogue = source if isinstance(source, pd.DataFrame) else pd.read_pickle(source)
    missing = [c for c in _LSST_PICKLE_COLS if c not in catalogue.columns]
    if missing:
        raise ValueError(f'Not an LSST lensed-SN catalogue: missing columns {missing}')

    records = []
    for idx, row in catalogue.iterrows():
        n_images = row['obs_mag_micro'].shape[1]
        labels = [chr(ord('A') + i) for i in range(n_images)]
        frames = [pd.DataFrame({
            'image': label,
            'time': np.asarray(row['obs_times'], dtype=float),
            'band': [f'{b}_LSST' for b in row['obs_bands']],
            'mag': row['obs_mag_micro'][:, i].astype(float),
            'magerr': row['mag_micro_error'][:, i].astype(float),
        }) for i, label in enumerate(labels)]
        phot = pd.concat(frames, ignore_index=True)
        records.append(_sn_dict(str(idx), phot, z=float(row['z_source']),
                                peak_mjds={label: float(row['time_delay'][i]) for i, label in enumerate(labels)}))
    return records


def write_ecsv(path, mjds, bands, peak_mjds, z, ebv_mw, values, errors, mag, true_params):
    """
    Write simulated light curves to ECSV files, one per SN, with true
    parameters in the YAML header.

    Images are labelled ``'A'``, ``'B'``, ... in the same order as
    ``peak_mjds[n, :]``, which ``read_table`` recovers by sorting the labels.
    Only the simulated quantity (``mag``/``magerr`` or ``flux``/``fluxerr``)
    is written.

    Parameters
    ----------
    path : str
        Output path. For more than one SN, an index suffix is added
        (e.g. ``sim_0.ecsv``).
    mjds, bands : array-like
        Observation MJD and band of each observation, shape ``(n_obs,)``.
    peak_mjds : array-like
        Shape ``(N, n_images)``.
    z, ebv_mw : array-like
        Shape ``(N,)``.
    values, errors : array-like
        Simulated photometry and errors, shape ``(n_obs, n_images, N)``.
    mag : bool
        Whether ``values`` are magnitudes (else FLUXCAL fluxes).
    true_params : dict
        ``mu`` (shape ``(N, n_images)``), ``theta``, ``AV``, ``RV``, ``del_M``
        (shape ``(N,)``).
    """
    from . import __version__

    n_obs, n_images, N = values.shape
    labels = np.array([chr(ord('A') + i) for i in range(n_images)])
    value_col, error_col = ('mag', 'magerr') if mag else ('flux', 'fluxerr')

    root, ext = os.path.splitext(path)
    ext = ext or '.ecsv'
    for n in range(N):
        table = Table({
            'mjd': np.repeat(np.asarray(mjds, dtype=float), n_images),
            'filter': np.repeat(np.asarray(bands).astype(str), n_images),
            value_col: values[:, :, n].ravel(),
            error_col: errors[:, :, n].ravel(),
            'image': np.tile(labels, n_obs),
        })
        table.meta = {
            'z': float(z[n]),
            'ebv_mw': float(ebv_mw[n]),
            'peak_mjds': [float(v) for v in peak_mjds[n, :]],
            'true_mu': [float(v) for v in true_params['mu'][n, :]],
            'true_theta': float(true_params['theta'][n]),
            'true_AV': float(true_params['AV'][n]),
            'true_RV': float(true_params['RV'][n]),
            'true_del_M': float(true_params['del_M'][n]),
            'bayesn_td_version': __version__,
        }
        out = root + ext if N == 1 else f'{root}_{n}{ext}'
        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        table.write(out, format='ascii.ecsv', overwrite=True)
