#!/usr/bin/env python
"""
Command-line interface for bayesn-td.

Usage:
    run_bayesn_td input.yaml [--outputdir DIR] [--load_model MODEL] ...
"""

import argparse
import inspect
import os

from ruamel.yaml import YAML

from .model import SEDmodel

yaml = YAML(typ='safe')

# Top-level YAML keys passed to the SEDmodel constructor
MODEL_KEYS = {'load_model': 'load_model', 'filters': 'filter_yaml'}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='bayesn-td: BayeSN for lensed SNe Ia')
    parser.add_argument('input', type=str, help='Path to YAML input file')
    parser.add_argument('--filters', type=str, help='Path to filter YAML file')
    parser.add_argument('--outputdir', type=str, help='Output directory for results')
    parser.add_argument('--load_model', type=str, help='BayeSN model name or path to BAYESN.YAML')
    parser.add_argument('--num_chains', type=int, help='Number of MCMC chains')
    parser.add_argument('--num_warmup', type=int, help='Number of warmup steps')
    parser.add_argument('--num_samples', type=int, help='Number of posterior samples per chain')
    parser.add_argument('--chain_method', type=str, choices=['parallel', 'sequential', 'vectorized'],
                        help='Method for distributing chains')
    parser.add_argument('--init_strategy', type=str, choices=['median', 'sample'],
                        help='Initialisation strategy for HMC chains')
    parser.add_argument('--include_eps', type=lambda x: str(x).lower() == 'true',
                        help='Include epsilon in model')
    parser.add_argument('--include_ml', type=lambda x: str(x).lower() == 'true',
                        help='Include microlensing GP in model')
    return parser.parse_args(argv)


def load_config(path, overrides=None):
    """
    Read a YAML input file and apply command-line overrides.

    Parameters
    ----------
    path : str
        Path to the YAML input file.
    overrides : dict, optional
        Top-level keys to override; None values are ignored.

    Returns
    -------
    model_kwargs : dict
        Arguments for the ``SEDmodel`` constructor.
    data_kwargs : dict
        Arguments for ``SEDmodel.process_dataset`` (the ``data`` block).
    fit_kwargs : dict
        Arguments for ``SEDmodel.fit``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'Specified input file ({path}) was not found. '
                                f'Please provide the path to an input YAML file.')
    with open(path, 'r') as file:
        config = yaml.load(file) or {}
    config.update({key: val for key, val in (overrides or {}).items() if val is not None})

    data_kwargs = config.pop('data', None)
    if not isinstance(data_kwargs, dict):
        raise ValueError(f'{path} needs a "data" block giving at least the photometry')
    if 'map' in data_kwargs:  # BayeSN's YAML key for the filter mapping
        data_kwargs['filt_map'] = data_kwargs.pop('map')
    model_kwargs = {MODEL_KEYS[key]: config.pop(key) for key in list(config) if key in MODEL_KEYS}
    fit_params = inspect.signature(SEDmodel.fit).parameters
    unknown = sorted(key for key in config if key not in fit_params or key == 'self')
    if unknown:
        raise ValueError(f'Unknown keys in {path}: {unknown}. Valid top-level keys are "data", '
                         f'{sorted(MODEL_KEYS)} and the arguments of SEDmodel.fit: '
                         f'{[p for p in fit_params if p != "self"]}')
    return model_kwargs, data_kwargs, config


def main(argv=None):
    args = vars(parse_args(argv))
    model_kwargs, data_kwargs, fit_kwargs = load_config(args.pop('input'), args)
    model = SEDmodel(**model_kwargs)
    model.process_dataset(**data_kwargs)
    print('Running bayesn-td fitting...')
    model.fit(**fit_kwargs)
    print('Done.')


if __name__ == '__main__':
    main()
