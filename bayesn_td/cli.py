#!/usr/bin/env python
"""
Command-line interface for bayesn-td.

Usage:
    run_bayesn_td input.yaml [--outputdir DIR] [--load_model MODEL] ...
"""

import os
import argparse
from ruamel.yaml import YAML

from .model import SEDmodel

yaml = YAML(typ='safe')


def main():
    parser = argparse.ArgumentParser(description='bayesn-td: BayeSN for lensed SNe Ia')
    parser.add_argument('input', type=str, help='Path to YAML input file')
    parser.add_argument('--filters', type=str, required=False,
                        help='Path to filter YAML file')
    parser.add_argument('--outputdir', type=str, required=False,
                        help='Output directory for results')
    parser.add_argument('--load_model', type=str, required=False,
                        help='BayeSN model name or path to BAYESN.YAML')
    parser.add_argument('--num_chains', type=int, required=False,
                        help='Number of MCMC chains')
    parser.add_argument('--num_warmup', type=int, required=False,
                        help='Number of warmup steps')
    parser.add_argument('--num_samples', type=int, required=False,
                        help='Number of posterior samples per chain')
    parser.add_argument('--chain_method', type=str, required=False,
                        choices=['parallel', 'sequential', 'vectorized'],
                        help='Method for distributing chains')
    parser.add_argument('--init_strategy', type=str, required=False,
                        choices=['median', 'sample'],
                        help='Initialisation strategy for HMC chains')
    parser.add_argument('--include_eps', type=lambda x: str(x).lower() == 'true',
                        required=False, help='Include epsilon in model')
    parser.add_argument('--include_ml', type=lambda x: str(x).lower() == 'true',
                        required=False, help='Include microlensing GP in model')
    cmd_args = parser.parse_args()

    if not os.path.exists(cmd_args.input):
        raise FileNotFoundError(
            f'Specified input file ({cmd_args.input}) was not found. '
            f'Please provide the path to an input YAML file.')
    with open(cmd_args.input, 'r') as file:
        args = yaml.load(file)

    if cmd_args.load_model is not None:
        args['load_model'] = cmd_args.load_model
    elif 'load_model' not in args:
        args['load_model'] = 'G26x_model'

    if cmd_args.filters is not None:
        args['filters'] = cmd_args.filters
    elif 'filters' not in args:
        args['filters'] = None

    model = SEDmodel(
        load_model=args['load_model'],
        filter_yaml=args.get('filters'),
    )
    model.run(args, cmd_args)


if __name__ == '__main__':
    main()
