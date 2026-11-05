"""Explicit, fail-fast inputs for historical cache-dependent experiments."""
from pathlib import Path


def add_cache_arguments(parser):
    parser.add_argument('--reference-run-dir', type=Path, help='Completed compare_models run folder, not its parent directory')
    parser.add_argument('--housing-run-dir', type=Path, help='Completed train_hdb_attributes run folder containing resale_features.pkl')
    parser.add_argument('--output-dir', type=Path, help='Fresh isolated output directory; do not overwrite historical evidence')


def require_files(folder, names, option='--reference-run-dir'):
    folder = Path(folder).resolve()
    missing = [name for name in names if not (folder/name).is_file()]
    if missing:
        commands = {
            '--classifier-run-dir': 'python -m src.research.check_low_rent_information --reference-run-dir <reference folder> --housing-run-dir <housing folder>',
            '--stock-run-dir': 'python -m src.research.submit_stock_probe',
            '--selected-run-dir': 'python -m src.research.probe_rmse_structure --reference-run-dir <reference folder> --candidates market_normalized',
            '--housing-run-dir': 'python -m src.research.train_hdb_attributes --candidates resale --reference-run-dir <reference folder>',
        }
        command = commands.get(option, 'python -m src.research.compare_models --candidates reference --windows <required windows>')
        raise FileNotFoundError(
            f'Incomplete cache at {folder}. Missing: {", ".join(missing)}. '
            f'Pass {option} with the actual completed run folder. '
            f'Prerequisite command: {command}. '
            'Run IDs change with code/environment; '
            'do not rename folders to historical IDs. Only load trusted local pickle files.')
    return folder


def reference_run(args, default, windows=('short_recent',)):
    names = []
    for window in windows:
        names += [f'{window}_scoring_rows.csv', f'{window}_local_trend_0.pkl',
                  f'{window}_reference_ensemble.npy']
        names += [f'{window}_reference_{view}_{mode}_42.npy'
                  for view in ['local_trend_0','hierarchical_6'] for mode in ['direct','residual']]
    return require_files(args.reference_run_dir or default, names)


def housing_run(args, default):
    return require_files(args.housing_run_dir or default, ['resale_features.pkl'], '--housing-run-dir')
