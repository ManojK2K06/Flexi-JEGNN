import argparse
import sys


def _run_classification(args):
    from experiments import classification
    return classification.run(
        datasets_dir=args.datasets_dir,
        out_csv=args.classification_out,
        models=args.models,
        seeds=args.seeds,
    )


def _run_pdbbind(args):
    from experiments import pdbbind
    return pdbbind.run(
        refined_set_dir=args.refined_set_dir,
        out_csv=args.pdbbind_out,
        models=args.models,
        seeds=args.seeds,
    )


def _run_qm9(args):
    from experiments import qm9
    return qm9.run(
        datasets_dir=args.datasets_dir,
        out_csv=args.qm9_out,
        models=args.models,
        seeds=args.seeds,
    )


def build_parser():
    p = argparse.ArgumentParser(
        description='Run molecular GNN experiments (classification, PDBbind, QM9).'
    )
    p.add_argument('--task', choices=['all', 'classification', 'pdbbind', 'qm9'],
                   default='all', help='Which experiment to run.')
    p.add_argument('--datasets-dir', default='datasets',
                   help='Folder with classification/QM9 dataset CSV files.')
    p.add_argument('--refined-set-dir', default='refined-set',
                   help='PDBbind refined-set folder.')
    p.add_argument('--classification-out', default='classification_results.csv')
    p.add_argument('--pdbbind-out', default='PDBbind_results.csv')
    p.add_argument('--qm9-out', default='qm9_raw_seeds.csv')
    p.add_argument('--models', nargs='+', default=None,
                   help='Optional subset of model names to run.')
    p.add_argument('--seeds', nargs='+', type=int, default=None,
                   help='Optional subset of integer seeds to run.')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    written = []
    if args.task in ('all', 'classification'):
        written.append(_run_classification(args))
    if args.task in ('all', 'pdbbind'):
        written.append(_run_pdbbind(args))
    if args.task in ('all', 'qm9'):
        written.append(_run_qm9(args))

    print('\nDONE. Output files:')
    for path in written:
        print(f'  {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
