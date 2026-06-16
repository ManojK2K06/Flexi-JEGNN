# Geometry-Aware GNN Benchmarks for Molecular Property Prediction

A unified codebase for three independent graph-neural-network experiments on
molecular data. Each experiment studies how the _quality of pairwise distance
information_ fed to a message-passing network affects predictive performance,
by sweeping a set of "distance approximation levels" from purely topological
estimates up to true 3D geometry.

The three experiments are fully self-contained and never share data, models, or
dataset codes:

| Experiment     | Task                               | Output file                  |
| -------------- | ---------------------------------- | ---------------------------- |
| Classification | Binary molecular property          | `classification_results.csv` |
| PDBbind        | Protein–ligand binding affinity    | `PDBbind_results.csv`        |
| QM9            | Continuous quantum property (HOMO) | `qm9_raw_seeds.csv`          |

## Repository layout

```
unified_experiment.py        # orchestrator / CLI entry point
experiments/
    __init__.py
    classification.py         # logic from threshold_experiment-v4.py
    pdbbind.py                # logic from pdbbind-optimized.py
    qm9.py                    # logic from unified_continuous_experiment.py
requirements.txt
README.md
datasets/                     # you provide (see below)
refined-set/                  # you provide (PDBbind only)
```

## Installation

```bash
pip install -r requirements.txt
```

RDKit and PyTorch Geometric are required. A CUDA GPU is used automatically when
available; everything also runs on CPU (slowly).

## Input data

All dataset information is read **only** from a folder called `datasets/`
(CSV files), with the single exception of PDBbind, which is read from a folder
called `refined-set/`.

### `datasets/` — classification and QM9

Place these CSV files in `datasets/`:

| File        | SMILES column | Target column | Notes                        |
| ----------- | ------------- | ------------- | ---------------------------- |
| `BACE.csv`  | `mol`         | `Class`       | 0/1 label                    |
| `HIV.csv`   | `smiles`      | `HIV_active`  | 0/1 label                    |
| `BBBP.csv`  | `smiles`      | `p_np`        | 0/1 label                    |
| `ADMET.csv` | `smiles`      | `NR-AR`       | 0/1 label                    |
| `QM9.csv`   | `smiles`      | `homo`        | continuous regression target |

Column names are matched case-insensitively and a few common aliases
(`smiles`, `canonical_smiles`, `label`, `activity`, …) are accepted, but the
names above are the defaults.

### `refined-set/` — PDBbind

Use the standard PDBbind refined-set directory layout:

```
refined-set/
    index/
        INDEX_refined_data.2020      # any INDEX_*_data.* file works
    1a1e/
        1a1e_pocket.pdb
        1a1e_ligand.sdf              # or 1a1e_ligand.mol2
    1a4k/
        ...
```

The index file is parsed for the PDB id, the binding affinity (`pKd`, column 4),
and the ligand SMILES (column 5). Complexes missing a pocket or ligand file are
skipped.

## Running

Run everything:

```bash
python unified_experiment.py --task all
```

Run a single experiment:

```bash
python unified_experiment.py --task classification
python unified_experiment.py --task pdbbind
python unified_experiment.py --task qm9
```

Useful options:

```
--datasets-dir DIR        folder with classification/QM9 CSVs (default: datasets)
--refined-set-dir DIR     PDBbind folder (default: refined-set)
--classification-out PATH output CSV path (default: classification_results.csv)
--pdbbind-out PATH        output CSV path (default: PDBbind_results.csv)
--qm9-out PATH            output CSV path (default: qm9_raw_seeds.csv)
--models NAME [NAME ...]  run only a subset of models
--seeds N [N ...]         run only a subset of seeds
```

Each experiment writes its rows incrementally and flushes after every run, so
partial results survive an interruption.

## Models and distance levels

Models are selected per experiment:

- Classification & QM9: PharmaJEGNN, D-MPNN, GIN, SchNet, DimeNet, Uni-Mol
- PDBbind: the same six plus AttentiveFP

D-MPNN and GIN are pure 2D-topology models. They are run **only** at levels 0–2
and are intentionally skipped at the 3D levels (3 and 4), because feeding them
3D-derived inputs would not change their computation and would make the
comparison misleading. All other (geometry-aware) models run at every level.

Distance approximation levels:

| Level | Meaning                                     |
| ----- | ------------------------------------------- |
| 0     | constant per-hop distance (topological)     |
| 1     | bond-length path sum (topological)          |
| 2     | distance-bounds matrix estimate             |
| 3     | true 3D geometry (ETKDGv3 + MMFF embedding) |
| 4     | random distances (negative control)         |

Every (dataset × model × level) combination is repeated across 20 fixed seeds.

## Output formats

All three files contain **one row per seed** (raw, un-aggregated results).

`classification_results.csv`

```
key,auc,auprc,accuracy,precision_,recall,f1,mcc,brier,specificity,
threshold,train_time,ms_per_mol,n_params,epochs_run,stopped_early
```

Key format: `{dataset}_{model}_{level}_{seed}` (e.g. `BACE_PharmaJEGNN_0_42`).

`PDBbind_results.csv`

```
key,pearson_r,rmse,mae,train_time,ms_per_mol,n_params,epochs_run,stopped_early
```

Key format: `PDBbind_{model}_L{level}_s{seed}` (e.g. `PDBbind_AttentiveFP_L3_s42`).

`qm9_raw_seeds.csv`

```
key,pearson_r,mae,rmse,train_time,ms_per_mol,n_params,epochs_run,
stopped_early,dataset,model,level_id,seed
```

Key format: `QM9_{model}_{level}_{seed}` (e.g. `QM9_D-MPNN_0_1024`). The last
four columns are parsed back out of the key for convenience.

## Training configuration

Shared across experiments: 80 epochs max, Adam (lr 1e-4, weight decay 1e-5),
cosine-annealing schedule, gradient clipping, scaffold-based train/val/test
split (per-seed), and early stopping on the validation metric (AUC for
classification, RMSE for the regression tasks). Featurization uses an 18-d atom
feature vector and Gaussian-smeared edge distances; PDBbind adds an
`is_protein` atom flag and a cross-edge flag for the protein–ligand interface.
