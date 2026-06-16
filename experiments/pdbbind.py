"""
PDBbind protein-ligand binding-affinity regression.

Predicts pKd from the 3D protein-ligand complex graph. Ligand-internal
distances are approximated at five fidelity levels; protein-internal and
protein-ligand contacts always use the real 3D coordinates from the PDB
pocket file. Performance is measured across 20 seeds.

Source data: the PDBbind refined-set folder (refined-set/), containing one
sub-folder per complex with <pdbid>_pocket.pdb and <pdbid>_ligand.sdf|.mol2,
plus an index/INDEX_*_data.* file giving pKd values.
Output:      PDBbind_results.csv (one row per model/level/seed).
"""

import os
import csv
import math
import time
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import Linear, Sequential, ReLU, LayerNorm, Dropout
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.nn.aggr import MultiAggregation
from torch_geometric.utils import softmax as pyg_softmax, to_dense_batch

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 9999, 7777, 5555,
         1111, 2222, 3333, 4444, 6666, 8888, 314, 271, 1618, 1729]

EPOCHS = 80
BATCH_SIZE = 64

LEVELS = [0, 1, 2, 3, 4]
MODELS_2D = {'D-MPNN', 'GIN'}
LEVELS_3D_ONLY = {3, 4}

OUTPUT_COLUMNS = ['key', 'pearson_r', 'rmse', 'mae', 'train_time',
                  'ms_per_mol', 'n_params', 'epochs_run', 'stopped_early']

ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'I', 'B', 'other']
IN_DIM = len(ATOM_TYPES) + 6 + 1   # 19 (extra is_protein flag)
EDGE_DIM = 5 + 16 + 1              # 22 (5 bond + 16 distance Gaussians + 1 cross flag)

_BL = {
    ('C', 'C', 'SINGLE'): 1.540, ('C', 'C', 'DOUBLE'): 1.340, ('C', 'C', 'TRIPLE'): 1.204, ('C', 'C', 'AROMATIC'): 1.395,
    ('C', 'N', 'SINGLE'): 1.469, ('C', 'N', 'DOUBLE'): 1.279, ('C', 'N', 'TRIPLE'): 1.158, ('C', 'N', 'AROMATIC'): 1.340,
    ('C', 'O', 'SINGLE'): 1.432, ('C', 'O', 'DOUBLE'): 1.229, ('C', 'O', 'AROMATIC'): 1.360,
    ('C', 'S', 'SINGLE'): 1.820, ('C', 'F', 'SINGLE'): 1.350, ('C', 'Cl', 'SINGLE'): 1.767, ('C', 'Br', 'SINGLE'): 1.944,
    ('N', 'N', 'SINGLE'): 1.449, ('N', 'O', 'SINGLE'): 1.400,
}
_BL_FB = {'SINGLE': 1.500, 'DOUBLE': 1.320, 'TRIPLE': 1.200, 'AROMATIC': 1.380}


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=8.0, num_gaussians=16):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


distance_expansion = GaussianSmearing(start=0.0, stop=8.0, num_gaussians=16)


def parse_pocket_pdb(pdb_path):
    positions, symbols = [], []
    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ('ATOM', 'HETATM'):
                continue
            element = line[76:78].strip() if len(line) > 76 else ''
            if not element:
                atom_name = line[12:16].strip()
                element = ''.join(c for c in atom_name if c.isalpha())[:1]
            if element.upper() == 'H':
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            positions.append([x, y, z])
            symbols.append(element.capitalize())
    if not positions:
        return None, None
    return np.array(positions, dtype=np.float32), symbols


def load_ligand_mol(pdbid, data_root):
    d = Path(data_root) / pdbid
    sanitize_no_valence = (Chem.SanitizeFlags.SANITIZE_ALL ^
                           Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    for suffix in (f'{pdbid}_ligand.sdf', f'{pdbid}_ligand.mol2'):
        fpath = d / suffix
        if not fpath.exists():
            continue
        if suffix.endswith('.sdf'):
            suppl = Chem.SDMolSupplier(str(fpath), removeHs=True, sanitize=False)
            for mol in suppl:
                if mol is None:
                    continue
                try:
                    Chem.SanitizeMol(mol, sanitize_no_valence)
                except Exception:
                    pass
                if mol.GetNumConformers() > 0:
                    return mol
        else:
            mol = Chem.MolFromMol2File(str(fpath), removeHs=True, sanitize=False)
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol, sanitize_no_valence)
            except Exception:
                pass
            if mol.GetNumConformers() > 0:
                return mol
    return None


def _build_lig_dist_matrix(lig_mol, level, seed, n_lig, lig_pos):
    if level == 0:
        INF = 1e9
        D = np.full((n_lig, n_lig), INF, dtype=np.float32)
        np.fill_diagonal(D, 0.0)
        for bond in lig_mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            D[i, j] = D[j, i] = 1.4
        for k in range(n_lig):
            D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
        D[D >= INF] = 999.0
        return D
    elif level == 1:
        bond_type_str = {BT.SINGLE: 'SINGLE', BT.DOUBLE: 'DOUBLE',
                         BT.TRIPLE: 'TRIPLE', BT.AROMATIC: 'AROMATIC'}
        INF = 1e9
        B = np.full((n_lig, n_lig), INF, dtype=np.float32)
        np.fill_diagonal(B, 0.0)
        for bond in lig_mol.GetBonds():
            bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bts = bond_type_str.get(bond.GetBondType(), 'SINGLE')
            a1s = lig_mol.GetAtomWithIdx(bi).GetSymbol()
            a2s = lig_mol.GetAtomWithIdx(bj).GetSymbol()
            v = _BL.get((a1s, a2s, bts), _BL.get((a2s, a1s, bts), _BL_FB.get(bts, 1.5)))
            B[bi, bj] = B[bj, bi] = v
        for k in range(n_lig):
            B = np.minimum(B, B[:, k:k + 1] + B[k:k + 1, :])
        B[B >= INF] = 999.0
        return B
    elif level == 2:
        try:
            Bounds = AllChem.GetMoleculeBoundsMatrix(lig_mol)
        except Exception:
            Bounds = None
        if Bounds is None:
            return np.full((n_lig, n_lig), 999.0, dtype=np.float32)
        return ((Bounds + Bounds.T) / 2.0).astype(np.float32)
    elif level == 3:
        dd = lig_pos[:, None, :] - lig_pos[None, :, :]
        return np.sqrt((dd ** 2).sum(-1)).astype(np.float32)
    elif level == 4:
        rng = np.random.default_rng(seed)
        D = rng.uniform(1.0, 8.0, size=(n_lig, n_lig)).astype(np.float32)
        D = (D + D.T) / 2.0
        np.fill_diagonal(D, 0.0)
        return D
    return None


def build_complex_graph(pdbid, level, seed, data_root,
                        lig_tau=5.0, prot_tau=4.0, cross_tau=5.0):
    lig_mol = load_ligand_mol(pdbid, data_root)
    if lig_mol is None or lig_mol.GetNumAtoms() < 2:
        return None
    n_lig = lig_mol.GetNumAtoms()
    conf = lig_mol.GetConformer()
    lig_pos = np.array([[conf.GetAtomPosition(i).x,
                         conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z] for i in range(n_lig)], dtype=np.float32)

    pocket_path = Path(data_root) / pdbid / f'{pdbid}_pocket.pdb'
    if not pocket_path.exists():
        return None
    prot_pos, prot_symbols = parse_pocket_pdb(pocket_path)
    if prot_pos is None or len(prot_pos) < 1:
        return None

    cross_D = np.sqrt(((prot_pos[:, None, :] - lig_pos[None, :, :]) ** 2).sum(-1))
    min_cross = cross_D.min(axis=1)
    keep = min_cross <= cross_tau
    if keep.sum() < 1:
        return None
    prot_pos = prot_pos[keep]
    prot_symbols = [s for s, k in zip(prot_symbols, keep) if k]
    cross_D = cross_D[keep]
    n_prot = len(prot_pos)

    at = ATOM_TYPES

    def feat_rdkit(atom, is_prot):
        sym = atom.GetSymbol()
        oh = [int(sym == a) for a in at[:-1]] + [int(sym not in at[:-1])]
        return oh + [atom.GetDegree() / 6.0, atom.GetFormalCharge() / 4.0,
                     atom.GetNumImplicitHs() / 4.0, int(atom.GetIsAromatic()),
                     int(atom.IsInRing()),
                     int(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
                     float(is_prot)]

    def feat_sym(sym, is_prot):
        oh = [int(sym == a) for a in at[:-1]] + [int(sym not in at[:-1])]
        return oh + [0.0, 0.0, 0.0, 0, 0, 0, float(is_prot)]

    x = ([feat_rdkit(lig_mol.GetAtomWithIdx(i), False) for i in range(n_lig)] +
         [feat_sym(prot_symbols[i], True) for i in range(n_prot)])

    bond_type_flags = {BT.SINGLE: [1, 0, 0, 0, 0], BT.DOUBLE: [0, 1, 0, 0, 0],
                       BT.TRIPLE: [0, 0, 1, 0, 0], BT.AROMATIC: [0, 0, 0, 1, 0]}
    bond_dict = {}
    for bond in lig_mol.GetBonds():
        bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        flags = bond_type_flags.get(bond.GetBondType(), [0, 0, 0, 0, 0])[:4] + [int(bond.IsInRing())]
        bond_dict[(bi, bj)] = flags
        bond_dict[(bj, bi)] = flags
    no_bond = [0.0, 0.0, 0.0, 0.0, 0.0]

    LIG_D = _build_lig_dist_matrix(lig_mol, level, seed, n_lig, lig_pos)
    if LIG_D is None:
        return None

    ii, jj = np.where((LIG_D < lig_tau) & (LIG_D > 0))
    edge_src = ii.tolist()
    edge_dst = jj.tolist()
    distances = LIG_D[ii, jj].tolist()
    bond_feats = [bond_dict.get((int(i), int(j)), no_bond) for i, j in zip(ii, jj)]
    cross_flags = [0.0] * len(edge_src)

    pd2 = np.sqrt(((prot_pos[:, None, :] - prot_pos[None, :, :]) ** 2).sum(-1))
    pi_i, pi_j = np.where((pd2 < prot_tau) & (pd2 > 0))
    edge_src.extend((pi_i + n_lig).tolist())
    edge_dst.extend((pi_j + n_lig).tolist())
    distances.extend(pd2[pi_i, pi_j].tolist())
    bond_feats.extend([no_bond] * len(pi_i))
    cross_flags.extend([0.0] * len(pi_i))

    cross_pi, cross_lj = np.where(cross_D < cross_tau)
    n_cross = len(cross_pi)
    if n_cross > 0:
        cross_d_vals = cross_D[cross_pi, cross_lj]
        edge_src.extend((cross_pi + n_lig).tolist())
        edge_dst.extend(cross_lj.tolist())
        distances.extend(cross_d_vals.tolist())
        bond_feats.extend([no_bond] * n_cross)
        cross_flags.extend([1.0] * n_cross)
        edge_src.extend(cross_lj.tolist())
        edge_dst.extend((cross_pi + n_lig).tolist())
        distances.extend(cross_d_vals.tolist())
        bond_feats.extend([no_bond] * n_cross)
        cross_flags.extend([1.0] * n_cross)

    if not edge_src:
        return None

    return {
        'x': np.array(x, dtype=np.float32),
        'edge_src': np.array(edge_src, dtype=np.int64),
        'edge_dst': np.array(edge_dst, dtype=np.int64),
        'bond_feats': np.array(bond_feats, dtype=np.float32),
        'distances': np.array(distances, dtype=np.float32),
        'cross_flags': np.array(cross_flags, dtype=np.float32),
        'n': n_lig + n_prot,
    }


def tensors_from_dict(t, label):
    x = torch.from_numpy(t['x'])
    ei = torch.stack([torch.from_numpy(t['edge_src']), torch.from_numpy(t['edge_dst'])], dim=0)
    bf = torch.from_numpy(t['bond_feats'])
    dists = torch.from_numpy(t['distances'])
    gauss = distance_expansion(dists)
    cf = torch.from_numpy(t['cross_flags']).unsqueeze(-1)
    ea = torch.cat([bf, gauss, cf], dim=-1)
    g = Data(x=x, edge_index=ei, edge_attr=ea, num_nodes=t['n'])
    g.y = torch.tensor([label], dtype=torch.float)
    return g


def featurize_complex(df, level, seed, data_root):
    graphs = []
    for row in df.itertuples(index=False):
        t = build_complex_graph(str(row.pdbid), level, seed, data_root)
        if t is None:
            continue
        graphs.append(tensors_from_dict(t, float(row.pKd)))
    return graphs


def _find_index_file(root):
    idx_dir = Path(root) / 'index'
    if not idx_dir.exists():
        return None
    candidates = (sorted(idx_dir.glob('INDEX_*_data.*')) +
                  sorted(idx_dir.glob('INDEX_*_PL_*')))
    return candidates[0] if candidates else None


def load_pdbbind(data_root):
    data_root = Path(data_root)
    idx_file = _find_index_file(data_root)
    if idx_file is None:
        raise RuntimeError(f"No PDBbind index file under {data_root}/index/. "
                           "Expected e.g. INDEX_refined_data.2020")
    print(f"  [PDBbind] index file: {idx_file.name}")
    rows = []
    with open(idx_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            pdbid = parts[0].lower()
            try:
                pkd = float(parts[3])
            except ValueError:
                continue
            smiles = parts[4] if len(parts) > 4 else ''
            rows.append({'pdbid': pdbid, 'pKd': pkd, 'smiles': smiles})

    df = pd.DataFrame(rows).dropna(subset=['pKd'])
    valid = []
    for row in df.itertuples(index=False):
        pid = row.pdbid
        d = data_root / pid
        has_pocket = (d / f'{pid}_pocket.pdb').exists()
        has_ligand = ((d / f'{pid}_ligand.sdf').exists() or (d / f'{pid}_ligand.mol2').exists())
        if has_pocket and has_ligand:
            valid.append({'pdbid': pid, 'pKd': row.pKd, 'smiles': row.smiles})

    df = pd.DataFrame(valid).reset_index(drop=True)
    print(f"  [PDBbind] {len(df)} complexes with pocket + ligand files")
    if df.empty:
        raise RuntimeError(f"No valid complexes found under {data_root}.")
    return df


def scaffold_split(df, seed, test_frac=0.1, val_frac=0.1):
    has_smiles = ('smiles' in df.columns and df['smiles'].str.len().gt(0).any())
    if has_smiles:
        groups = {}
        for i, row in enumerate(df.itertuples(index=False)):
            smi = str(row.smiles)
            try:
                mol = Chem.MolFromSmiles(smi)
                sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else smi
            except Exception:
                sc = smi
            groups.setdefault(sc, []).append(i)
    else:
        groups = {row.pdbid: [i] for i, row in enumerate(df.itertuples(index=False))}

    rng = np.random.default_rng(seed)
    order = sorted(groups.values(), key=len, reverse=True)
    rng.shuffle(order)
    flat = [i for grp in order for i in grp]

    n = len(flat)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    te_idx = sorted(flat[:n_test])
    va_idx = sorted(flat[n_test:n_test + n_val])
    tr_idx = sorted(flat[n_test + n_val:])
    return (df.iloc[tr_idx].reset_index(drop=True),
            df.iloc[va_idx].reset_index(drop=True),
            df.iloc[te_idx].reset_index(drop=True))


class JointEdgeConv(MessagePassing):
    def __init__(self, node_dim, hidden_dim):
        super().__init__(aggr='mean')
        self.msg_mlp = Sequential(
            Linear(node_dim * 2 + hidden_dim, hidden_dim * 2),
            LayerNorm(hidden_dim * 2), ReLU(), Linear(hidden_dim * 2, hidden_dim))
        self.upd_mlp = Sequential(
            Linear(node_dim + hidden_dim, hidden_dim),
            LayerNorm(hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.upd_mlp(torch.cat([x, aggr_out], dim=-1))


class PharmaJEGNN(nn.Module):
    def __init__(self, node_dim=IN_DIM, edge_dim=EDGE_DIM, hidden_dim=256, num_layers=5, dropout=0.3):
        super().__init__()
        self.node_emb = Linear(node_dim, hidden_dim)
        self.edge_emb = Sequential(Linear(edge_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.convs = nn.ModuleList([JointEdgeConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.readout = MultiAggregation(['sum', 'mean', 'max'], mode='cat')
        self.mlp = Sequential(
            Linear(hidden_dim * 3, hidden_dim * 2), LayerNorm(hidden_dim * 2),
            ReLU(), Dropout(dropout), Linear(hidden_dim * 2, hidden_dim), ReLU(), Linear(hidden_dim, 1))

    def forward(self, data):
        x = self.node_emb(data.x)
        edge_attr = self.edge_emb(data.edge_attr)
        for conv in self.convs:
            x = x + conv(x, data.edge_index, edge_attr)
        return self.mlp(self.readout(x, data.batch)).squeeze(-1)


class DMPNNConv(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr='add')
        self.W_msg = Linear(hidden_dim + 5, hidden_dim)
        self.W_upd = Linear(hidden_dim, hidden_dim)
        self.act = ReLU()

    def forward(self, x, edge_index, edge_attr):
        ea = edge_attr[:, :5] if edge_attr.size(1) > 5 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=ea)

    def message(self, x_j, edge_attr):
        return self.act(self.W_msg(torch.cat([x_j, edge_attr], dim=-1)))

    def update(self, aggr_out):
        return self.act(self.W_upd(aggr_out))


class DMPNN(nn.Module):
    def __init__(self, node_dim=IN_DIM, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.input_proj = Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([DMPNNConv(hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.mlp = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Dropout(dropout), Linear(hidden_dim, 1))

    def forward(self, data):
        x = self.input_proj(data.x)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x + conv(x, data.edge_index, data.edge_attr))
        return self.mlp(global_mean_pool(x, data.batch)).squeeze(-1)


class GINModel(nn.Module):
    def __init__(self, node_dim=IN_DIM, hidden_dim=256, num_layers=5, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import GINConv
        self.input_proj = Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GINConv(Sequential(Linear(hidden_dim, hidden_dim * 2), ReLU(),
                               Linear(hidden_dim * 2, hidden_dim)))
            for _ in range(num_layers)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.mlp = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Dropout(dropout), Linear(hidden_dim, 1))

    def forward(self, data):
        x = self.input_proj(data.x)
        for conv, bn in zip(self.convs, self.bn):
            x = F.relu(bn(conv(x, data.edge_index)))
        return self.mlp(global_add_pool(x, data.batch)).squeeze(-1)


class ShiftedSoftplus(nn.Module):
    def forward(self, x):
        return F.softplus(x) - math.log(2)


class SchNetLayer(MessagePassing):
    def __init__(self, hidden_dim, num_rbf=16):
        super().__init__(aggr='add')
        self.rbf_proj = Linear(num_rbf, hidden_dim)
        self.W = Sequential(Linear(hidden_dim, hidden_dim), ShiftedSoftplus(), Linear(hidden_dim, hidden_dim))
        self.upd = Sequential(Linear(hidden_dim, hidden_dim), ShiftedSoftplus(), Linear(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        rbf = edge_attr[:, 5:21] if edge_attr.size(1) > 5 else edge_attr[:, :16]
        return self.propagate(edge_index, x=x, rbf=rbf)

    def message(self, x_j, rbf):
        return x_j * self.W(self.rbf_proj(rbf))

    def update(self, aggr_out):
        return self.upd(aggr_out)


class SchNet(nn.Module):
    def __init__(self, node_dim=IN_DIM, hidden_dim=256, num_layers=6, dropout=0.3):
        super().__init__()
        self.emb = Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([SchNetLayer(hidden_dim) for _ in range(num_layers)])
        self.mlp = Sequential(Linear(hidden_dim, hidden_dim // 2), ShiftedSoftplus(),
                              Dropout(dropout), Linear(hidden_dim // 2, 1))

    def forward(self, data):
        x = self.emb(data.x)
        for conv in self.convs:
            x = x + conv(x, data.edge_index, data.edge_attr)
        return self.mlp(global_add_pool(x, data.batch)).squeeze(-1)


class DimeNetBlock(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr='add')
        self.rbf_proj = Linear(16, hidden_dim)
        self.msg_linear = Linear(hidden_dim * 2, hidden_dim)
        self.upd_linear = Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        rbf = edge_attr[:, 5:21] if edge_attr.size(1) > 5 else edge_attr[:, :16]
        return self.propagate(edge_index, x=x, rbf=rbf)

    def message(self, x_i, x_j, rbf):
        return F.silu(self.msg_linear(torch.cat([x_j * self.rbf_proj(rbf), x_i], dim=-1)))

    def update(self, aggr_out):
        return F.silu(self.upd_linear(aggr_out))


class DimeNet(nn.Module):
    def __init__(self, node_dim=IN_DIM, hidden_dim=256, num_layers=4, dropout=0.3):
        super().__init__()
        self.emb = Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        self.convs = nn.ModuleList([DimeNetBlock(hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.mlp = Sequential(Linear(hidden_dim, hidden_dim), nn.SiLU(), Dropout(dropout), Linear(hidden_dim, 1))

    def forward(self, data):
        x = self.emb(data.x)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x + conv(x, data.edge_index, data.edge_attr))
        return self.mlp(global_mean_pool(x, data.batch)).squeeze(-1)


class UniMolLite(nn.Module):
    def __init__(self, node_dim=IN_DIM, hidden_dim=256, num_heads=8, num_layers=6, dropout=0.3):
        super().__init__()
        self.input_proj = Linear(node_dim, hidden_dim)
        self.pair_proj = Linear(16, num_heads)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                       dim_feedforward=hidden_dim * 2,
                                       dropout=dropout, batch_first=True, norm_first=True),
            num_layers=num_layers)
        self.mlp = Sequential(Linear(hidden_dim, hidden_dim // 2), ReLU(), Dropout(dropout), Linear(hidden_dim // 2, 1))

    def forward(self, data):
        x, mask = to_dense_batch(self.input_proj(data.x), data.batch)
        B, N, _ = x.shape
        rbf = data.edge_attr[:, 5:21] if data.edge_attr.size(1) > 5 else data.edge_attr[:, :16]
        pbf = self.pair_proj(rbf)
        nh = pbf.size(-1)
        ab = torch.zeros(B, N, N, nh, device=x.device, dtype=torch.float32)
        bs = torch.bincount(data.batch, minlength=B)
        off = torch.zeros(B + 1, dtype=torch.long, device=x.device)
        off[1:] = bs.cumsum(0)
        sg, dg = data.edge_index[0], data.edge_index[1]
        bi = data.batch[sg]
        sl = sg - off[bi]
        dl = dg - off[bi]
        vld = (sl < N) & (dl < N)
        ab[bi[vld], sl[vld], dl[vld]] = pbf[vld].float()
        ab = ab.permute(0, 3, 1, 2).reshape(B * nh, N, N)
        out = self.transformer(x, mask=ab, src_key_padding_mask=~mask)
        out = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.mlp(out).squeeze(-1)


class AttentiveFPAtomUpdate(MessagePassing):
    def __init__(self, hidden_dim, edge_dim, dropout=0.3):
        super().__init__(aggr='add')
        self.align = Sequential(Linear(hidden_dim * 2 + edge_dim, hidden_dim),
                                nn.LeakyReLU(0.2), Dropout(dropout), Linear(hidden_dim, 1))
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        alpha = self.align(torch.cat([x_i, x_j, edge_attr], dim=-1))
        alpha = pyg_softmax(alpha, index, ptr, size_i)
        return self.dropout(alpha) * x_j

    def update(self, aggr_out, x):
        return self.gru(aggr_out, x)


class AttentiveFPModel(nn.Module):
    def __init__(self, node_dim=IN_DIM, edge_dim=EDGE_DIM, hidden_dim=256, num_layers=5, dropout=0.3):
        super().__init__()
        self.input_proj = Sequential(Linear(node_dim, hidden_dim), ReLU())
        self.atom_layers = nn.ModuleList([
            AttentiveFPAtomUpdate(hidden_dim, edge_dim, dropout) for _ in range(num_layers)])
        self.graph_align = Sequential(Linear(hidden_dim, hidden_dim), nn.Tanh(), Linear(hidden_dim, 1))
        self.graph_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.graph_drop = Dropout(dropout)
        self.mlp = Sequential(Linear(hidden_dim, hidden_dim // 2), ReLU(), Dropout(dropout), Linear(hidden_dim // 2, 1))

    def forward(self, data):
        x, ei, ea, batch = self.input_proj(data.x), data.edge_index, data.edge_attr, data.batch
        for layer in self.atom_layers:
            x = x + layer(x, ei, ea)
        alpha = pyg_softmax(self.graph_align(x), batch)
        context = global_add_pool(self.graph_drop(alpha) * x, batch)
        h = self.graph_gru(context, global_mean_pool(x, batch))
        return self.mlp(h).squeeze(-1)


MODEL_REGISTRY = {
    'PharmaJEGNN': lambda: PharmaJEGNN(node_dim=IN_DIM, edge_dim=EDGE_DIM, hidden_dim=256, num_layers=5),
    'D-MPNN':      lambda: DMPNN(node_dim=IN_DIM, hidden_dim=256, num_layers=3),
    'GIN':         lambda: GINModel(node_dim=IN_DIM, hidden_dim=256, num_layers=5),
    'SchNet':      lambda: SchNet(node_dim=IN_DIM, hidden_dim=256, num_layers=6),
    'DimeNet':     lambda: DimeNet(node_dim=IN_DIM, hidden_dim=256, num_layers=4),
    'Uni-Mol':     lambda: UniMolLite(node_dim=IN_DIM, hidden_dim=256, num_heads=8, num_layers=6),
    'AttentiveFP': lambda: AttentiveFPModel(node_dim=IN_DIM, edge_dim=EDGE_DIM, hidden_dim=256, num_layers=5),
}


def _collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return Batch.from_data_list(batch) if batch else None


def evaluate_regression(model, loader):
    model.eval()
    all_preds, all_labs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            all_preds.append(model(batch).view(-1))
            all_labs.append(batch.y.view(-1))
    preds = torch.cat(all_preds).cpu().numpy().astype(np.float32)
    labs = torch.cat(all_labs).cpu().numpy().astype(np.float32)
    metrics = {'pearson_r': float('nan'), 'rmse': float('nan'), 'mae': float('nan')}
    try:
        metrics['rmse'] = float(np.sqrt(mean_squared_error(labs, preds)))
        metrics['mae'] = float(mean_absolute_error(labs, preds))
        if np.std(preds) > 0 and np.std(labs) > 0:
            pr, _ = pearsonr(labs, preds)
            metrics['pearson_r'] = float(pr)
        else:
            metrics['pearson_r'] = 0.0
    except Exception:
        pass
    return metrics


def train_one_epoch(model, loader, opt, crit):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(batch).view(-1), batch.y.float().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        opt.step()


def run_training(model, tr_loader, va_loader, epochs=EPOCHS, lr=1e-4, patience=15, min_epochs=30):
    crit = nn.MSELoss()
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best_rmse = float('inf')
    best_w = copy.deepcopy(model.state_dict())
    no_improve = 0
    stopped_early = False
    ep = 0
    for ep in range(1, epochs + 1):
        train_one_epoch(model, tr_loader, opt, crit)
        val_rmse = evaluate_regression(model, va_loader)['rmse']
        sched.step()
        if not math.isnan(val_rmse) and val_rmse < best_rmse - 1e-4:
            best_rmse = val_rmse
            best_w = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if ep >= min_epochs and no_improve >= patience:
            stopped_early = True
            break
    model.load_state_dict(best_w)
    return model, ep, stopped_early


def run_experiment(df, level_id, seed, model_name, data_root):
    tr, va, te = scaffold_split(df, seed)
    t0 = time.time()
    tr_g = featurize_complex(tr, level_id, seed, data_root)
    va_g = featurize_complex(va, level_id, seed, data_root)
    te_g = featurize_complex(te, level_id, seed, data_root)
    feat_time = time.time() - t0

    if len(tr_g) < 16:
        return None

    pin = torch.cuda.is_available()
    tl = DataLoader(tr_g, BATCH_SIZE, shuffle=True, pin_memory=pin, collate_fn=_collate_skip_none)
    vl = DataLoader(va_g, BATCH_SIZE, shuffle=False, pin_memory=pin, collate_fn=_collate_skip_none)
    el = DataLoader(te_g, BATCH_SIZE, shuffle=False, pin_memory=pin, collate_fn=_collate_skip_none)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MODEL_REGISTRY[model_name]().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    t1 = time.time()
    model, ep, stopped = run_training(model, tl, vl, epochs=EPOCHS, patience=15, min_epochs=30)
    metrics = evaluate_regression(model, el)
    metrics['ms_per_mol'] = feat_time * 1000 / max(len(tr_g) + len(va_g) + len(te_g), 1)
    metrics['train_time'] = time.time() - t1
    metrics['n_params'] = n_params
    metrics['epochs_run'] = ep
    metrics['stopped_early'] = int(stopped)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _row_from_metrics(key, m):
    return {
        'key': key,
        'pearson_r': m.get('pearson_r'),
        'rmse': m.get('rmse'),
        'mae': m.get('mae'),
        'train_time': m.get('train_time'),
        'ms_per_mol': m.get('ms_per_mol'),
        'n_params': m.get('n_params'),
        'epochs_run': m.get('epochs_run'),
        'stopped_early': m.get('stopped_early'),
    }


def run(refined_set_dir='refined-set', out_csv='PDBbind_results.csv',
        models=None, seeds=None):
    models = models or list(MODEL_REGISTRY.keys())
    seeds = seeds or SEEDS

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, 'w', newline='')
    writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
    writer.writeheader()

    print(f"\n{'=' * 60}\nPDBBIND REGRESSION  (device={DEVICE})\n{'=' * 60}")
    df = load_pdbbind(refined_set_dir)
    for level_id in LEVELS:
        for model_name in models:
            if model_name in MODELS_2D and level_id in LEVELS_3D_ONLY:
                continue
            for seed in seeds:
                key = f"PDBbind_{model_name}_L{level_id}_s{seed}"
                print(f"  {key}")
                m = run_experiment(df, level_id, seed, model_name, refined_set_dir)
                if m is None:
                    continue
                writer.writerow(_row_from_metrics(key, m))
                f.flush()
    f.close()
    print(f"\n[pdbbind] results written -> {out_path}")
    return str(out_path)
