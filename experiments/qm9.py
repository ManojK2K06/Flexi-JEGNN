"""
QM9 continuous-property regression.

Predicts a continuous quantum property (HOMO) from the molecular graph.
Atom-pair distances are approximated at five fidelity levels and fed to one
of six GNNs. Performance is measured across 20 seeds.

Source data: read from datasets/QM9.csv (SMILES + continuous target).
Output:      qm9_raw_seeds.csv (one row per model/level/seed, with the key
             parsed back into dataset/model/level_id/seed columns).
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, GINConv
from torch_geometric.nn.aggr import MultiAggregation
from torch_geometric.utils import to_dense_batch

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 9999, 7777, 5555,
         1111, 2222, 3333, 4444, 6666, 8888, 314, 271, 1618, 1729]

EPOCHS = 80
TAU = 5.0
DATASET_NAME = 'QM9'
SMILES_COL = 'smiles'
LABEL_COL = 'homo'

LEVELS = [0, 1, 2, 3, 4]
MODELS_2D = {'D-MPNN', 'GIN'}
LEVELS_3D_ONLY = {3, 4}

OUTPUT_COLUMNS = ['key', 'pearson_r', 'mae', 'rmse', 'train_time', 'ms_per_mol',
                  'n_params', 'epochs_run', 'stopped_early',
                  'dataset', 'model', 'level_id', 'seed']

ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'I', 'B', 'other']
IN_DIM = len(ATOM_TYPES) + 6   # 18
EDGE_DIM = 5 + 16              # 21

_BL = {
    ('C', 'C', 'SINGLE'): 1.540, ('C', 'C', 'DOUBLE'): 1.340, ('C', 'C', 'TRIPLE'): 1.204, ('C', 'C', 'AROMATIC'): 1.395,
    ('C', 'N', 'SINGLE'): 1.469, ('C', 'N', 'DOUBLE'): 1.279, ('C', 'N', 'TRIPLE'): 1.158, ('C', 'N', 'AROMATIC'): 1.340,
    ('C', 'O', 'SINGLE'): 1.432, ('C', 'O', 'DOUBLE'): 1.229, ('C', 'O', 'AROMATIC'): 1.360,
    ('C', 'S', 'SINGLE'): 1.820, ('C', 'F', 'SINGLE'): 1.350, ('C', 'Cl', 'SINGLE'): 1.767, ('C', 'Br', 'SINGLE'): 1.944,
    ('N', 'N', 'SINGLE'): 1.449, ('N', 'O', 'SINGLE'): 1.400,
}
_BL_FB = {'SINGLE': 1.500, 'DOUBLE': 1.320, 'TRIPLE': 1.200, 'AROMATIC': 1.380}


def _bl(a1, a2, bt):
    return _BL.get((a1, a2, bt), _BL.get((a2, a1, bt), _BL_FB.get(bt, 1.500)))


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=16):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


distance_expansion = GaussianSmearing(start=0.0, stop=5.0, num_gaussians=16)


def _atom_feats(atom):
    at = atom.GetSymbol()
    oh = [int(at == a) for a in ATOM_TYPES[:-1]] + [int(at not in ATOM_TYPES[:-1])]
    return oh + [
        atom.GetDegree() / 6.0,
        atom.GetFormalCharge() / 4.0,
        atom.GetNumImplicitHs() / 4.0,
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        int(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
    ]


def _build_graph_tensors(smiles, level, seed=42, tau=TAU):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    n = mol.GetNumAtoms()
    x = [_atom_feats(a) for a in mol.GetAtoms()]

    if level == 0:
        def dist_fn(i, j):
            path = rdmolops.GetShortestPath(mol, i, j)
            return (len(path) - 1) * 1.4 if len(path) >= 2 else 999.0
    elif level == 1:
        B = np.zeros((n, n), dtype=np.float32)
        for bond in mol.GetBonds():
            bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bts = {BT.SINGLE: 'SINGLE', BT.DOUBLE: 'DOUBLE', BT.TRIPLE: 'TRIPLE',
                   BT.AROMATIC: 'AROMATIC'}.get(bond.GetBondType(), 'SINGLE')
            a1s = mol.GetAtomWithIdx(bi).GetSymbol()
            a2s = mol.GetAtomWithIdx(bj).GetSymbol()
            B[bi][bj] = B[bj][bi] = _bl(a1s, a2s, bts)

        def dist_fn(i, j):
            try:
                path = rdmolops.GetShortestPath(mol, i, j)
                if len(path) < 2:
                    return 999.0
                return sum(float(B[path[s]][path[s + 1]]) or 1.5 for s in range(len(path) - 1))
            except Exception:
                return 999.0
    elif level == 2:
        try:
            Bounds = AllChem.GetMoleculeBoundsMatrix(mol)
        except Exception:
            Bounds = None

        def dist_fn(i, j):
            if Bounds is None:
                return 999.0
            idx1, idx2 = min(i, j), max(i, j)
            return float((Bounds[idx1, idx2] + Bounds[idx2, idx1]) / 2.0)
    elif level == 3:
        mol_h = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) == -1:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
        conf = mol_h.GetConformer()
        pos = np.array([[conf.GetAtomPosition(i).x,
                         conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z] for i in range(n)],
                       dtype=np.float32)
        diff = pos[:, None, :] - pos[None, :, :]
        D3 = np.sqrt((diff ** 2).sum(axis=-1))

        def dist_fn(i, j):
            return float(D3[i][j])
    elif level == 4:
        def dist_fn(i, j):
            lo, hi = min(i, j), max(i, j)
            rng = np.random.default_rng(seed + lo * 1000 + hi)
            return float(rng.uniform(1.0, 8.0))
    else:
        return None

    edge_src, edge_dst, bond_feats_list, distances = [], [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                d = dist_fn(i, j)
            except Exception:
                d = 999.0
            if d >= tau:
                continue
            edge_src.append(i)
            edge_dst.append(j)
            distances.append(d)
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                bt = bond.GetBondType()
                bond_feats_list.append([
                    float(bt == BT.SINGLE), float(bt == BT.DOUBLE),
                    float(bt == BT.TRIPLE), float(bt == BT.AROMATIC),
                    float(bond.IsInRing()),
                ])
            else:
                bond_feats_list.append([0.0, 0.0, 0.0, 0.0, 0.0])

    if not edge_src:
        return None

    return {'x': x, 'edge_src': edge_src, 'edge_dst': edge_dst,
            'bond_feats': bond_feats_list, 'distances': distances, 'n': n}


def _tensors_to_data(t, label):
    x = torch.tensor(t['x'], dtype=torch.float)
    ei = torch.tensor([t['edge_src'], t['edge_dst']], dtype=torch.long)
    dists = torch.tensor(t['distances'], dtype=torch.float)
    gauss = distance_expansion(dists)
    bond_t = torch.tensor(t['bond_feats'], dtype=torch.float)
    ea = torch.cat([bond_t, gauss], dim=-1)
    g = Data(x=x, edge_index=ei, edge_attr=ea, num_nodes=t['n'])
    g.y = torch.tensor([float(label)], dtype=torch.float)
    return g


def featurize(df, smiles_col, label_col, level, seed):
    smiles_list = df[smiles_col].tolist()
    labels_list = df[label_col].tolist()
    graphs = []
    for smi, label in zip(smiles_list, labels_list):
        t = _build_graph_tensors(str(smi), level, seed)
        if t is None:
            continue
        graphs.append(_tensors_to_data(t, label))
    return graphs


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
            ReLU(), Dropout(dropout), Linear(hidden_dim * 2, hidden_dim),
            ReLU(), Linear(hidden_dim, 1))

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
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                                   dim_feedforward=hidden_dim * 2,
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = Sequential(Linear(hidden_dim, hidden_dim // 2), ReLU(), Dropout(dropout), Linear(hidden_dim // 2, 1))

    def forward(self, data):
        x = self.input_proj(data.x)
        x_dense, mask = to_dense_batch(x, data.batch)
        B, N, H = x_dense.shape
        rbf = data.edge_attr[:, 5:21] if data.edge_attr.size(1) > 5 else data.edge_attr[:, :16]
        pair_bias_flat = self.pair_proj(rbf)
        num_heads = pair_bias_flat.size(-1)
        attn_bias = torch.zeros(B, N, N, num_heads, device=x.device)
        batch_size_per_graph = torch.bincount(data.batch, minlength=B)
        offsets = torch.zeros(B + 1, dtype=torch.long, device=x.device)
        offsets[1:] = batch_size_per_graph.cumsum(0)
        src_glob, dst_glob = data.edge_index[0], data.edge_index[1]
        b_idx = data.batch[src_glob]
        src_loc = src_glob - offsets[b_idx]
        dst_loc = dst_glob - offsets[b_idx]
        valid = (src_loc < N) & (dst_loc < N)
        attn_bias[b_idx[valid], src_loc[valid], dst_loc[valid]] = pair_bias_flat[valid]
        attn_bias = attn_bias.permute(0, 3, 1, 2).reshape(B * num_heads, N, N)
        out = self.transformer(x_dense, mask=attn_bias, src_key_padding_mask=~mask)
        out = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.mlp(out).squeeze(-1)


MODEL_REGISTRY = {
    'PharmaJEGNN': lambda: PharmaJEGNN(node_dim=IN_DIM, edge_dim=EDGE_DIM, hidden_dim=256, num_layers=5),
    'D-MPNN':      lambda: DMPNN(node_dim=IN_DIM, hidden_dim=256, num_layers=3),
    'GIN':         lambda: GINModel(node_dim=IN_DIM, hidden_dim=256, num_layers=5),
    'SchNet':      lambda: SchNet(node_dim=IN_DIM, hidden_dim=256, num_layers=6),
    'DimeNet':     lambda: DimeNet(node_dim=IN_DIM, hidden_dim=256, num_layers=4),
    'Uni-Mol':     lambda: UniMolLite(node_dim=IN_DIM, hidden_dim=256, num_heads=8, num_layers=6),
}


def evaluate(model, loader):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch).view(-1).cpu().numpy()
            preds.extend(out.tolist())
            labs.extend(batch.y.view(-1).cpu().numpy().tolist())
    preds = np.array(preds, dtype=np.float32)
    labs = np.array(labs, dtype=np.float32)
    metrics = {'pearson_r': float('nan'), 'rmse': float('nan'), 'mae': float('nan')}
    try:
        metrics['rmse'] = float(np.sqrt(mean_squared_error(labs, preds)))
        metrics['mae'] = float(mean_absolute_error(labs, preds))
        if np.std(preds) == 0 or np.std(labs) == 0:
            metrics['pearson_r'] = 0.0
        else:
            pr, _ = pearsonr(labs, preds)
            metrics['pearson_r'] = float(pr)
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
        val_rmse = evaluate(model, va_loader)['rmse']
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


def scaffold_split(df, smiles_col, label_col, seed, test_frac=0.1, val_frac=0.1):
    scaffolds = {}
    for i, smi in enumerate(df[smiles_col]):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            sc = str(smi)
        scaffolds.setdefault(sc, []).append(i)

    rng = np.random.default_rng(seed)
    order = sorted(scaffolds.values(), key=len, reverse=True)
    rng.shuffle(order)
    flat = [i for grp in order for i in grp]

    n = len(flat)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    te_idx = sorted(flat[:n_test])
    va_idx = sorted(flat[n_test:n_test + n_val])
    tr_idx = sorted(flat[n_test + n_val:])

    tr = df.iloc[tr_idx].reset_index(drop=True)
    va = df.iloc[va_idx].reset_index(drop=True)
    te = df.iloc[te_idx].reset_index(drop=True)
    for split in (tr, va, te):
        split[label_col] = pd.to_numeric(split[label_col], errors='coerce')
    return tr, va, te


def _resolve_columns(fpath, wanted_smiles, wanted_label):
    header = pd.read_csv(fpath, nrows=0)
    header.columns = [c.strip() for c in header.columns]
    cols = list(header.columns)
    col_lower = {c.lower(): c for c in cols}

    if wanted_smiles in cols:
        sc = wanted_smiles
    else:
        for alias in ('smiles', 'mol', 'canonical_smiles'):
            if alias in col_lower:
                sc = col_lower[alias]
                break
        else:
            raise ValueError(f"Cannot find smiles column '{wanted_smiles}' in {fpath.name}. "
                             f"Columns: {cols[:10]}")
    if wanted_label in cols:
        lc = wanted_label
    else:
        for alias in ('homo', 'label', 'y', 'target'):
            if alias in col_lower:
                lc = col_lower[alias]
                break
        else:
            raise ValueError(f"Cannot find label column '{wanted_label}' in {fpath.name}. "
                             f"Columns: {cols[:10]}")
    return sc, lc


def load_dataset(datasets_dir):
    fpath = Path(datasets_dir) / f'{DATASET_NAME}.csv'
    if not fpath.exists():
        raise FileNotFoundError(f"Expected dataset file not found: {fpath}")
    sc, lc = _resolve_columns(fpath, SMILES_COL, LABEL_COL)
    df = pd.read_csv(fpath, usecols=[sc, lc])
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=[sc]).copy()
    df[lc] = pd.to_numeric(df[lc], errors='coerce')
    df = df.dropna(subset=[lc])
    if len(df) < 50:
        raise RuntimeError(f"QM9 has only {len(df)} usable rows after cleaning.")
    print(f"  [QM9] loaded {len(df)} rows (smiles='{sc}', target='{lc}')")
    return df, sc, lc


def run_experiment(df, sc, lc, level_id, seed, model_name):
    tr, va, te = scaffold_split(df, sc, lc, seed)
    t_feat0 = time.time()
    tr_g = featurize(tr, sc, lc, level_id, seed)
    va_g = featurize(va, sc, lc, level_id, seed)
    te_g = featurize(te, sc, lc, level_id, seed)
    feat_time = time.time() - t_feat0
    n_mols = len(tr_g) + len(va_g) + len(te_g)
    ms_per_mol = (feat_time * 1000) / max(n_mols, 1)

    if len(tr_g) < 32:
        return None

    pin = torch.cuda.is_available()
    tl = DataLoader(tr_g, 64, shuffle=True, pin_memory=pin)
    vl = DataLoader(va_g, 64, shuffle=False, pin_memory=pin)
    el = DataLoader(te_g, 64, shuffle=False, pin_memory=pin)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MODEL_REGISTRY[model_name]().to(DEVICE)

    t_train0 = time.time()
    model, epochs_run, stopped_early = run_training(model, tl, vl, epochs=EPOCHS, lr=1e-4,
                                                    patience=15, min_epochs=30)
    train_time = time.time() - t_train0

    metrics = evaluate(model, el)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics['ms_per_mol'] = ms_per_mol
    metrics['train_time'] = train_time
    metrics['n_params'] = n_params
    metrics['epochs_run'] = epochs_run
    metrics['stopped_early'] = int(stopped_early)
    return metrics


def _row_from_metrics(key, m):
    parts = key.split('_')
    dataset = parts[0]
    seed = parts[-1]
    level_id = parts[-2]
    model = '_'.join(parts[1:-2])
    return {
        'key': key,
        'pearson_r': m.get('pearson_r'),
        'mae': m.get('mae'),
        'rmse': m.get('rmse'),
        'train_time': m.get('train_time'),
        'ms_per_mol': m.get('ms_per_mol'),
        'n_params': m.get('n_params'),
        'epochs_run': m.get('epochs_run'),
        'stopped_early': m.get('stopped_early'),
        'dataset': dataset,
        'model': model,
        'level_id': level_id,
        'seed': seed,
    }


def run(datasets_dir='datasets', out_csv='qm9_raw_seeds.csv', models=None, seeds=None):
    models = models or list(MODEL_REGISTRY.keys())
    seeds = seeds or SEEDS

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, 'w', newline='')
    writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
    writer.writeheader()

    print(f"\n{'=' * 60}\nQM9 REGRESSION  (device={DEVICE})\n{'=' * 60}")
    df, sc, lc = load_dataset(datasets_dir)
    for level_id in LEVELS:
        for model_name in models:
            if model_name in MODELS_2D and level_id in LEVELS_3D_ONLY:
                continue
            for seed in seeds:
                key = f"{DATASET_NAME}_{model_name}_{level_id}_{seed}"
                print(f"  {key}")
                m = run_experiment(df, sc, lc, level_id, seed, model_name)
                if m is None:
                    continue
                writer.writerow(_row_from_metrics(key, m))
                f.flush()
    f.close()
    print(f"\n[qm9] results written -> {out_path}")
    return str(out_path)
