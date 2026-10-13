"""
PACT-GPU: Protocol-Aware Convex-Partition Tree Evasion Algorithm.

A novel adversarial evasion algorithm tailored for Tree Ensembles (XGBoost, LightGBM)
in Network Intrusion Detection Systems (NIDS).

Key Innovations:
1. Exact Problem-Space Invariant Manifold Projection (guarantees 100% physical validity).
2. GPU-Accelerated Differentiable Soft-Tree Path Relaxation (smooth gradient flow across orthogonal splits).
3. Tree-Split Boundary Snapping (TSBS): discrete threshold snapping to cross exact decision boundaries.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from .base import BaseNIDSAttack, AttackResult
from ..datasets.schema import NetworkFeatureSchema, project_network_invariants, verify_physical_feasibility


class DifferentiableNetworkManifold(nn.Module):
    """
    PyTorch Differentiable Manifold Projection Layer on CUDA.
    Enforces immutability, monotonicity, and non-linear flow coupling invariants.
    """
    def __init__(self, schema: NetworkFeatureSchema, device: torch.device):
        super().__init__()
        self.schema = schema
        self.device = device
        
        self.immutable_indices = torch.tensor(schema.immutable_indices, dtype=torch.long, device=device)
        self.monotonic_indices = torch.tensor(schema.monotonic_increasing_indices, dtype=torch.long, device=device)
        self.costs = torch.tensor(schema.perturbation_costs, dtype=torch.float32, device=device)

    def forward(self, x_orig: torch.Tensor, delta_raw: torch.Tensor) -> torch.Tensor:
        """
        x_orig: (1, D) original attack feature vector on GPU
        delta_raw: (1, D) unconstrained perturbation parameter
        Returns: x_proj (1, D) strictly feasible network flow on GPU
        """
        x_mod = x_orig.clone()
        
        # 1. Zero out immutable features (ports, protocol, server bytes)
        delta_filtered = delta_raw.clone()
        delta_filtered[:, self.immutable_indices] = 0.0
        
        # 2. Enforce monotonicity (only positive shifts allowed for packet/byte injection and delay)
        for idx in self.schema.monotonic_increasing_indices:
            delta_filtered[:, idx] = torch.relu(delta_filtered[:, idx])
            
        # 3. Apply base perturbations
        x_base = x_orig + delta_filtered
        
        # Clamp base features to reasonable physical network bounds
        tot_fwd_pkts = torch.clamp(x_base[:, 2], min=1.0)
        tot_bwd_pkts = torch.clamp(x_base[:, 3], min=0.0)
        totlen_fwd_pkts = torch.clamp(x_base[:, 4], min=tot_fwd_pkts * 20.0) # at least TCP headers
        totlen_bwd_pkts = torch.clamp(x_base[:, 5], min=0.0)
        flow_duration = torch.clamp(x_base[:, 6], min=1.0) # microsecs
        init_win = torch.clamp(x_base[:, 7], min=0.0, max=65535.0)
        
        # 4. Analytically project coupled flow statistics (Non-linear Manifold M_flow)
        duration_sec = flow_duration / 1e6
        flow_bytes_s = (totlen_fwd_pkts + totlen_bwd_pkts) / (duration_sec + 1e-9)
        flow_pkts_s = (tot_fwd_pkts + tot_bwd_pkts) / (duration_sec + 1e-9)
        fwd_pkt_len_mean = totlen_fwd_pkts / tot_fwd_pkts
        fwd_header_len = torch.maximum(tot_fwd_pkts * 20.0, x_base[:, 11])
        
        # Assemble complete projected vector
        x_proj = torch.zeros_like(x_orig)
        x_proj[:, 0] = x_orig[:, 0]        # Dst Port (Immutable)
        x_proj[:, 1] = x_orig[:, 1]        # Protocol (Immutable)
        x_proj[:, 2] = tot_fwd_pkts        # Fwd Pkts
        x_proj[:, 3] = tot_bwd_pkts        # Bwd Pkts
        x_proj[:, 4] = totlen_fwd_pkts     # Fwd Bytes
        x_proj[:, 5] = totlen_bwd_pkts     # Bwd Bytes
        x_proj[:, 6] = flow_duration       # Duration
        x_proj[:, 7] = init_win            # TCP Window
        x_proj[:, 8] = flow_bytes_s        # Coupled: Flow Bytes/s
        x_proj[:, 9] = flow_pkts_s         # Coupled: Flow Pkts/s
        x_proj[:, 10] = fwd_pkt_len_mean   # Coupled: Fwd Pkt Len Mean
        x_proj[:, 11] = fwd_header_len     # Coupled: Header Length
        
        return x_proj


class SoftTreeSurrogate(nn.Module):
    """
    Differentiable GPU relaxation of tree split boundaries.
    Allows smooth gradient backpropagation to guide problem-space search.
    """
    def __init__(self, splits: List[Tuple[int, float, float]], temperature: float = 2.0, device: torch.device = None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        
        # splits: list of (feature_idx, threshold, score_weight)
        if len(splits) > 0:
            feat_indices = [s[0] for s in splits]
            thresholds = [s[1] for s in splits]
            weights = [s[2] for s in splits]
            
            self.feat_indices = torch.tensor(feat_indices, dtype=torch.long, device=self.device)
            self.thresholds = torch.tensor(thresholds, dtype=torch.float32, device=self.device)
            self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            self.feat_indices = torch.empty(0, dtype=torch.long, device=self.device)
            self.thresholds = torch.empty(0, dtype=torch.float32, device=self.device)
            self.weights = torch.empty(0, dtype=torch.float32, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.feat_indices) == 0:
            return torch.tensor(0.0, device=self.device)
        
        x_selected = x[:, self.feat_indices]
        # Smooth sigmoid gate over tree splits
        diff = (x_selected - self.thresholds) / self.temperature
        gates = torch.sigmoid(diff)
        score = torch.sum(gates * self.weights)
        return score


class PACTEvasionAttack(BaseNIDSAttack):
    """
    PACT-GPU (Protocol-Aware Convex-Partition Tree Evasion):
    Dual-phase algorithm combining GPU-accelerated manifold gradient flow with
    discrete tree split boundary snapping.
    """
    def __init__(
        self,
        model,
        schema: NetworkFeatureSchema,
        lr: float = 0.5,
        max_grad_steps: int = 40,
        max_snapping_steps: int = 20,
        device: Optional[torch.device] = None
    ):
        super().__init__(model, schema)
        self.lr = lr
        self.max_grad_steps = max_grad_steps
        self.max_snapping_steps = max_snapping_steps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.manifold = DifferentiableNetworkManifold(schema, self.device)
        self._parsed_splits = self._extract_splits()

    def _extract_splits(self) -> List[Tuple[int, float, float]]:
        """
        Parses XGBoost or LightGBM model splits to construct the differentiable surrogate.
        """
        splits = []
        try:
            # Check for XGBoost booster
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'get_booster'):
                booster = self.model.model.get_booster()
                dump = booster.get_dump(dump_format='text')
                for tree_text in dump:
                    for line in tree_text.splitlines():
                        line = line.strip()
                        if '<' in line and '[' in line and ']' in line:
                            bracket_content = line[line.find('[') + 1:line.find(']')]
                            parts = bracket_content.split('<')
                            if len(parts) == 2:
                                feat_str = parts[0].strip()
                                try:
                                    # feature names might be f0, f1 or actual names
                                    if feat_str.startswith('f') and feat_str[1:].isdigit():
                                        f_idx = int(feat_str[1:])
                                    elif feat_str in self.schema.feature_names:
                                        f_idx = self.schema.feature_names.index(feat_str)
                                    else:
                                        continue
                                    thresh = float(parts[1].strip())
                                    splits.append((f_idx, thresh, 1.0))
                                except ValueError:
                                    pass
        except Exception:
            pass
        return splits

    def attack_sample(self, x: np.ndarray) -> AttackResult:
        self.query_count = 0
        x_orig = x.copy()
        
        # Ensure base flow is strictly physically valid initially
        x_curr = project_network_invariants(x_orig, self.schema)
        
        # Initial check
        p_init = self.predict_malicious_prob(x_curr)
        if p_init < 0.5:
            return AttackResult(
                x_original=x_orig,
                x_adversarial=x_curr,
                success=True,
                physically_valid=True,
                queries_used=self.query_count,
                weighted_cost=0.0,
                feature_shifts=np.zeros_like(x_orig)
            )

        # ---------------- Phase 1: GPU Manifold Gradient Guidance ----------------
        x_orig_t = torch.tensor(x_orig.reshape(1, -1), dtype=torch.float32, device=self.device)
        delta_param = torch.zeros(1, len(x_orig), dtype=torch.float32, device=self.device, requires_grad=True)
        
        surrogate = SoftTreeSurrogate(self._parsed_splits, temperature=10.0, device=self.device)
        optimizer = torch.optim.Adam([delta_param], lr=self.lr)
        
        for g_step in range(self.max_grad_steps):
            optimizer.zero_grad()
            x_proj_t = self.manifold(x_orig_t, delta_param)
            
            # Loss: surrogate tree malicious score + L1 perturbation penalty
            loss_surr = surrogate(x_proj_t)
            cost_l1 = torch.sum(self.manifold.costs * torch.abs(x_proj_t - x_orig_t))
            loss = loss_surr + 0.01 * cost_l1
            
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            x_guided_t = self.manifold(x_orig_t, delta_param)
            x_guided = x_guided_t.cpu().numpy()[0]
            
        # Re-project to exact discrete network invariants
        x_curr = project_network_invariants(x_guided, self.schema)
        p_guided = self.predict_malicious_prob(x_curr)
        
        if p_guided < 0.5:
            cost = float(np.sum(self.schema.perturbation_costs * np.abs(x_curr - x_orig)))
            return AttackResult(
                x_original=x_orig,
                x_adversarial=x_curr,
                success=True,
                physically_valid=verify_physical_feasibility(x_orig, x_curr, self.schema),
                queries_used=self.query_count,
                weighted_cost=cost,
                feature_shifts=x_curr - x_orig
            )

        # ---------------- Phase 2: Tree-Split Boundary Snapping (TSBS) ----------------
        # Greedily search along mutable base dimensions (Packets, Bytes, Duration, Window)
        # to cross the nearest threshold partition that minimizes tree ensemble score
        mutable_dims = [2, 4, 6, 7] # Fwd Pkts, Fwd Bytes, Duration, Win
        
        for snap_iter in range(self.max_snapping_steps):
            best_dim = None
            best_prob = self.predict_malicious_prob(x_curr)
            best_x_cand = None
            
            # Explore adaptive multipliers (packet injection, delay stretching, padding)
            multipliers = [1.2, 1.5, 2.0, 3.0, 5.0]
            
            for dim in mutable_dims:
                for mult in multipliers:
                    x_cand = x_curr.copy()
                    
                    if dim == 6: # Duration: extending duration reduces flow rates
                        x_cand[dim] = max(x_cand[dim] * mult, x_cand[dim] + 50000.0)
                    elif dim in [2, 4]: # Packets / Bytes
                        x_cand[dim] = x_cand[dim] * mult
                    elif dim == 7: # Window: try typical benign OS windows
                        x_cand[dim] = 64240.0
                        
                    # Project strictly via physical network conservation laws
                    x_cand = project_network_invariants(x_cand, self.schema)
                    
                    prob = self.predict_malicious_prob(x_cand)
                    if prob < best_prob:
                        best_prob = prob
                        best_dim = dim
                        best_x_cand = x_cand.copy()
                        
            if best_x_cand is not None and best_prob < self.predict_malicious_prob(x_curr):
                x_curr = best_x_cand
            else:
                # Force duration elongation (classic low-and-slow evasion strategy)
                x_curr[6] = x_curr[6] * 2.5 + 100000.0
                x_curr = project_network_invariants(x_curr, self.schema)
                
            p_now = self.predict_malicious_prob(x_curr)
            if p_now < 0.5:
                break

        final_prob = self.predict_malicious_prob(x_curr)
        success = (final_prob < 0.5)
        phys_valid = verify_physical_feasibility(x_orig, x_curr, self.schema)
        cost = float(np.sum(self.schema.perturbation_costs * np.abs(x_curr - x_orig)))

        return AttackResult(
            x_original=x_orig,
            x_adversarial=x_curr,
            success=success,
            physically_valid=phys_valid,
            queries_used=self.query_count,
            weighted_cost=cost,
            feature_shifts=x_curr - x_orig
        )
