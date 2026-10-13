"""
Unconstrained Feature-Space Attack Baseline.

Performs mathematical gradient/coordinate perturbation in feature space
without respecting problem-space constraints (demonstrates the 'illusive' evasion flaw).
"""

import numpy as np
from .base import BaseNIDSAttack, AttackResult
from ..datasets.schema import verify_physical_feasibility


class UnconstrainedFeatureSpaceAttack(BaseNIDSAttack):
    """
    Standard naive coordinate-descent attack operating purely in feature space.
    Ignores protocol invariants, immutability, and monotonicity.
    """
    def __init__(self, model, schema, max_iter: int = 50, step_size: float = 0.05):
        super().__init__(model, schema)
        self.max_iter = max_iter
        self.step_size = step_size

    def attack_sample(self, x: np.ndarray) -> AttackResult:
        self.query_count = 0
        x_orig = x.copy()
        x_adv = x.copy()
        
        init_prob = self.predict_malicious_prob(x_adv)
        if init_prob < 0.5:
            return AttackResult(
                x_original=x_orig,
                x_adversarial=x_adv,
                success=True,
                physically_valid=True,
                queries_used=self.query_count,
                weighted_cost=0.0,
                feature_shifts=np.zeros_like(x_orig)
            )

        n_features = len(x_orig)
        std_estimates = np.maximum(np.abs(x_orig) * 0.1, 1.0)
        
        for iteration in range(self.max_iter):
            best_feat = None
            best_dir = 0
            best_prob = self.predict_malicious_prob(x_adv)
            
            # Finite-difference numerical gradient in unconstrained feature space
            for f in range(n_features):
                delta = self.step_size * std_estimates[f]
                
                # Test positive step
                x_cand = x_adv.copy()
                x_cand[f] += delta
                p_pos = self.predict_malicious_prob(x_cand)
                if p_pos < best_prob:
                    best_prob = p_pos
                    best_feat = f
                    best_dir = 1
                    
                # Test negative step
                x_cand[f] -= 2 * delta
                p_neg = self.predict_malicious_prob(x_cand)
                if p_neg < best_prob:
                    best_prob = p_neg
                    best_feat = f
                    best_dir = -1
                    
            if best_feat is not None and best_dir != 0:
                x_adv[best_feat] += best_dir * self.step_size * std_estimates[best_feat]
            else:
                break
                
            if best_prob < 0.5:
                break

        final_prob = self.predict_malicious_prob(x_adv)
        success = (final_prob < 0.5)
        phys_valid = verify_physical_feasibility(x_orig, x_adv, self.schema)
        cost = float(np.sum(self.schema.perturbation_costs * np.abs(x_adv - x_orig)))

        return AttackResult(
            x_original=x_orig,
            x_adversarial=x_adv,
            success=success,
            physically_valid=phys_valid,
            queries_used=self.query_count,
            weighted_cost=cost,
            feature_shifts=x_adv - x_orig
        )
