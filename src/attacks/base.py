"""
Base classes and result containers for adversarial evasion attacks against NIDS.
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from ..datasets.schema import NetworkFeatureSchema, verify_physical_feasibility


@dataclass
class AttackResult:
    x_original: np.ndarray
    x_adversarial: np.ndarray
    success: bool                # Model predicted Benign (< 0.5)
    physically_valid: bool       # Respects all network invariants & monotonicity
    queries_used: int            # Number of model evaluation calls
    weighted_cost: float         # Problem-space perturbation cost
    feature_shifts: np.ndarray   # Delta vector (x_adv - x_orig)


class BaseNIDSAttack(ABC):
    def __init__(self, model, schema: NetworkFeatureSchema):
        self.model = model
        self.schema = schema
        self.query_count = 0

    def predict_malicious_prob(self, x: np.ndarray) -> float:
        self.query_count += 1
        probs = self.model.predict_proba(x)
        return float(probs[0, 1] if probs.ndim == 2 else probs[1])

    @abstractmethod
    def attack_sample(self, x: np.ndarray) -> AttackResult:
        """Execute evasion attack on a single malicious flow."""
        pass
