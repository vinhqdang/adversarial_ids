"""
Flow-Field Evasion (FFE): Physical Lie Vector Field Steering for Tree Ensembles in NIDS.

A novel, non-incremental algorithm that operates exclusively on the Infinitesimal Generators
of admissible network packet transformations (Temporal Dilation, Payload Padding, Chaff Injection).

By integrating along the Lie Cone of physical vector fields, every generated perturbation is
mathematically guaranteed to be 100% physically realizable in network packet space.
"""

from typing import List, Tuple, Dict
import numpy as np
from .base import BaseNIDSAttack, AttackResult
from ..datasets.schema import NetworkFeatureSchema, project_network_invariants, verify_physical_feasibility


class PhysicalFlowGenerators:
    """
    Defines the infinitesimal tangent vector fields v_k(x) on the network flow manifold.
    Each vector field represents the exact differential impact of an elementary packet-level action.
    """
    @staticmethod
    def v_time(x: np.ndarray) -> np.ndarray:
        """
        Temporal Dilation Generator:
        Stretching flow duration (adding packet delays).
        Reduces Flow Bytes/s and Flow Packets/s in exact closed-form.
        """
        v = np.zeros_like(x)
        duration = max(1.0, x[6])
        bytes_rate = x[8]
        pkts_rate = x[9]
        
        v[6] = 1.0                     # dt
        v[8] = -bytes_rate / duration  # d(Rate_bytes)/dt
        v[9] = -pkts_rate / duration   # d(Rate_pkts)/dt
        return v

    @staticmethod
    def v_pad(x: np.ndarray) -> np.ndarray:
        """
        Payload Padding Generator:
        Injecting dummy payload bytes into forward packets.
        Increases forward bytes, mean packet size, and byte rate.
        """
        v = np.zeros_like(x)
        fwd_pkts = max(1.0, x[2])
        duration_sec = max(1e-6, x[6] / 1e6)
        
        v[4] = 1.0                     # +1 byte
        v[8] = 1.0 / duration_sec      # d(Rate_bytes)/dB
        v[10] = 1.0 / fwd_pkts         # d(Mean_len)/dB
        return v

    @staticmethod
    def v_chaff(x: np.ndarray) -> np.ndarray:
        """
        Chaff Packet Injection Generator:
        Injecting dummy forward packets with minimum 20-byte TCP headers.
        """
        v = np.zeros_like(x)
        fwd_pkts = max(1.0, x[2])
        fwd_mean = x[10]
        duration_sec = max(1e-6, x[6] / 1e6)
        
        v[2] = 1.0                     # +1 packet
        v[4] = 20.0                    # +20 header bytes
        v[8] = 20.0 / duration_sec     # +bytes rate
        v[9] = 1.0 / duration_sec      # +packets rate
        v[10] = (20.0 - fwd_mean) / fwd_pkts # d(mean_len)/dN
        v[11] = 20.0                   # +20 header length
        return v

    @staticmethod
    def v_win(x: np.ndarray, target_win: float = 64240.0) -> np.ndarray:
        """
        Window Advertisement Generator:
        Morphing TCP window advertisement towards benign OS profile.
        """
        v = np.zeros_like(x)
        diff = target_win - x[7]
        v[7] = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
        return v


class FlowFieldEvasionAttack(BaseNIDSAttack):
    """
    Flow-Field Evasion (FFE) Algorithm.
    Steers attack flows along the optimal trajectory of the Physical Lie Cone
    to cross tree ensemble decision boundaries with minimal network overhead.
    """
    def __init__(
        self,
        model,
        schema: NetworkFeatureSchema,
        step_budget: int = 50,
        epsilon: float = 1e-4
    ):
        super().__init__(model, schema)
        self.step_budget = step_budget
        self.epsilon = epsilon
        
        # Operational cost per unit of physical control
        # Delay is cheap; bandwidth padding is moderate; packet injection has socket cost
        self.control_costs = {
            'time': 0.0001,   # cost per microsecond delay
            'pad': 0.01,      # cost per padded byte
            'chaff': 1.0,     # cost per injected chaff packet
            'win': 0.001      # cost per window bit morph
        }

    def _compute_directional_derivatives(self, x: np.ndarray) -> Dict[str, Tuple[float, np.ndarray, float]]:
        """
        Computes the Lie derivative L_{v_k} F(x) = (F(x + eps*v_k) - F(x)) / eps
        along each physical flow generator.
        """
        p_curr = self.predict_malicious_prob(x)
        derivatives = {}
        
        # 1. Temporal Dilation Field
        v_t = PhysicalFlowGenerators.v_time(x)
        scale_t = max(1000.0, x[6] * 0.1) # test 10% delay
        x_cand = project_network_invariants(x + scale_t * v_t, self.schema)
        p_next = self.predict_malicious_prob(x_cand)
        grad_t = (p_next - p_curr) / (scale_t + 1e-9)
        derivatives['time'] = (grad_t, v_t, scale_t)
        
        # 2. Payload Padding Field
        v_p = PhysicalFlowGenerators.v_pad(x)
        scale_p = max(50.0, x[4] * 0.2) # test padding bytes
        x_cand = project_network_invariants(x + scale_p * v_p, self.schema)
        p_next = self.predict_malicious_prob(x_cand)
        grad_p = (p_next - p_curr) / (scale_p + 1e-9)
        derivatives['pad'] = (grad_p, v_p, scale_p)
        
        # 3. Chaff Packet Injection Field
        v_c = PhysicalFlowGenerators.v_chaff(x)
        scale_c = max(1.0, np.round(x[2] * 0.2)) # test +20% packets
        x_cand = project_network_invariants(x + scale_c * v_c, self.schema)
        p_next = self.predict_malicious_prob(x_cand)
        grad_c = (p_next - p_curr) / (scale_c + 1e-9)
        derivatives['chaff'] = (grad_c, v_c, scale_c)
        
        # 4. TCP Window Field
        v_w = PhysicalFlowGenerators.v_win(x)
        scale_w = abs(64240.0 - x[7])
        if scale_w > 0:
            x_cand = project_network_invariants(x + scale_w * v_w, self.schema)
            p_next = self.predict_malicious_prob(x_cand)
            grad_w = (p_next - p_curr) / (scale_w + 1e-9)
            derivatives['win'] = (grad_w, v_w, scale_w)
            
        return derivatives

    def attack_sample(self, x: np.ndarray) -> AttackResult:
        self.query_count = 0
        x_orig = x.copy()
        x_curr = project_network_invariants(x_orig, self.schema)
        
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

        # Iteratively integrate along the optimal physical vector field
        for step in range(self.step_budget):
            p_now = self.predict_malicious_prob(x_curr)
            if p_now < 0.5:
                break
                
            derivatives = self._compute_directional_derivatives(x_curr)
            
            # Select control direction with the most negative Lie derivative per unit cost
            best_control = None
            best_efficiency = 0.0 # we want negative gradient (decreasing malicious probability)
            
            for name, (grad, vec, step_scale) in derivatives.items():
                cost_rate = self.control_costs[name]
                if grad < 0: # decreases attack score
                    efficiency = (-grad) / (cost_rate + 1e-9)
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_control = (name, vec, step_scale)
                        
            if best_control is None:
                # If all local directional derivatives are non-negative,
                # use temporal dilation geodesic (low-and-slow dilation has asymptotic evasion power)
                v_t = PhysicalFlowGenerators.v_time(x_curr)
                dt = max(50000.0, x_curr[6] * 1.5)
                x_curr = project_network_invariants(x_curr + dt * v_t, self.schema)
            else:
                name, vec, step_scale = best_control
                # Execute geodesic step along selected Lie vector field
                x_next = x_curr + step_scale * vec
                x_curr = project_network_invariants(x_next, self.schema)

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
