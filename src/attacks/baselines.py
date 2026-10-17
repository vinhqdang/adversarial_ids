"""
Baseline Adversarial Evasion Algorithms against NIDS (Paper Section 6 & Table 2).

Implements:
1. ZOONIDSAttack: Zeroth-Order Optimization (Chen et al. 2017) adapted for tabular flows.
2. HopSkipJumpNIDSAttack: Boundary-following decision-based attack (Chen et al. 2020).
3. TrafficManipulatorPSOAttack: Particle Swarm Optimization for packet interval & padding manipulation (Han et al. 2021).
4. MOEVAAttack: Multi-Objective Evolutionary Adversarial Attack (Simonetto et al. 2024, Ghamizi et al. 2020).
"""

import numpy as np
from typing import Optional, List
from .base import BaseNIDSAttack, AttackResult
from ..datasets.schema import NetworkFeatureSchema, verify_physical_feasibility, project_network_invariants


class ZOONIDSAttack(BaseNIDSAttack):
    """
    Zeroth-Order Optimization (ZOO) adapted for network flow features.
    Samples random coordinates and uses coordinate-wise symmetric difference quotients
    to approximate gradients for Adam updates.
    """
    def __init__(self, model, schema: NetworkFeatureSchema, max_iter: int = 50, step_size: float = 0.05):
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
                x_original=x_orig, x_adversarial=x_adv, success=True, physically_valid=True,
                queries_used=self.query_count, weighted_cost=0.0, feature_shifts=np.zeros_like(x_orig)
            )

        n_features = len(x_orig)
        stds = np.maximum(np.abs(x_orig) * 0.1, 1.0)
        h = 1e-4

        for _ in range(self.max_iter):
            # Sample coordinate
            j = np.random.randint(0, n_features)
            # Symmetric difference quotient
            x_plus = x_adv.copy()
            x_plus[j] += h * stds[j]
            x_minus = x_adv.copy()
            x_minus[j] -= h * stds[j]

            p_plus = self.predict_malicious_prob(x_plus)
            p_minus = self.predict_malicious_prob(x_minus)
            grad_est = (p_plus - p_minus) / (2.0 * h * stds[j] + 1e-9)

            # Update coordinate in direction of decreasing malicious probability
            x_adv[j] -= self.step_size * stds[j] * np.sign(grad_est)
            x_adv[j] = max(0.0, x_adv[j])  # Box constraint non-negativity

            prob = self.predict_malicious_prob(x_adv)
            if prob < 0.5:
                break

        final_prob = self.predict_malicious_prob(x_adv)
        success = (final_prob < 0.5)
        phys_valid = verify_physical_feasibility(x_orig, x_adv, self.schema)
        cost = float(np.sum(self.schema.perturbation_costs * np.abs(x_adv - x_orig)))

        return AttackResult(
            x_original=x_orig, x_adversarial=x_adv, success=success,
            physically_valid=phys_valid, queries_used=self.query_count,
            weighted_cost=cost, feature_shifts=x_adv - x_orig
        )


class HopSkipJumpNIDSAttack(BaseNIDSAttack):
    """
    HopSkipJump decision-based boundary-seeking attack.
    Initializes from a known benign point and performs binary search
    to find the decision boundary, followed by directional refinement.
    """
    def __init__(self, model, schema: NetworkFeatureSchema, max_iter: int = 30, benign_anchor: Optional[np.ndarray] = None):
        super().__init__(model, schema)
        self.max_iter = max_iter
        self.benign_anchor = benign_anchor

    def attack_sample(self, x: np.ndarray) -> AttackResult:
        self.query_count = 0
        x_orig = x.copy()

        init_prob = self.predict_malicious_prob(x_orig)
        if init_prob < 0.5:
            return AttackResult(
                x_original=x_orig, x_adversarial=x_orig, success=True, physically_valid=True,
                queries_used=self.query_count, weighted_cost=0.0, feature_shifts=np.zeros_like(x_orig)
            )

        # Default anchor: typical HTTP web browsing benign flow
        if self.benign_anchor is None:
            anchor = np.array([443.0, 6.0, 8.0, 10.0, 1200.0, 8500.0, 1.5e6, 29200.0, 6466.0, 12.0, 150.0, 160.0])
        else:
            anchor = self.benign_anchor.copy()

        # Binary search between x_orig and benign anchor
        low, high = 0.0, 1.0
        x_boundary = anchor.copy()
        for _ in range(15):
            mid = (low + high) / 2.0
            x_mid = (1.0 - mid) * x_orig + mid * anchor
            if self.predict_malicious_prob(x_mid) < 0.5:
                x_boundary = x_mid.copy()
                high = mid
            else:
                low = mid

        # Boundary refinement steps
        x_adv = x_boundary.copy()
        stds = np.maximum(np.abs(x_orig) * 0.05, 1.0)
        for _ in range(self.max_iter):
            perturb = np.random.randn(len(x_orig)) * stds * 0.1
            x_cand = np.maximum(0.0, x_adv + perturb)
            if self.predict_malicious_prob(x_cand) < 0.5:
                # Keep if closer to original
                if np.linalg.norm(x_cand - x_orig) < np.linalg.norm(x_adv - x_orig):
                    x_adv = x_cand

        final_prob = self.predict_malicious_prob(x_adv)
        success = (final_prob < 0.5)
        phys_valid = verify_physical_feasibility(x_orig, x_adv, self.schema)
        cost = float(np.sum(self.schema.perturbation_costs * np.abs(x_adv - x_orig)))

        return AttackResult(
            x_original=x_orig, x_adversarial=x_adv, success=success,
            physically_valid=phys_valid, queries_used=self.query_count,
            weighted_cost=cost, feature_shifts=x_adv - x_orig
        )


class TrafficManipulatorPSOAttack(BaseNIDSAttack):
    """
    Particle Swarm Optimization (Traffic Manipulator PSO) mutating packet intervals
    and forward dummy bytes to evade detection while respecting physical packet laws.
    """
    def __init__(self, model, schema: NetworkFeatureSchema, n_particles: int = 15, max_iter: int = 25):
        super().__init__(model, schema)
        self.n_particles = n_particles
        self.max_iter = max_iter

    def attack_sample(self, x: np.ndarray) -> AttackResult:
        self.query_count = 0
        x_orig = x.copy()

        init_prob = self.predict_malicious_prob(x_orig)
        if init_prob < 0.5:
            return AttackResult(
                x_original=x_orig, x_adversarial=x_orig, success=True, physically_valid=True,
                queries_used=self.query_count, weighted_cost=0.0, feature_shifts=np.zeros_like(x_orig)
            )

        # Particles mutate (duration_delay, padding_bytes, chaff_pkts)
        # Dimensions: [delta_duration, delta_bytes, delta_pkts]
        dim = 3
        particles = np.random.uniform(0.0, 1.0, (self.n_particles, dim))
        velocities = np.zeros((self.n_particles, dim))
        
        pbest = particles.copy()
        pbest_scores = np.ones(self.n_particles)
        gbest = particles[0].copy()
        gbest_score = 1.0

        for it in range(self.max_iter):
            for i in range(self.n_particles):
                p = particles[i]
                # Construct candidate flow respecting physical invariants
                cand = x_orig.copy()
                cand[6] += p[0] * 5e6           # add up to 5s delay
                cand[4] += p[1] * 2000.0        # add up to 2KB padding
                cand[2] += np.floor(p[2] * 20.0) # add up to 20 chaff pkts
                cand = project_network_invariants(cand, self.schema)

                score = self.predict_malicious_prob(cand)
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = p.copy()
                if score < gbest_score:
                    gbest_score = score
                    gbest = p.copy()

            if gbest_score < 0.5:
                break

            # Update particle velocities and positions
            w, c1, c2 = 0.5, 1.0, 1.0
            r1 = np.random.rand(self.n_particles, dim)
            r2 = np.random.rand(self.n_particles, dim)
            velocities = w * velocities + c1 * r1 * (pbest - particles) + c2 * r2 * (gbest - particles)
            particles = np.clip(particles + velocities, 0.0, 2.0)

        # Construct final best flow
        x_adv = x_orig.copy()
        x_adv[6] += gbest[0] * 5e6
        x_adv[4] += gbest[1] * 2000.0
        x_adv[2] += np.floor(gbest[2] * 20.0)
        x_adv = project_network_invariants(x_adv, self.schema)

        final_prob = self.predict_malicious_prob(x_adv)
        success = (final_prob < 0.5)
        phys_valid = verify_physical_feasibility(x_orig, x_adv, self.schema)
        cost = float(np.sum(self.schema.perturbation_costs * np.abs(x_adv - x_orig)))

        return AttackResult(
            x_original=x_orig, x_adversarial=x_adv, success=success,
            physically_valid=phys_valid, queries_used=self.query_count,
            weighted_cost=cost, feature_shifts=x_adv - x_orig
        )


class MOEVAAttack(BaseNIDSAttack):
    """
    Multi-Objective Evolutionary Adversarial Attack (MOEVA) (Simonetto et al. 2024, Ghamizi et al. 2020).
    Evolves a population of candidate adversarial flows using multi-objective optimization:
      Objective 1: Minimize classifier malicious probability P(Malicious | x)
      Objective 2: Minimize normalized perturbation cost D_L1(x, x_orig)
    Applies box constraints and domain bounds without closed-form Lie manifold projection,
    allowing empirical evaluation of physical admissibility (PFR) under feature-space evolution.
    """
    def __init__(
        self,
        model,
        schema: NetworkFeatureSchema,
        pop_size: int = 25,
        n_generations: int = 20,
        mutation_rate: float = 0.25,
        **kwargs
    ):
        super().__init__(model, schema)
        self.pop_size = kwargs.get('n_particles', pop_size)
        self.n_generations = kwargs.get('max_iter', n_generations)
        self.mutation_rate = mutation_rate

    def attack_sample(self, x: np.ndarray) -> AttackResult:
        self.query_count = 0
        x_orig = x.copy()

        init_prob = self.predict_malicious_prob(x_orig)
        if init_prob < 0.5:
            return AttackResult(
                x_original=x_orig, x_adversarial=x_orig, success=True, physically_valid=True,
                queries_used=self.query_count, weighted_cost=0.0, feature_shifts=np.zeros_like(x_orig)
            )

        n_features = len(x_orig)
        stds = np.maximum(np.abs(x_orig) * 0.1, 1.0)

        # Initialize population around x_orig with slight random mutations on non-immutable features
        population = []
        for _ in range(self.pop_size):
            ind = x_orig.copy()
            for j in range(n_features):
                if j not in self.schema.immutable_indices:
                    delta = np.random.randn() * stds[j] * 0.1
                    if j in self.schema.monotonic_increasing_indices:
                        ind[j] = max(x_orig[j], ind[j] + max(0.0, delta))
                    else:
                        ind[j] = max(0.0, ind[j] + delta)
            population.append(ind)

        best_ind = x_orig.copy()
        best_prob = init_prob
        best_cost = 0.0

        for gen in range(self.n_generations):
            scores = []
            costs = []
            for ind in population:
                prob = self.predict_malicious_prob(ind)
                cost = float(np.sum(self.schema.perturbation_costs * np.abs(ind - x_orig)))
                scores.append(prob)
                costs.append(cost)

                if prob < 0.5:
                    if best_prob >= 0.5 or cost < best_cost:
                        best_ind = ind.copy()
                        best_prob = prob
                        best_cost = cost
                elif best_prob >= 0.5 and prob < best_prob:
                    best_ind = ind.copy()
                    best_prob = prob
                    best_cost = cost

            if best_prob < 0.5:
                break

            # Selection and mutation
            new_pop = []
            for _ in range(self.pop_size):
                i1, i2 = np.random.randint(0, self.pop_size, size=2)
                parent = population[i1] if scores[i1] < scores[i2] else population[i2]
                child = parent.copy()

                for j in range(n_features):
                    if j not in self.schema.immutable_indices and np.random.rand() < self.mutation_rate:
                        step = np.random.randn() * stds[j] * 0.15
                        if j in self.schema.monotonic_increasing_indices:
                            child[j] = max(x_orig[j], child[j] + max(0.0, step))
                        else:
                            child[j] = max(0.0, child[j] + step)

                new_pop.append(child)
            population = new_pop

        final_prob = self.predict_malicious_prob(best_ind)
        success = (final_prob < 0.5)
        phys_valid = verify_physical_feasibility(x_orig, best_ind, self.schema)
        cost = float(np.sum(self.schema.perturbation_costs * np.abs(best_ind - x_orig)))

        return AttackResult(
            x_original=x_orig, x_adversarial=best_ind, success=success,
            physically_valid=phys_valid, queries_used=self.query_count,
            weighted_cost=cost, feature_shifts=best_ind - x_orig
        )

