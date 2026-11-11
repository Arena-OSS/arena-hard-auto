import unittest
from typing import Tuple

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit

from utils.math_utils import bt_loss, fit_pairwise_model


class PairwiseModelTest(unittest.TestCase):
    def test_bt_fit_matches_independent_maximum_likelihood(self) -> None:
        rng = np.random.default_rng(42)
        models = rng.integers(0, 3, size=3000)
        styles = rng.normal(size=(3000, 3))
        design = np.column_stack((np.eye(3)[models], styles))
        probabilities = expit(design @ np.array([-1.2, 0.1, 1.4, 0.7, -0.5, 0.3]))
        outcomes = rng.binomial(1, probabilities).astype(np.float32)
        outcomes[::19] = 0.5

        def objective(coefs: np.ndarray) -> Tuple[float, np.ndarray]:
            logits = design @ coefs
            return (
                np.logaddexp(0, logits).sum() - outcomes @ logits,
                design.T @ (expit(logits) - outcomes),
            )

        reference = minimize(
            objective, np.zeros(design.shape[1]), jac=True, method="BFGS",
            options={"gtol": 1e-7},
        )
        self.assertLess(np.abs(reference.jac).max(), 1e-5)
        fitted, eta = fit_pairwise_model(
            torch.tensor(design, dtype=torch.float32), torch.tensor(outcomes)
        )
        self.assertIsNone(eta)
        self.assertLess(np.abs(objective(fitted.numpy())[1]).max(), 0.01)
        np.testing.assert_allclose(
            expit(fitted.numpy()[:3]), expit(reference.x[:3]), atol=1e-5
        )

    def test_bt_bootstrap_indices_match_explicit_sample(self) -> None:
        features = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
        outcomes = torch.tensor([0.0, 1.0, 1.0, 0.5])
        indices = torch.tensor([0, 1, 2, 2, 3])
        sampled, _ = fit_pairwise_model(features, outcomes, indices=indices)
        explicit, _ = fit_pairwise_model(features[indices], outcomes[indices])
        torch.testing.assert_close(sampled, explicit)

    def test_bt_loss_preserves_fractional_outcome_precision(self) -> None:
        logits = torch.tensor([1.0], dtype=torch.float64)
        outcomes = torch.tensor([0.123456789123], dtype=torch.float64)
        loss = bt_loss(logits, outcomes)
        expected = np.logaddexp(0, 1.0) - outcomes.item()
        self.assertEqual(loss.dtype, torch.float64)
        self.assertAlmostEqual(loss.item(), expected, places=12)

    def test_rk_defaults_remain_unchanged(self) -> None:
        features = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
        outcomes = torch.tensor([0.0, 0.5, 1.0, 1.0])
        default, default_eta = fit_pairwise_model(features, outcomes, loss_type="rk")
        explicit, explicit_eta = fit_pairwise_model(
            features, outcomes, loss_type="rk", lr=0.1, max_epochs=50
        )
        torch.testing.assert_close(default, explicit)
        torch.testing.assert_close(default_eta, explicit_eta)


if __name__ == "__main__":
    unittest.main()
