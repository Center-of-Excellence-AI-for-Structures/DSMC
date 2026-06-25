"""
Smoke tests for DSMC.

These build the model from scratch and run it on synthetic data, so they need no
dataset and no trained weights. They run in seconds and cover the core property
of the model (monotonic features) and the cluster-assignment head.

Run from the repository root:
    pytest
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "dsmc")
sys.path.insert(0, PKG)

import settings          # noqa: E402
settings.init()
import models            # noqa: E402


def _ae():
    torch.manual_seed(0)
    return models.AE(
        n_inputs=14, length=10, n_features=1, hidden_dim=32, dr_rate=0.1, use_demo=False
    )


def test_ae_builds_and_runs():
    """The autoencoder accepts a batch of sequences plus a time value per sample."""
    ae = _ae()
    ae.eval()
    series = torch.randn(4, 10, 14)
    t = torch.rand(4)
    with torch.no_grad():
        _, _, encoded, _, _ = ae(series, t, None)
    assert encoded.shape == (4, 1)


def test_feature_is_monotonic_in_time():
    """Holding the input fixed and increasing time must not decrease the feature."""
    ae = _ae()
    ae.eval()
    steps = 40
    series = torch.randn(1, 10, 14).repeat(steps, 1, 1)
    t = torch.linspace(0.0, 1.0, steps)
    with torch.no_grad():
        _, _, encoded, _, _ = ae(series, t, None)
    changes = encoded[1:, 0] - encoded[:-1, 0]
    assert changes.min().item() >= -1e-4


def test_cluster_assignment_is_a_distribution():
    """The soft cluster assignment sums to 1 over clusters for every sample."""
    ca = models.ClusterAssignment(
        cluster_number=5, embedding_dimension=1, device=torch.device("cpu")
    )
    z = torch.randn(8, 1)
    q = ca(z)
    assert q.shape == (8, 5)
    assert torch.allclose(q.sum(dim=1), torch.ones(8), atol=1e-5)
