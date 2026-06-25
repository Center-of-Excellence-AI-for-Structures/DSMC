"""
Dataset-free demo of the DSMC monotonic feature extractor.

It builds the monotonic autoencoder, feeds it one synthetic input sequence, and
sweeps the time input from 0 to 1. The extracted health feature increases at
every step, which is the core property of the model: the learned feature can
only grow as the system ages (a "soft" monotonic health indicator). No dataset
or trained weights are needed, so a reviewer can run it in a few seconds on CPU.

Run from the repository root:
    python examples/demo_monotonic.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "dsmc")
sys.path.insert(0, PKG)

import settings          # noqa: E402
settings.init()          # defines the global seed the model layers read at build time
import models            # noqa: E402


def main():
    torch.manual_seed(0)
    n_inputs, length, n_features, hidden_dim = 14, 10, 1, 32
    ae = models.AE(
        n_inputs=n_inputs,
        length=length,
        n_features=n_features,
        hidden_dim=hidden_dim,
        dr_rate=0.1,
        use_demo=False,
    )
    ae.eval()
    n_params = sum(p.numel() for p in ae.parameters())
    print(f"Monotonic autoencoder built. Trainable parameters: {n_params / 1e6:.3f}M")

    # One fixed synthetic input sequence, repeated; sweep the time input from 0 to 1.
    steps = 50
    series = torch.randn(1, length, n_inputs).repeat(steps, 1, 1)
    t = torch.linspace(0.0, 1.0, steps)
    with torch.no_grad():
        _, _, encoded, _, _ = ae(series, t, demo=None)
    feature = encoded[:, 0]
    changes = feature[1:] - feature[:-1]

    print(f"Health feature at t=0.0 : {feature[0].item():.3f}")
    print(f"Health feature at t=1.0 : {feature[-1].item():.3f}")
    print(f"Smallest step-to-step change: {changes.min().item():.3e}")
    if changes.min().item() >= -1e-4:
        print("\nThe health feature is monotonically non-decreasing in time, as designed.")
    else:
        print("\nWARNING: monotonicity was not satisfied.")
    print("\nNote: the weights are random and untrained, so the absolute values carry no")
    print("meaning. The point being demonstrated is the strictly increasing trend with time.")


if __name__ == "__main__":
    main()
