import numpy as np
import torch


def trajectory_metrics(predicted, target):
    """E,T,D arrays; first two coordinates are positions, normalized units."""
    error = np.asarray(predicted) - np.asarray(target)
    displacement = np.linalg.norm(error[..., :2], axis=-1)
    return {
        "state_mse": float(np.mean(error**2)),
        "average_displacement": float(displacement.mean()),
        "final_displacement": float(displacement[:, -1].mean()),
        "mse_by_horizon": np.mean(error**2, axis=(0, 2)).tolist(),
    }


def representation_metrics(z):
    z = z.detach().float().cpu()
    centered = z - z.mean(0)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered / max(len(z) - 1, 1)).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    return {
        "mean_feature_std": float(z.std(0, unbiased=False).mean()),
        "effective_rank": float(rank) if eigenvalues.sum() > 0 else 0.0,
        "latent_dim": z.shape[-1],
    }
