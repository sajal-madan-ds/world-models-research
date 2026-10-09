import torch


def variance_covariance(z, target_std=0.2):
    """Educational VICReg-like regularization. Scale differs from official recipes."""
    if len(z) < 2:
        raise ValueError("Covariance requires at least two samples")
    centered = z - z.mean(0)
    variance = torch.relu(target_std - torch.sqrt(z.var(0, unbiased=False) + 1e-4)).mean()
    cov = centered.T @ centered / (len(z) - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return variance, off_diag.square().sum() / z.shape[-1]


def expectile(residual, tau=0.8):
    if not 0 < tau < 1:
        raise ValueError("tau must lie strictly between 0 and 1")
    return (torch.where(residual < 0, 1 - tau, tau) * residual.square()).mean()


def goal_value(encoder, state, goal):
    return -torch.linalg.vector_norm(encoder(state) - encoder(goal), dim=-1)


def value_loss(encoder, state, next_state, goal, reached, gamma=0.98, tau=0.8):
    """Reaching-cost Bellman expectile, with an absorbing goal in our toy world."""
    with torch.no_grad():
        target = torch.where(reached, 0.0, -1.0 + gamma * goal_value(encoder, next_state, goal))
    return expectile(target - goal_value(encoder, state, goal), tau)
