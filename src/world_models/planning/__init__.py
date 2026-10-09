import torch


@torch.no_grad()
def cem_action(model, state, goal, generator, horizon=12, candidates=128, iterations=3, elite=16):
    """Bounded CEM shooting MPC. Only the first action is executed."""
    if not 1 <= elite <= candidates:
        raise ValueError("Invalid elite count")
    mean = torch.zeros(horizon, 2, device=state.device)
    std = torch.ones_like(mean)
    for _ in range(iterations):
        actions = (
            mean
            + std * torch.randn(candidates, horizon, 2, generator=generator, device=state.device)
        ).clamp(-1, 1)
        predicted = state.expand(candidates, -1)
        costs = torch.zeros(candidates, device=state.device)
        for t in range(horizon):
            predicted = model(predicted, actions[:, t])
            costs += (predicted[:, :2] - goal).square().sum(-1)
            costs += 0.01 * actions[:, t].square().sum(-1)
        indices = costs.topk(elite, largest=False).indices
        selected = actions[indices]
        mean, std = selected.mean(0), selected.std(0, unbiased=False).clamp_min(0.05)
    return mean[0]
