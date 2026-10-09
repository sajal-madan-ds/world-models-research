import copy
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_supervised(model, inputs, targets, steps=250, seed=0, actions=None, lr=0.002):
    """Uniform minibatches from TRAIN episodes only; CPU generator supports MPS."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    model.train()
    for _ in range(steps):
        indices = torch.randint(len(inputs), (min(128, len(inputs)),), generator=generator)
        indices = indices.to(inputs.device)
        predicted = model(inputs[indices], None if actions is None else actions[indices])
        loss = (predicted - targets[indices]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return losses


def make_teacher(encoder):
    teacher = copy.deepcopy(encoder).eval()
    teacher.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_teacher(teacher, student, momentum=0.99):
    for target, source in zip(teacher.parameters(), student.parameters(), strict=True):
        target.mul_(momentum).add_(source, alpha=1 - momentum)
