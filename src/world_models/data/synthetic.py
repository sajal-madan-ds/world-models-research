"""Small deterministic worlds with explicit state and reproducible random generators."""

import numpy as np


def ball_step(state, dt=0.08):
    """Exact elastic reflections in a unit box, including multiple boundary crossings.

    State order is x,y,vx,vy. The ball is a point mass. At exact boundaries velocity
    is directed inward; no gravity, friction, contact radius, or other objects.
    """
    state = np.asarray(state)
    raw = state[..., :2] + dt * state[..., 2:]
    phase = np.mod(raw, 2.0)
    position = np.where(phase <= 1, phase, 2 - phase)
    velocity = state[..., 2:] * np.where(phase < 1, 1.0, -1.0)
    velocity = np.where(position == 0, np.abs(velocity), velocity)
    velocity = np.where(position == 1, -np.abs(velocity), velocity)
    return np.concatenate([position, velocity], axis=-1).astype(np.float32)


def ball_trajectories(seed=0, episodes=96, length=40):
    rng = np.random.default_rng(seed)
    state = np.concatenate(
        [rng.uniform(0.05, 0.95, (episodes, 2)), rng.uniform(-0.7, 0.7, (episodes, 2))], axis=-1
    ).astype(np.float32)
    frames = [state]
    for _ in range(length - 1):
        state = ball_step(state)
        frames.append(state)
    return np.stack(frames, axis=1)


def render_history(states, size=16):
    """Render Gaussian blobs; two ordered frames make velocity partly observable.

    Output E,T-1,2,H,W. Blur is smooth and includes clipped boundary appearances.
    """
    grid = np.linspace(0, 1, size, dtype=np.float32)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    d2 = (xx - states[..., 0, None, None]) ** 2 + (yy - states[..., 1, None, None]) ** 2
    images = np.exp(-d2 / (2 * 0.06**2)).astype(np.float32)
    return np.stack([images[:, :-1], images[:, 1:]], axis=2)


def control_step(state, action):
    """Damped point robot: action is bounded acceleration, dt=0.1."""
    velocity = 0.9 * state[..., 2:] + 0.1 * np.clip(action, -1, 1)
    position = state[..., :2] + 0.1 * velocity
    return np.concatenate([position, velocity], axis=-1).astype(np.float32)


def control_trajectories(seed=0, episodes=96, length=32):
    rng = np.random.default_rng(seed)
    state = rng.uniform(-0.5, 0.5, (episodes, 4)).astype(np.float32)
    states, actions = [state], []
    for _ in range(length - 1):
        action = rng.uniform(-1, 1, (episodes, 2)).astype(np.float32)
        actions.append(action)
        state = control_step(state, action)
        states.append(state)
    return np.stack(states, 1), np.stack(actions, 1)
