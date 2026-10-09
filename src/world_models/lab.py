"""Nine linked experiments. Scientific scope and altered reproduction are explicit."""

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from world_models.data.synthetic import (
    ball_step,
    ball_trajectories,
    control_step,
    control_trajectories,
    render_history,
)
from world_models.dynamics import Transition, rollout
from world_models.encoders import ImageDecoder, ImageEncoder
from world_models.evaluation import representation_metrics, trajectory_metrics
from world_models.losses import value_loss, variance_covariance
from world_models.planning import cem_action
from world_models.policies import proportional_derivative
from world_models.training import make_teacher, seed_everything, train_supervised, update_teacher


def tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def plot_rollouts(path, truth, predictions):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(truth[0, :, 0], truth[0, :, 1], "k.-", label="truth")
    for name, values in predictions.items():
        axes[0].plot(values[0, :, 0], values[0, :, 1], ".-", label=name)
        axes[1].plot(np.mean((values - truth) ** 2, axis=(0, 2)), label=name)
    axes[0].set(xlabel="x", ylabel="y", title="First held-out trajectory")
    axes[1].set(xlabel="forecast horizon", ylabel="state MSE")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def motion(out, seed, device, steps):
    """Synthetic video avoids permissions and file dependencies; optical flow is LK."""
    positions = np.stack([np.arange(20, 80, 2), np.full(30, 48)], -1)
    frames = []
    for x, y in positions:
        image = np.zeros((96, 96, 3), np.uint8)
        cv2.circle(image, (int(x), int(y)), 7, (0, 0, 255), -1)
        frames.append(image)
    estimates = []
    for image in frames:
        mask = cv2.inRange(image, np.array([0, 0, 100]), np.array([20, 20, 255]))
        m = cv2.moments(mask)
        if not m["m00"]:
            raise RuntimeError("Object lost")
        estimates.append([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    gray0, gray1 = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[:2]]
    corners = cv2.goodFeaturesToTrack(gray0, 20, 0.01, 3)
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, corners, None)
    flow = (nxt - corners)[status[:, 0] == 1].reshape(-1, 2)
    cv2.imwrite(str(out / "first_frame.png"), frames[0])
    estimates = np.asarray(estimates)
    np.savez(
        out / "tracking.npz",
        truth=positions,
        estimates=estimates,
        velocity=np.diff(estimates, axis=0) * 30,
        flow=flow,
    )
    return {
        "centroid_rmse_pixels": float(np.sqrt(np.mean((positions - estimates) ** 2))),
        "mean_velocity_pixels_per_second": (np.diff(estimates, axis=0).mean(0) * 30).tolist(),
        "lk_flow_pixels_per_frame": flow.mean(0).tolist(),
        "frames": len(frames),
        "fps_assumed": 30,
        "limitation": "Known color, single object, fixed camera, no occlusion",
    }


def neural(out, seed, device, steps, experiment):
    train = ball_trajectories(seed, 96, 40)
    test = ball_trajectories(seed + 10000, 24, 40)
    model = Transition().to(device)
    losses = train_supervised(
        model,
        tensor(train[:, :-1].reshape(-1, 4), device),
        tensor(train[:, 1:].reshape(-1, 4), device),
        steps,
        seed,
    )
    horizon = 24
    truth = test[:, 1 : horizon + 1]
    with torch.no_grad():
        learned = rollout(model, tensor(test[:, 0], device), horizon).cpu().numpy()
        one_step = model(tensor(test[:, :-1].reshape(-1, 4), device)).cpu().numpy()
    state = test[:, 0].copy()
    analytical, constant = [], []
    cv_state = state.copy()
    for _ in range(horizon):
        state = ball_step(state)
        analytical.append(state)
        cv_state = cv_state.copy()
        cv_state[:, :2] += 0.08 * cv_state[:, 2:]
        constant.append(cv_state)
    predictions = {
        "analytical": np.stack(analytical, 1),
        "constant_velocity": np.stack(constant, 1),
        "neural": learned,
    }
    plot_rollouts(out / "trajectories.png", truth, predictions)
    torch.save(model.state_dict(), out / "transition.pt")
    save_json(out / "training_loss.json", losses)
    return {
        "baselines": {k: trajectory_metrics(v, truth) for k, v in predictions.items()},
        "neural_one_step_mse": float(np.mean((one_step - test[:, 1:].reshape(-1, 4)) ** 2)),
        "train_episodes": 96,
        "test_episodes": 24,
        "horizon": horizon,
        "split": "Independent initial states; same physical laws",
        "experiment": experiment,
        "limitation": "One-step MLP struggles at bounce discontinuities",
    }


def latent(out, seed, device, steps, jepa=False):
    train = ball_trajectories(seed, 64, 28)
    test = ball_trajectories(seed + 10000, 24, 28)
    images = tensor(render_history(train), device)
    held = tensor(render_history(test), device)
    encoder, decoder = ImageEncoder().to(device), ImageDecoder().to(device)
    predictor = Transition(16).to(device)
    generator = torch.Generator().manual_seed(seed)
    teacher = make_teacher(encoder) if jepa else None
    optimizer = torch.optim.AdamW(
        list(encoder.parameters())
        + list(predictor.parameters())
        + ([] if jepa else list(decoder.parameters())),
        lr=0.002,
    )
    x, y = images[:, :-1].flatten(0, 1), images[:, 1:].flatten(0, 1)
    losses = []
    for _ in range(steps):
        idx = torch.randint(len(x), (128,), generator=generator).to(device)
        z = encoder(x[idx])
        if jepa:
            with torch.no_grad():
                target = teacher(y[idx])
            v, c = variance_covariance(z)
            loss = (predictor(z) - target).square().mean() + 5 * v + 0.1 * c
        else:
            target = encoder(y[idx])
            loss = (
                (decoder(z) - x[idx]).square().mean()
                + (decoder(target) - y[idx]).square().mean()
                + 0.1 * (predictor(z) - target.detach()).square().mean()
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
        optimizer.step()
        if jepa:
            update_teacher(teacher, encoder)
        losses.append(float(loss.detach()))
    encoder.eval()
    predictor.eval()
    with torch.no_grad():
        train_z = encoder(images.flatten(0, 1))
        test_z = encoder(held.flatten(0, 1))
        target_encoder = teacher if jepa else encoder
        targets = target_encoder(held[:, 1:].flatten(0, 1))
        z0 = encoder(held[:, :-1].flatten(0, 1))
        latent_mse = float((predictor(z0) - targets).square().mean())
        persistence_z = target_encoder(held[:, :-1].flatten(0, 1))
        persistence = float((persistence_z - targets).square().mean())
        scale = float(targets.var(0, unbiased=False).mean())
        # Frozen linear probe with TRAIN fit only, including a bias.
        design = torch.cat([train_z, torch.ones(len(train_z), 1, device=device)], -1)
        labels = tensor(train[:, 1:].reshape(-1, 4), device)
        # CPU double solve works consistently on macOS and CUDA, avoiding MPS linalg gaps.
        a, b = design.cpu().double(), labels.cpu().double()
        ridge = torch.eye(a.shape[1], dtype=torch.float64) * 0.01
        weights = torch.linalg.solve(a.T @ a + ridge, a.T @ b)
        probe_design = torch.cat([test_z.cpu().double(), torch.ones(len(test_z), 1)], -1)
        probe_prediction = probe_design @ weights
        probe_mse = float(
            (probe_prediction - tensor(test[:, 1:].reshape(-1, 4), "cpu")).square().mean()
        )
        horizon = 12
        predicted = rollout(predictor, encoder(held[:, 0]), horizon)
        latent_truth = target_encoder(held[:, 1 : horizon + 1].flatten(0, 1)).reshape(
            24, horizon, 16
        )
        per_horizon = (predicted - latent_truth).square().mean((0, 2)).cpu().tolist()
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "decoder": None if jepa else decoder.state_dict(),
            "teacher": None if teacher is None else teacher.state_dict(),
        },
        out / "model.pt",
    )
    save_json(out / "training_loss.json", losses)
    fig, ax = plt.subplots()
    ax.plot(per_horizon)
    ax.set(xlabel="horizon", ylabel="latent MSE", title="Frozen final target encoder")
    fig.savefig(out / "latent_rollout.png", dpi=140)
    plt.close(fig)
    return {
        "latent_one_step_mse": latent_mse,
        "latent_persistence_mse": persistence,
        "normalized_latent_mse": latent_mse / max(scale, 1e-12),
        "state_linear_probe_mse": probe_mse,
        "state_mean_baseline_mse": float(np.mean((test[:, 1:] - train[:, 1:].mean((0, 1))) ** 2)),
        "representation": representation_metrics(test_z),
        "mse_by_horizon": per_horizon,
        "jepa": jepa,
        "limitations": "Two-frame CNN, no block masking; explicit VC regularizer; synthetic data",
    }


def action_models(seed, device, steps):
    states, actions = control_trajectories(seed)
    held, held_actions = control_trajectories(seed + 10000, 24)
    x, y = (
        tensor(states[:, :-1].reshape(-1, 4), device),
        tensor(states[:, 1:].reshape(-1, 4), device),
    )
    a = tensor(actions.reshape(-1, 2), device)
    models = {}
    losses = {}
    for name, dims in [("conditioned", 2), ("unconditioned", 0)]:
        torch.manual_seed(seed)  # matching initialization distribution, different input shapes
        model = Transition(action_dim=dims).to(device)
        losses[name] = train_supervised(model, x, y, steps, seed, actions=a if dims else None)
        models[name] = model
    metrics = {}
    with torch.no_grad():
        for name, model in models.items():
            predicted = rollout(
                model,
                tensor(held[:, 0], device),
                24,
                tensor(held_actions[:, :24], device) if model.action_dim else None,
            )
            metrics[name] = trajectory_metrics(predicted.cpu().numpy(), held[:, 1:25])
    return models, metrics, losses


def planning_episodes(model, seed, device, episodes=12):
    # CEM RNG on MPS is unsupported in some versions: CPU sampling/planning fallback is explicit.
    plan_device = "cpu" if device == "mps" else device
    if plan_device != device:
        model = model.to(plan_device)
    generator = torch.Generator(device=plan_device).manual_seed(seed)
    rng = np.random.default_rng(seed + 20000)
    starts = rng.uniform(-0.5, 0.5, (episodes, 4)).astype(np.float32)
    goals = rng.uniform(-0.5, 0.5, (episodes, 2)).astype(np.float32)
    summary = {}
    for method in ["random", "zero", "pd", "mpc"]:
        errors, rewards, successes = [], [], []
        method_rng = np.random.default_rng(seed + 30000)
        for start, goal in zip(starts, goals, strict=True):
            state, reward = start.copy(), 0.0
            for _ in range(40):
                if method == "mpc":
                    action = (
                        cem_action(
                            model, tensor(state, plan_device), tensor(goal, plan_device), generator
                        )
                        .cpu()
                        .numpy()
                    )
                elif method == "pd":
                    action = proportional_derivative(state, goal)
                elif method == "random":
                    action = method_rng.uniform(-1, 1, 2).astype(np.float32)
                else:
                    action = np.zeros(2, np.float32)
                state = control_step(state, action)
                reward -= float(np.linalg.norm(state[:2] - goal))
            error = float(np.linalg.norm(state[:2] - goal))
            errors.append(error)
            rewards.append(reward)
            successes.append(error < 0.10)
        summary[method] = {
            "success_rate": float(np.mean(successes)),
            "mean_final_distance": float(np.mean(errors)),
            "mean_return": float(np.mean(rewards)),
            "episodes": episodes,
        }
    return summary


def control(out, seed, device, steps, experiment):
    models, metrics, losses = action_models(seed, device, steps)
    for name, model in models.items():
        torch.save(model.state_dict(), out / f"{name}.pt")
    save_json(out / "training_loss.json", losses)
    planning = planning_episodes(models["conditioned"], seed, device)
    # An unconditional model cannot distinguish candidate actions; evaluate its MPC consequence.
    planning_without = planning_episodes(models["unconditioned"], seed, device)
    return {
        "prediction": metrics,
        "planning": planning,
        "unconditioned_mpc": planning_without["mpc"],
        "success_definition": "final position within 0.10 at step 40",
        "protocol": "Identical held-out starts/goals; random offline data; bounded actions",
        "experiment": experiment,
        "limitation": "Fully observed linear point system; no online policy learning, contact, pixels",
    }


def value_adaptation(out, seed, device, steps):
    """Isolate representation geometry on a discrete wall world; not paper replication.

    One-hot positions, known environment transitions at evaluation, no image encoder,
    no quasimetric or MPPI. A separately learned latent predictor is measured but
    not used by the greedy planner, avoiding a false claim of full JEPA MPC.
    """
    cells = [(x, y) for y in range(7) for x in range(7) if x != 3 or y == 5]
    lookup = {cell: i for i, cell in enumerate(cells)}
    actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    transitions = np.empty((len(cells), 4), int)
    for i, (x, y) in enumerate(cells):
        for j, (dx, dy) in enumerate(actions):
            transitions[i, j] = lookup.get((x + dx, y + dy), i)
    observations = torch.eye(len(cells), device=device)
    encoder = nn.Sequential(nn.Linear(len(cells), 64), nn.Tanh(), nn.Linear(64, 8)).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=0.004)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(steps * 4):
        states = rng.integers(len(cells), size=256)
        chosen = rng.integers(4, size=256)
        goals = rng.integers(len(cells), size=256)
        future = transitions[states, chosen]
        loss = value_loss(
            encoder,
            observations[states],
            observations[future],
            observations[goals],
            torch.as_tensor(states == goals, device=device),
            gamma=0.98,
            tau=0.8,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(observations)
    predictor = Transition(8, 4).to(device)
    inputs = embeddings[:, None].expand(-1, 4, -1).reshape(-1, 8)
    targets = embeddings[transitions.reshape(-1)]
    act = torch.eye(4, device=device).repeat(len(cells), 1)
    train_supervised(predictor, inputs, targets, steps, seed, actions=act)
    with torch.no_grad():
        prediction_mse = float((predictor(inputs, act) - targets).square().mean())
    pairs = []
    eval_rng = np.random.default_rng(seed + 10000)
    for _ in range(80):
        # Opposite sides force navigation through the off-center door.
        start = lookup[(int(eval_rng.integers(0, 3)), int(eval_rng.integers(0, 7)))]
        goal = lookup[(int(eval_rng.integers(4, 7)), int(eval_rng.integers(0, 7)))]
        pairs.append((start, goal))
    positions = np.array(cells)
    results = {}
    for method in ["euclidean", "learned_value", "shortest_path_oracle"]:
        successes, lengths = [], []
        for start, goal in pairs:
            state = start
            # BFS oracle supplies exact graph distances; it is a diagnostic upper baseline.
            distance = np.full(len(cells), np.inf)
            distance[goal] = 0
            queue = [goal]
            for current in queue:
                for neighbor in transitions[current]:
                    if not np.isfinite(distance[neighbor]):
                        distance[neighbor] = distance[current] + 1
                        queue.append(int(neighbor))
            count = 0
            while state != goal and count < 40:
                candidates = transitions[state]
                if method == "learned_value":
                    scores = (
                        torch.linalg.vector_norm(embeddings[candidates] - embeddings[goal], dim=-1)
                        .cpu()
                        .numpy()
                    )
                elif method == "euclidean":
                    scores = np.linalg.norm(positions[candidates] - positions[goal], axis=-1)
                else:
                    scores = distance[candidates]
                state = int(candidates[np.argmin(scores)])
                count += 1
            successes.append(state == goal)
            lengths.append(count)
        results[method] = {
            "success_rate": float(np.mean(successes)),
            "mean_steps": float(np.mean(lengths)),
        }
    fig, ax = plt.subplots()
    goal = lookup[(6, 1)]
    costs = torch.linalg.vector_norm(embeddings - embeddings[goal], dim=-1).cpu().numpy()
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c=costs, s=150)
    fig.colorbar(scatter, ax=ax, label="learned embedding distance to goal (6,1)")
    fig.savefig(out / "value_geometry.png", dpi=140)
    plt.close(fig)
    torch.save(
        {"encoder": encoder.state_dict(), "predictor": predictor.state_dict()}, out / "model.pt"
    )
    save_json(out / "training_loss.json", losses)
    return {
        "planning": results,
        "latent_prediction_training_mse": prediction_mse,
        "representation": representation_metrics(embeddings),
        "goal_pairs": 80,
        "status": "Educational adaptation: known-transition greedy planning, training-support states",
        "not_reproduced": [
            "image observations",
            "random layouts",
            "MPPI",
            "quasimetric",
            "paper benchmark",
        ],
    }


EXPERIMENTS = {
    "01": motion,
    "02": neural,
    "03": neural,
    "04": latent,
    "05": latent,
    "07": control,
    "08": control,
    "09": value_adaptation,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["all", *EXPERIMENTS], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    run = args.output / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True, exist_ok=False)
    save_json(
        run / "manifest.json",
        {
            "args": {**vars(args), "output": str(args.output)},
            "torch": torch.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
            "results_are": "small-scale educational measurements",
        },
    )
    selected = EXPERIMENTS if args.experiment == "all" else [args.experiment]
    results = []
    for experiment in selected:
        for seed in args.seeds:
            seed_everything(seed)
            out = run / f"{experiment}_seed{seed}"
            out.mkdir()
            start = time.perf_counter()
            fn = EXPERIMENTS[experiment]
            kwargs = {"jepa": experiment == "05"} if experiment in ["04", "05"] else {}
            if experiment in ["02", "03", "07", "08"]:
                kwargs["experiment"] = experiment
            metrics = fn(out, seed, args.device, args.steps, **kwargs)
            metrics["wall_seconds"] = time.perf_counter() - start
            metrics["seed"] = seed
            save_json(out / "metrics.json", metrics)
            results.append({"experiment": experiment, **metrics})
            print(
                f"Experiment {experiment}, seed {seed}: {metrics['wall_seconds']:.2f}s", flush=True
            )
    save_json(run / "summary.json", results)
    print(f"Artifacts: {run.resolve()}")


if __name__ == "__main__":
    main()
