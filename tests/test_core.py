import numpy as np
import torch

from world_models.data.synthetic import ball_step, ball_trajectories, control_step, render_history
from world_models.dynamics import Transition, rollout
from world_models.evaluation import representation_metrics
from world_models.losses import expectile, value_loss, variance_covariance
from world_models.planning import cem_action
from world_models.training import make_teacher, train_supervised, update_teacher


def test_reflection_conserves_speed_and_handles_multiple_crossings():
    state = np.array([[0.9, 0.3, 1.0, -0.2], [0.2, 0.5, 10.0, 0.0]], np.float32)
    nxt = ball_step(state, dt=0.3)
    assert np.all((nxt[:, :2] >= 0) & (nxt[:, :2] <= 1))
    np.testing.assert_allclose(nxt[0, 0], 0.8, atol=1e-6)
    np.testing.assert_allclose((nxt[:, 2:] ** 2).sum(1), (state[:, 2:] ** 2).sum(1))
    assert nxt[0, 2] < 0


def test_dataset_seed_and_ordered_history():
    a, b = ball_trajectories(8, 4, 5), ball_trajectories(8, 4, 5)
    np.testing.assert_array_equal(a, b)
    assert render_history(a).shape == (4, 4, 2, 16, 16)
    assert not np.array_equal(a, ball_trajectories(9, 4, 5))


def test_actions_change_identical_states():
    state = np.zeros(4, np.float32)
    assert not np.array_equal(control_step(state, np.ones(2)), control_step(state, -np.ones(2)))


def test_expectile_weights_and_stop_gradient():
    torch.testing.assert_close(expectile(torch.tensor([-2.0, 2.0]), 0.8), torch.tensor(2.0))
    encoder = torch.nn.Linear(3, 2)
    state, goal = torch.randn(4, 3), torch.randn(4, 3)
    future = torch.randn(4, 3, requires_grad=True)
    loss = value_loss(encoder, state, future, goal, torch.zeros(4, dtype=torch.bool))
    loss.backward()
    assert future.grad is None
    assert encoder.weight.grad is not None


def test_teacher_has_no_grad_and_ema_math():
    student = torch.nn.Linear(2, 2)
    teacher = make_teacher(student)
    old = teacher.weight.detach().clone()
    with torch.no_grad():
        student.weight.add_(1)
    update_teacher(teacher, student, 0.9)
    torch.testing.assert_close(teacher.weight, old + 0.1)
    assert not teacher.weight.requires_grad


def test_collapse_detected_and_penalized():
    z = torch.zeros(16, 8)
    variance, covariance = variance_covariance(z)
    assert variance > 0.1 and covariance == 0
    assert representation_metrics(z)["effective_rank"] == 0


def test_training_learns_action_influence_on_held_out_samples():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    x, a = torch.randn(256, 4), torch.randn(256, 2).clamp(-1, 1)
    y = x.clone()
    y[:, :2] += 0.1 * a
    model = Transition(action_dim=2)
    train_supervised(model, x, y, steps=100, actions=a)
    test_x, test_a = torch.randn(64, 4), torch.randn(64, 2).clamp(-1, 1)
    expected = test_x.clone()
    expected[:, :2] += 0.1 * test_a
    with torch.no_grad():
        error = (model(test_x, test_a) - expected).square().mean()
    assert error < 0.003


def test_cem_bounded_and_rollout_shape():
    model = Transition(action_dim=2)
    generator = torch.Generator().manual_seed(0)
    action = cem_action(
        model,
        torch.zeros(4),
        torch.ones(2),
        generator,
        horizon=3,
        candidates=16,
        elite=4,
        iterations=1,
    )
    assert action.shape == (2,) and torch.all(action.abs() <= 1)
    assert rollout(model, torch.zeros(5, 4), 3, torch.zeros(5, 3, 2)).shape == (5, 3, 4)
