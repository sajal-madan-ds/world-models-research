"""Forecast with a saved four-state toy transition checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from world_models.dynamics import Transition, rollout

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--state", nargs=4, type=float, default=[0.5, 0.5, 0.2, -0.1])
parser.add_argument("--horizon", type=int, default=12)
parser.add_argument("--action", nargs=2, type=float)
args = parser.parse_args()
if args.horizon < 1:
    parser.error("horizon must be positive")
model = Transition(action_dim=2 if args.action is not None else 0)
model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
model.eval()
state = torch.tensor([args.state], dtype=torch.float32)
actions = None if args.action is None else torch.tensor([[args.action] * args.horizon])
with torch.inference_mode():
    predictions = rollout(model, state, args.horizon, actions)
print(json.dumps({"predicted_states": predictions[0].tolist(), "units": "toy normalized state"}))
