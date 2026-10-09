"""Tiny random model verifies integration mechanics, not official-weight performance."""

import numpy as np
import pytest
import torch


def test_tiny_official_vjepa_encoder_and_processor():
    pytest.importorskip("transformers")
    from transformers import VJEPA2Config, VJEPA2Model, VJEPA2VideoProcessor

    config = VJEPA2Config(
        crop_size=16,
        patch_size=8,
        frames_per_clip=4,
        tubelet_size=2,
        hidden_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        pred_hidden_size=48,
        pred_num_attention_heads=4,
        pred_num_hidden_layers=1,
    )
    model = VJEPA2Model(config).eval()
    processor = VJEPA2VideoProcessor(
        size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
    )
    video = np.zeros((4, 20, 20, 3), dtype=np.uint8)
    inputs = processor(video, return_tensors="pt")
    with torch.inference_mode():
        output = model(**inputs, skip_predictor=True)
    assert output.last_hidden_state.shape == (1, 8, 96)
    assert torch.isfinite(output.last_hidden_state).all()
