import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def sample_video(path, count=64):
    cap = cv2.VideoCapture(str(path))
    try:
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if length < 1:
            raise ValueError("Video has no reported frames or cannot be decoded")
        indices = np.linspace(0, length - 1, count).astype(int)
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {index}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.stack(frames), indices
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local Transformers checkpoint directory")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/vjepa_features.npz"))
    args = parser.parse_args()
    if not Path(args.model).is_dir():
        parser.error("A local checkpoint directory is required; downloads are not enabled")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Output exists; choose a fresh output path")
    from transformers import AutoModel, AutoVideoProcessor

    model = AutoModel.from_pretrained(args.model, local_files_only=True).to(args.device).eval()
    processor = AutoVideoProcessor.from_pretrained(args.model, local_files_only=True)
    frames, indices = sample_video(args.video, model.config.frames_per_clip)
    inputs = processor(frames, return_tensors="pt").to(args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    if args.device == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        hidden = model(**inputs, skip_predictor=True).last_hidden_state
    if args.device == "cuda":
        torch.cuda.synchronize()
    if args.device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    pooled = hidden.mean(1)
    # A downstream temporal comparison: same encoder on first and second clip halves.
    half_features = []
    with torch.inference_mode():
        for half in np.array_split(frames, 2):
            clip = half[np.linspace(0, len(half) - 1, len(frames)).astype(int)]
            batch = processor(clip, return_tensors="pt").to(args.device)
            half_features.append(model(**batch, skip_predictor=True).last_hidden_state.mean(1))
    similarity = float(torch.nn.functional.cosine_similarity(*half_features).item())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, tokens=hidden.cpu().numpy(), pooled=pooled.cpu().numpy(), indices=indices)
    report = {
        "token_shape": list(hidden.shape),
        "seconds_single_clip": elapsed,
        "half_clip_cosine_similarity": similarity,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else None,
        "interpretation": "Similarity is exploratory; no physical-reasoning or control benchmark",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
