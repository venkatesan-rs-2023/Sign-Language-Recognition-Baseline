import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

import videotransforms
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset_all import NSLT as Dataset
from custom_models import I3DFeatureExtractor
from cbam_models import SignLanguageRecognitionModelCBAM
from sklearn.metrics import classification_report, confusion_matrix


class Compose:
    """Minimal replacement for torchvision.transforms.Compose."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def compute_topk_tp(outputs: torch.Tensor, labels: torch.Tensor, topk=(1, 5, 10)):
    """Compute top-k accuracy (sample-level)."""
    outputs_np = outputs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    results = {}
    for k in topk:
        topk_indices = np.argsort(outputs_np, axis=1)[:, -k:]
        correct = 0
        for i in range(len(labels_np)):
            if labels_np[i] in topk_indices[i]:
                correct += 1
        results[k] = 100.0 * correct / len(labels_np) if len(labels_np) else 0.0
    return results


def _strip_module_prefix(state_dict: dict) -> dict:
    return {k.replace('module.', ''): v for k, v in state_dict.items()}


def load_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """Load either weights-only (.pth with state_dict) or 'last.pth' style dict."""
    obj = torch.load(str(checkpoint_path), map_location=device)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
    else:
        state = obj

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")

    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[warn] Missing keys (showing up to 20): {missing[:20]}")
    if unexpected:
        print(f"[warn] Unexpected keys (showing up to 20): {unexpected[:20]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to weights file (e.g., best_model_*.pth or last.pth).")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Directory to write evaluation outputs. Defaults to the checkpoint's directory.")
    parser.add_argument("--root", type=str, default="data/WLASL2000",
                        help="Dataset root (relative to CWD unless absolute).")
    parser.add_argument("--test_split", type=str, default="preprocess/nslt_100.json",
                        help="Split json path (relative to CWD unless absolute).")
    parser.add_argument("--num_classes", type=int, default=100,
                        help="Number of classes for the classifier head.")
    parser.add_argument("--i3d_num_classes", type=int, default=400,
                        help="Output classes for the I3D backbone initialization (commonly 400).")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()

    if args.run_dir is None:
        run_dir = checkpoint_path.parent
    else:
        run_dir = Path(args.run_dir).expanduser().resolve()

    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Outputs ->", run_dir)

    # Model
    i3d = InceptionI3d(args.i3d_num_classes, in_channels=3)
    feature_extractor = I3DFeatureExtractor(i3d)
    model = SignLanguageRecognitionModelCBAM(feature_extractor, args.num_classes)
    model.to(device).eval()

    print(f"Loading weights from {checkpoint_path} ...")
    load_weights(model, checkpoint_path, device)

    # Data
    test_transforms = Compose([videotransforms.CenterCrop(224)])
    dataset = Dataset(args.test_split, 'test', args.root, 'rgb', test_transforms)

    pin_memory = bool(torch.cuda.is_available())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    all_outputs = []
    all_labels = []

    print("Starting evaluation...")
    with torch.no_grad():
        for inputs, labels, video_id in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Metrics
    topk = (1, 5, 10)
    topk_acc = compute_topk_tp(all_outputs, all_labels, topk=topk)
    for k in topk:
        print(f"Top-{k} Accuracy: {topk_acc[k]:.2f}%")

    preds = torch.max(all_outputs, 1)[1].numpy()
    truth = all_labels.numpy()

    report = classification_report(truth, preds, digits=4)
    cm = confusion_matrix(truth, preds)

    # Save artifacts in run_dir
    np.savetxt(run_dir / "outputs.txt", all_outputs.numpy(), fmt="%.6f")
    np.savetxt(run_dir / "labels.txt", truth, fmt="%d")
    np.save(run_dir / "confusionMatrix.npy", cm)

    with open(run_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")

    with open(run_dir / "topk_accuracy.txt", "w", encoding="utf-8") as f:
        for k in topk:
            f.write(f"Top-{k} Accuracy: {topk_acc[k]:.2f}%\n")

    print("\nDetailed Classification Report:")
    print(report)
    print(f"Saved outputs to: {run_dir}")


if __name__ == "__main__":
    main()
