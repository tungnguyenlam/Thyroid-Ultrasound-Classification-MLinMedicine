import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from resnet_classifier import ResNetClassifier
from datasets_utils import get_datasets
from utils import get_device
from test_utils import run_test, plot_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="model/resnet/best_model.pt",
        help="Path to model weights (.pt file)",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split to run testing on",
    )
    args = parser.parse_args()

    output_dir = "output/resnet"
    device = get_device()

    model = ResNetClassifier().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"Loaded weights from {args.weights}")

    _, val_dataset, test_dataset = get_datasets()
    dataset = val_dataset if args.split == "val" else test_dataset
    print(f"Testing on {args.split} set ({len(dataset)} samples)...")

    labels, preds, probs = run_test(model, dataset, device)
    plot_test(labels, preds, probs, output_dir, title_prefix="ResNet | ")


if __name__ == "__main__":
    main()
