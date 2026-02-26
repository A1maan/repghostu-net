import argparse
import csv
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from repghost_unet import RepGhostUNet


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_gflops(model: torch.nn.Module, x: torch.Tensor) -> float:
    activities = [ProfilerActivity.CPU]
    if x.is_cuda:
        activities.append(ProfilerActivity.CUDA)

    model.eval()
    with torch.no_grad():
        with profile(activities=activities, record_shapes=False, with_flops=True) as prof:
            _ = model(x)

    total_flops = 0
    for evt in prof.key_averages():
        if evt.flops is not None:
            total_flops += evt.flops
    return total_flops / 1e9


def save_results(output: str, rows: dict[str, object]) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        file_exists = path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(rows)
    else:
        with path.open("a") as f:
            for k, v in rows.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile RepGhostUNet params and GFLOPs")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--use-se", action="store_true")
    parser.add_argument("--bilinear", action="store_true")
    parser.add_argument("--deploy", action="store_true", help="Convert RepGhost blocks to deploy mode before profiling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="Optional output file path (.txt or .csv)")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = RepGhostUNet(
        n_channels=args.in_channels,
        n_classes=args.num_classes,
        base_ch=args.base_ch,
        use_se=args.use_se,
        bilinear=args.bilinear,
    ).to(device)

    if args.deploy:
        model.eval()
        model.convert_to_deploy()

    x = torch.randn(args.batch_size, args.in_channels, args.height, args.width, device=device)

    total_params, trainable_params = count_params(model)
    gflops = measure_gflops(model, x)
    input_shape = tuple(x.shape)

    print("Model: RepGhostUNet")
    print(f"Input shape: {input_shape}")
    print(f"Deploy mode: {args.deploy}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"GFLOPs (single forward): {gflops:.4f}")

    if args.output:
        rows = {
            "model": "RepGhostUNet",
            "input_shape": str(input_shape),
            "device": str(device),
            "deploy_mode": args.deploy,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "gflops_single_forward": round(gflops, 6),
        }
        save_results(args.output, rows)
        print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
