import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_ROOT = PROJECT_ROOT / "comparison-models"

MODEL_SPECS = {
    "dsu_net": {
        "file": str(COMPARISON_ROOT / "dsu-net" / "DSU_Net.py"),
        "class": "DSUNet",
        "kwargs_builder": lambda a: {"n_channels": a.in_channels, "n_classes": a.num_classes},
        "input_hw": (224, 224),
    },
    "eiu_net": {
        "file": str(COMPARISON_ROOT / "eiu-net" / "scripts" / "network.py"),
        "class": "EIU_Net",
        "kwargs_builder": lambda a: {"n_channels": a.in_channels, "n_classes": a.num_classes},
        "input_hw": (224, 320),
    },
    "eseunet": {
        "file": str(COMPARISON_ROOT / "eseunet" / "ESEUNet.py"),
        "class": "ESEUNet",
        "kwargs_builder": lambda a: {"img_channels": a.in_channels, "out_channels": a.num_classes},
    },
    "mucm_net": {
        "file": str(COMPARISON_ROOT / "mucm-net" / "archs_mucm_dev.py"),
        "class": "MUCM_Net",
        "kwargs_builder": lambda a: {"num_classes": a.num_classes, "input_channels": a.in_channels, "img_size": a.height},
        "input_hw": (256, 256),
    },
    "ultralight_vm_unet": {
        "file": str(COMPARISON_ROOT / "ultralight-vm-unet" / "UltraLight_VM_UNet.py"),
        "class": "UltraLight_VM_UNet",
        "kwargs_builder": lambda a: {"num_classes": a.num_classes, "input_channels": a.in_channels},
    },
}


def dynamic_import(module_path: str, module_name: str):
    path = Path(module_path).resolve()
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def save_rows(output: str, rows: list[dict[str, Any]]) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "model_key",
        "model_class",
        "source_file",
        "input_shape",
        "device",
        "status",
        "total_params",
        "trainable_params",
        "gflops_single_forward",
        "error",
    ]

    if path.suffix.lower() == ".csv":
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        with path.open("w") as f:
            for row in rows:
                for k in headers:
                    f.write(f"{k}: {row.get(k, '')}\n")
                f.write("\n")


def patch_dsu_swin(module: Any) -> None:
    if not hasattr(module, "swin_tiny_patch4_window7_224"):
        return

    try:
        from timm.models.swin_transformer import swin_tiny_patch4_window7_224 as timm_swin
    except Exception:
        return

    def _safe_swin(*args, **kwargs):
        kwargs["pretrained"] = False
        return timm_swin(*args, **kwargs)

    module.swin_tiny_patch4_window7_224 = _safe_swin


def run_one(model_key: str, args: argparse.Namespace) -> dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    model_height, model_width = spec.get("input_hw", (args.height, args.width))
    model_args = argparse.Namespace(**vars(args))
    model_args.height = model_height
    model_args.width = model_width
    input_shape = (args.batch_size, args.in_channels, model_height, model_width)
    row = {
        "model_key": model_key,
        "model_class": spec["class"],
        "source_file": spec["file"],
        "input_shape": str(input_shape),
        "device": args.device,
        "status": "ok",
        "total_params": "",
        "trainable_params": "",
        "gflops_single_forward": "",
        "error": "",
    }

    try:
        module = dynamic_import(spec["file"], f"profile_{model_key}")
        if model_key == "dsu_net":
            patch_dsu_swin(module)

        model_cls = getattr(module, spec["class"])
        model = model_cls(**spec["kwargs_builder"](model_args)).to(torch.device(args.device))
        x = torch.randn(*input_shape, device=torch.device(args.device))

        total_params, trainable_params = count_params(model)
        gflops = measure_gflops(model, x)

        row["total_params"] = total_params
        row["trainable_params"] = trainable_params
        row["gflops_single_forward"] = round(gflops, 6)
    except Exception as e:
        err = str(e).replace("\n", " ")

        # mamba_ssm currently requires CUDA for forward in this environment.
        # Keep parameter counts when available and mark profile as partial.
        if "Expected x.is_cuda() to be true" in err:
            try:
                if "model" in locals():
                    total_params, trainable_params = count_params(model)
                    row["total_params"] = total_params
                    row["trainable_params"] = trainable_params
            except Exception:
                pass
            row["status"] = "partial"
            row["error"] = "GFLOPs not computed on CPU: this model requires CUDA for mamba_ssm forward."
        else:
            row["status"] = "error"
            row["error"] = err

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile params and GFLOPs for selected comparison models")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated keys or 'all'")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=str(COMPARISON_ROOT / "comparison_model_stats.csv"))
    args = parser.parse_args()

    if args.models.strip().lower() == "all":
        selected = list(MODEL_SPECS.keys())
    else:
        selected = [m.strip() for m in args.models.split(",") if m.strip()]

    invalid = [m for m in selected if m not in MODEL_SPECS]
    if invalid:
        raise ValueError(f"Unknown model keys: {invalid}. Valid keys: {list(MODEL_SPECS.keys())}")

    rows = [run_one(model_key, args) for model_key in selected]

    for row in rows:
        print(f"[{row['model_key']}] status={row['status']} params={row['total_params']} gflops={row['gflops_single_forward']}")
        if row["status"] != "ok":
            print(f"  error: {row['error']}")

    save_rows(args.output, rows)
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
