import argparse
from pathlib import Path

from PIL import Image
import torch

import sys

sys.path.append(".")

from model.models import UNet
from scripts.test_functions import process_image


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_output_dir(input_dir: Path, suffix: str) -> Path:
    base = input_dir.with_name(f"{input_dir.name}{suffix}")
    if not base.exists():
        return base
    for version in range(2, 1000):
        candidate = input_dir.with_name(f"{input_dir.name}{suffix}_v{version:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not determine a unique output directory name.")


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
    )


def load_model(model_path: Path) -> UNet:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"model device: {next(model.parameters()).device}")
    return model


def process_folder(
    model: UNet,
    input_dir: Path,
    output_dir: Path,
    source_age: int,
    target_age: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in iter_images(input_dir):
        try:
            image = Image.open(image_path).convert("RGB")
            result = process_image(
                model,
                image,
                video=False,
                source_age=source_age,
                target_age=target_age,
            )
            output_name = f"{image_path.stem}{image_path.suffix}"
            result.save(output_dir / output_name)
            print(f"[OK] Processed {image_path.name} -> {output_name}")
        except Exception as exc:
            print(f"[FAIL] {image_path.name}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch face re-aging inference.")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("best_unet_model.pth"),
        help="Path to the model weights (.pth).",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--source_age",
        type=int,
        required=True,
        help="Source age of the person in all images.",
    )
    parser.add_argument(
        "--target_age",
        type=int,
        required=True,
        help="Target age to generate for all images.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_aged",
        help="Suffix for the output folder name.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")
    if not args.model_path.exists():
        raise SystemExit(f"Model file not found: {args.model_path}")
    if not (1 <= args.source_age <= 100):
        raise SystemExit("Source age must be between 1 and 100.")
    if not (1 <= args.target_age <= 100):
        raise SystemExit("Target age must be between 1 and 100.")

    output_dir = build_output_dir(args.input_dir, args.output_suffix)
    model = load_model(args.model_path)
    process_folder(model, args.input_dir, output_dir, args.source_age, args.target_age)
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
