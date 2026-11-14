"""Pre-fetch pretrained weights into the shared Hugging Face / project caches (sequential, resumable).

    python scripts/fetch_weights.py                # essential set for smoke tests + Part B
    python scripts/fetch_weights.py --all          # + DINOv2 ViT-B/14 and ConvNeXt-T

Sizes (approx.): EfficientAD teacher zip 40 MB (GitHub), DINOv2 ViT-S/14 88 MB, ResNet-18 47 MB,
WideResNet-50-2 275 MB, DINOv2 ViT-S/14-reg 88 MB, DINOv2 ViT-B/14 346 MB, ConvNeXt-T 114 MB.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ESSENTIAL = [
    "efficientad_teacher",
    "vit_small_patch14_dinov2.lvd142m",
    "resnet18.tv_in1k",
    "wide_resnet50_2.tv_in1k",
    "vit_small_patch14_reg4_dinov2.lvd142m",
]
EXTRA = ["vit_base_patch14_dinov2.lvd142m", "convnext_tiny.fb_in1k"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    import timm

    from semdefect.models.efficientad import fetch_teacher

    for name in ESSENTIAL + (EXTRA if args.all else []):
        t0 = time.time()
        for attempt in range(5):
            try:
                if name == "efficientad_teacher":
                    print("teacher:", fetch_teacher(), flush=True)
                else:
                    timm.create_model(name, pretrained=True)
                print(f"OK {name} ({time.time() - t0:.0f}s)", flush=True)
                break
            except Exception as e:  # network hiccups: retry (HF resumes partial downloads)
                print(f"retry {attempt} {name}: {type(e).__name__}: {e}", flush=True)
                time.sleep(10)


if __name__ == "__main__":
    main()
