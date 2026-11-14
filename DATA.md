# Data and pretrained weights

Raw data lives under `data/raw/` and caches under `data/cache/` (both gitignored). Split files in
`data/splits/` are small, deterministic and meant to be committed.

## Carinthia-S (used)

| Field | Value |
|---|---|
| Record | Zenodo 10.5281/zenodo.16895427 — "Carinthia-S dataset", v. published 2025-08-20 |
| Authors | Corinna Kofler, Vahidin Hasić |
| License | CC BY 4.0 |
| Download | `https://zenodo.org/api/records/16895427/files/data.zip/content` (+ `carinthia-s_dataset.html`) |
| Size | `data.zip` 139,225,942 bytes (133 MB); unpacked 146 MB (4,591 JPEG + 4,591 PNG + CSV) |
| Checksums | `data.zip` md5 `3a41242574635cf1302fab15accb7730` (matches Zenodo), sha256 `e7cc8507657611f89db33345d50fc7951e8784045684888dc997ac5298b7a3a2`; `carinthia-s_dataset.html` md5 `4d9ca3ed51dfc97444939f742496d2d9` |
| Derived from | Carinthia dataset, Zenodo 10.5281/zenodo.10715190 (Kofler et al., 2024, CC BY 4.0) — same 4,591 images without masks; **not downloaded** (Carinthia-S is a superset) |
| Local path | `data/raw/carinthia_s/data/{carinthia-s.csv, images/, masks/}` |

Citation (from the Zenodo record):

```
Kofler, C., & Hasić, V. (2025). Carinthia-S dataset [Data set]. Zenodo. https://doi.org/10.5281/zenodo.16895427
Kofler, C. et al. (2024). Carinthia dataset [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10715190
```

Reproduce: `python scripts/download_data.py` (downloads, verifies md5, unzips).

### Derived caches (gitignored, rebuildable)

| Path | Content | Size |
|---|---|---|
| `data/cache/arrays/img_{224,256,280,448}_area_*.npy` | cropped + area-resized uint8 images (memmap) | 230 / 301 / 360 / 922 MB |
| `data/cache/arrays/mask_{224,256,280,448}_*.npy` | binarised masks at the same sizes | same as images |
| `data/cache/features/*.npz` | frozen DINOv2 global descriptors for few-shot probes | < 20 MB each |
| `data/cache/pretrained/efficientad/` | EfficientAD distilled PDN teachers (see below) | 40 MB zip |

## Pretrained weights (downloaded into package caches)

| Model | Source | License |
|---|---|---|
| DINOv2 ViT-S/14, ViT-B/14, ViT-S/14-reg4 | timm / Hugging Face `timm/vit_*_patch14_*dinov2.lvd142m` | Apache-2.0 (Meta DINOv2) |
| WideResNet-50-2, ResNet-18 (torchvision ImageNet) | `timm/wide_resnet50_2.tv_in1k`, `timm/resnet18.tv_in1k` | BSD-3-Clause (torchvision) |
| ConvNeXt-T | `timm/convnext_tiny.fb_in1k` | Apache-2.0 |
| EfficientAD-S distilled teacher | anomalib release asset `efficientad_pretrained_weights.zip` (github.com/open-edge-platform/anomalib) | Apache-2.0 |
| TensorRT 10.8 public headers (C++ runner) | github.com/NVIDIA/TensorRT tag v10.8.0 `include/` → `cpp/third_party/tensorrt/include` | Apache-2.0 |

## Not downloaded (stubs only — `src/semdefect/data/stubs.py`)

| Dataset | Why not | How to add |
|---|---|---|
| MIIC (used by the semiconductor VAD benchmark, arXiv:2505.07576) | research-only license; access must be requested and accepted personally | request from the authors, extract to `data/external/miic/` |
| MVTec AD 2 | large (tens of GB), CC BY-NC-SA 4.0, registration | download from mvtec.com, extract to `data/external/mvtec_ad_2/<category>/` |
