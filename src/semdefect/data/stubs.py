"""Loader stubs for datasets that are intentionally NOT downloaded by this repo.

* MIIC - SEM dataset used by the semiconductor visual-anomaly-detection benchmark of Barusco et al.
  (arXiv:2505.07576); research-only license, must be requested from the authors and accepted personally.
* MVTec AD 2 - large download (tens of GB) under CC BY-NC-SA 4.0; registration required.

Both stubs define the expected on-disk layout so a user who obtains the data can plug it in
without touching the training code.  They raise a clear error instead of silently failing.
"""

from __future__ import annotations

from pathlib import Path

from ..utils import resolve_path


class _ManualDataset:
    name = "dataset"
    instructions = ""
    expected_root = ""

    def __init__(self, root: str | Path | None = None):
        self.root = resolve_path(root or self.expected_root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.name} not found at {self.root}.\n{self.instructions}\n"
                "This repository does not download it automatically (license/size)."
            )
        raise NotImplementedError(
            f"{self.name} loader is a stub: data found at {self.root}, implement `index()` "
            "following the layout documented in DATA.md."
        )


class MIIC(_ManualDataset):
    name = "MIIC"
    expected_root = "data/external/miic"
    instructions = (
        "MIIC is distributed under a research-only license (see arXiv:2505.07576). "
        "Request access from the authors, accept the license yourself, and extract to data/external/miic/."
    )


class MVTecAD2(_ManualDataset):
    name = "MVTec AD 2"
    expected_root = "data/external/mvtec_ad_2"
    instructions = (
        "Download MVTec AD 2 from https://www.mvtec.com/company/research/datasets/mvtec-ad-2 "
        "(registration, CC BY-NC-SA 4.0, large). Extract to data/external/mvtec_ad_2/<category>/."
    )
