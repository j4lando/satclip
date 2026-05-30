# Custom 4-Band Datamodule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `S2GeoDataModule` (12/13-band, single-folder) with a `CustomGeoDataModule` that trains from scratch on 4-band imagery from two CSV+folder pairs.

**Architecture:** A new `custom_dataset.py` file provides drop-in `CustomGeoDataset` and `CustomGeoDataModule` classes that match the `S2Geo` interface exactly. `main.py` and `default.yaml` are updated to wire in the new classes and change `in_channels` from 13 to 4. No existing files are deleted; the old S2Geo path remains usable.

**Tech Stack:** PyTorch Lightning, TorchGeo (`NonGeoDataset`), Rasterio, Pandas, PyTorch, Albumentations (existing project deps)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `satclip/datamodules/custom_dataset.py` | `get_4band_transform`, `CustomGeoDataset`, `CustomGeoDataModule` |
| Create | `tests/test_custom_dataset.py` | Unit tests for all public behaviour |
| Modify | `satclip/datamodules/__init__.py:1-2` | Add export for new classes |
| Modify | `satclip/configs/default.yaml:23` | `in_channels: 13` → `in_channels: 4` |
| Modify | `satclip/configs/default.yaml:33-38` | Replace `data_dir` with five new keys |
| Modify | `satclip/main.py:6-7,119` | Swap import + `datamodule_class` to `CustomGeoDataModule` |

---

## Task 1: Write failing tests

**Files:**
- Create: `tests/test_custom_dataset.py`

- [ ] **Step 1: Create the test file**

```python
# tests/test_custom_dataset.py
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
import pytest
from unittest.mock import patch, MagicMock

# Run from repo root: pytest tests/test_custom_dataset.py -v
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "satclip"))

from datamodules.custom_dataset import (
    get_4band_transform,
    CustomGeoDataset,
    CustomGeoDataModule,
    CHECK_MIN_FILESIZE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_csv(tmp_path, folder, rows):
    """Write a CSV with columns filepath, lon, lat and an img_folder column."""
    df = pd.DataFrame(rows)
    csv_path = os.path.join(tmp_path, "index.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def _fake_rasterio_open(data_array):
    """Return a context-manager mock whose .read() returns data_array."""
    mock_src = MagicMock()
    mock_src.__enter__ = MagicMock(return_value=mock_src)
    mock_src.__exit__ = MagicMock(return_value=False)
    mock_src.read.return_value = data_array
    return mock_src


# ── get_4band_transform ───────────────────────────────────────────────────────

class TestGet4BandTransform:
    def test_divides_by_10000(self):
        transform = get_4band_transform()
        raw = np.ones((4, 8, 8), dtype=np.float32) * 10000.0
        sample = {"image": raw, "point": torch.tensor([0.0, 0.0])}
        out = transform(sample)
        assert torch.allclose(out["image"], torch.ones(4, 8, 8))

    def test_clamps_to_one(self):
        transform = get_4band_transform()
        raw = np.ones((4, 8, 8), dtype=np.float32) * 20000.0  # above 10000
        sample = {"image": raw, "point": torch.tensor([0.0, 0.0])}
        out = transform(sample)
        assert out["image"].max() <= 1.0

    def test_clamps_to_zero(self):
        transform = get_4band_transform()
        raw = np.ones((4, 8, 8), dtype=np.float32) * -5000.0
        sample = {"image": raw, "point": torch.tensor([0.0, 0.0])}
        out = transform(sample)
        assert out["image"].min() >= 0.0

    def test_output_is_tensor(self):
        transform = get_4band_transform()
        raw = np.zeros((4, 8, 8), dtype=np.float32)
        sample = {"image": raw, "point": torch.tensor([1.0, 2.0])}
        out = transform(sample)
        assert isinstance(out["image"], torch.Tensor)

    def test_point_passed_through(self):
        transform = get_4band_transform()
        pt = torch.tensor([12.34, -56.78])
        sample = {"image": np.zeros((4, 8, 8), dtype=np.float32), "point": pt}
        out = transform(sample)
        assert torch.allclose(out["point"], pt)


# ── CustomGeoDataset ──────────────────────────────────────────────────────────

class TestCustomGeoDataset:
    def _build_df(self, tmp_dir, n_good=3, n_small=1):
        """
        Build a DataFrame with n_good large files and n_small undersized files.
        File existence and size are controlled via mocking — we don't write real tifs.
        """
        rows = []
        for i in range(n_good + n_small):
            rows.append({
                "filepath": f"patch_{i}.tif",
                "lon": float(i),
                "lat": float(i) * 0.5,
                "img_folder": tmp_dir,
            })
        return pd.DataFrame(rows), n_good, n_small

    def test_len_skips_small_files(self, tmp_path):
        df, n_good, n_small = self._build_df(str(tmp_path))
        total = n_good + n_small

        def fake_exists(path):
            return True

        def fake_size(path):
            idx = int(os.path.basename(path).split("_")[1].split(".")[0])
            # first n_good files are big enough
            return CHECK_MIN_FILESIZE + 1 if idx < n_good else CHECK_MIN_FILESIZE - 1

        with patch("os.path.exists", fake_exists), patch("os.path.getsize", fake_size):
            ds = CustomGeoDataset(df)

        assert len(ds) == n_good

    def test_getitem_returns_image_and_point(self, tmp_path):
        df = pd.DataFrame([{
            "filepath": "patch_0.tif",
            "lon": 10.0,
            "lat": 20.0,
            "img_folder": str(tmp_path),
        }])
        fake_data = np.ones((4, 64, 64), dtype=np.float32)

        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=CHECK_MIN_FILESIZE + 1):
            ds = CustomGeoDataset(df)

        mock_open = _fake_rasterio_open(fake_data)
        with patch("rasterio.open", return_value=mock_open):
            sample = ds[0]

        assert "image" in sample
        assert "point" in sample
        assert sample["point"].shape == (2,)

    def test_getitem_point_values(self, tmp_path):
        df = pd.DataFrame([{
            "filepath": "patch_0.tif",
            "lon": -73.5,
            "lat": 45.2,
            "img_folder": str(tmp_path),
        }])
        fake_data = np.ones((4, 64, 64), dtype=np.float32)

        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=CHECK_MIN_FILESIZE + 1):
            ds = CustomGeoDataset(df)

        mock_open = _fake_rasterio_open(fake_data)
        with patch("rasterio.open", return_value=mock_open):
            sample = ds[0]

        assert abs(float(sample["point"][0]) - (-73.5)) < 1e-5
        assert abs(float(sample["point"][1]) - 45.2) < 1e-5

    def test_transform_applied(self, tmp_path):
        df = pd.DataFrame([{
            "filepath": "patch_0.tif",
            "lon": 0.0,
            "lat": 0.0,
            "img_folder": str(tmp_path),
        }])
        fake_data = np.ones((4, 64, 64), dtype=np.float32) * 10000.0
        sentinel = {}

        def recording_transform(sample):
            sentinel["called"] = True
            return sample

        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=CHECK_MIN_FILESIZE + 1):
            ds = CustomGeoDataset(df, transform=recording_transform)

        mock_open = _fake_rasterio_open(fake_data)
        with patch("rasterio.open", return_value=mock_open):
            ds[0]

        assert sentinel.get("called") is True

    def test_mode_points_skips_image_load(self, tmp_path):
        df = pd.DataFrame([{
            "filepath": "patch_0.tif",
            "lon": 5.0,
            "lat": 5.0,
            "img_folder": str(tmp_path),
        }])
        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=CHECK_MIN_FILESIZE + 1):
            ds = CustomGeoDataset(df, mode="points")

        with patch("rasterio.open") as mock_rio:
            sample = ds[0]
            mock_rio.assert_not_called()

        assert "image" not in sample
        assert "point" in sample


# ── CustomGeoDataModule ───────────────────────────────────────────────────────

class TestCustomGeoDataModule:
    def _write_csv(self, path, folder, n):
        rows = [{"filepath": f"p_{i}.tif", "lon": float(i), "lat": float(i)} for i in range(n)]
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    def test_setup_creates_train_and_val(self, tmp_path):
        csv1 = str(tmp_path / "c1.csv")
        csv2 = str(tmp_path / "c2.csv")
        self._write_csv(csv1, str(tmp_path), 10)
        self._write_csv(csv2, str(tmp_path), 10)

        dm = CustomGeoDataModule(
            csv1=csv1,
            csv2=csv2,
            img_folder1=str(tmp_path),
            img_folder2=str(tmp_path),
            swap_frac=0.2,
        )

        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=CHECK_MIN_FILESIZE + 1):
            dm.setup(stage="fit")

        assert len(dm.train_dataset) > 0
        assert len(dm.val_dataset) > 0

    def test_setup_no_overlap_between_splits(self, tmp_path):
        """train and val filenames must not share rows from the same source df."""
        csv1 = str(tmp_path / "c1.csv")
        csv2 = str(tmp_path / "c2.csv")
        self._write_csv(csv1, str(tmp_path), 10)
        self._write_csv(csv2, str(tmp_path), 10)

        dm = CustomGeoDataModule(
            csv1=csv1,
            csv2=csv2,
            img_folder1=str(tmp_path),
            img_folder2=str(tmp_path),
            swap_frac=0.2,
        )

        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=CHECK_MIN_FILESIZE + 1):
            dm.setup(stage="fit")

        train_files = set(dm.train_dataset.filenames)
        val_files = set(dm.val_dataset.filenames)
        # The swap logic means some cross-pollination exists but no file appears in both
        assert len(train_files & val_files) == 0

    def test_train_dataloader_returns_dataloader(self, tmp_path):
        from torch.utils.data import DataLoader
        csv1 = str(tmp_path / "c1.csv")
        csv2 = str(tmp_path / "c2.csv")
        self._write_csv(csv1, str(tmp_path), 5)
        self._write_csv(csv2, str(tmp_path), 5)

        dm = CustomGeoDataModule(
            csv1=csv1,
            csv2=csv2,
            img_folder1=str(tmp_path),
            img_folder2=str(tmp_path),
        )

        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=CHECK_MIN_FILESIZE + 1):
            dm.setup()

        assert isinstance(dm.train_dataloader(), DataLoader)
        assert isinstance(dm.val_dataloader(), DataLoader)
```

- [ ] **Step 2: Run tests — expect ImportError (module doesn't exist yet)**

Run from repo root:
```bash
cd /Users/jlando/Documents/GitHub/satclip && python -m pytest tests/test_custom_dataset.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError: No module named 'datamodules.custom_dataset'`

---

## Task 2: Implement `custom_dataset.py`

**Files:**
- Create: `satclip/datamodules/custom_dataset.py`

- [ ] **Step 1: Create the file**

```python
# satclip/datamodules/custom_dataset.py
import os
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import lightning.pytorch as pl
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datasets.geo import NonGeoDataset

CHECK_MIN_FILESIZE = 10000  # 10 kb — matches S2Geo


def get_4band_transform():
    """Normalize 4-band rasters: divide by 10000 (Sentinel-2 reflectance scale) and clamp to [0, 1]."""
    def transform(sample):
        img = torch.tensor(sample["image"])
        img = img / 10000.0
        img = torch.clamp(img, 0.0, 1.0)
        sample["image"] = img
        return sample
    return transform


class CustomGeoDataModule(pl.LightningDataModule):
    """
    Drop-in replacement for S2GeoDataModule.

    Accepts two CSVs + two image folders. Each CSV must have columns:
        filepath  — filename or relative path to .tif image within img_folder
        lon       — longitude (decimal degrees)
        lat       — latitude  (decimal degrees)

    Train/val split: swap_frac of df1 goes to val; swap_frac of df2 goes to train.
    This cross-pollinates splits while preserving dataset identity.
    """

    def __init__(
        self,
        csv1: str,
        csv2: str,
        img_folder1: str,
        img_folder2: str,
        batch_size: int = 64,
        num_workers: int = 6,
        swap_frac: float = 0.2,
        mode: str = "both",
    ):
        super().__init__()
        self.csv1 = csv1
        self.csv2 = csv2
        self.img_folder1 = img_folder1
        self.img_folder2 = img_folder2
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.swap_frac = swap_frac
        self.mode = mode
        self.train_transform = get_4band_transform()
        self.save_hyperparameters()

    def setup(self, stage="fit"):
        df1 = pd.read_csv(self.csv1)
        df2 = pd.read_csv(self.csv2)
        df1["img_folder"] = self.img_folder1
        df2["img_folder"] = self.img_folder2

        swap1 = df1.sample(frac=self.swap_frac, random_state=42)
        swap2 = df2.sample(frac=self.swap_frac, random_state=42)
        train_df = pd.concat([df1.drop(swap1.index), swap2], ignore_index=True)
        val_df = pd.concat([df2.drop(swap2.index), swap1], ignore_index=True)

        self.train_dataset = CustomGeoDataset(
            train_df, transform=self.train_transform, mode=self.mode, split="train"
        )
        self.val_dataset = CustomGeoDataset(
            val_df, transform=self.train_transform, mode=self.mode, split="val"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        raise NotImplementedError


class CustomGeoDataset(NonGeoDataset):
    """
    Drop-in replacement for S2Geo.
    Returns {"image": Tensor(4, H, W), "point": Tensor(2,)} per sample.

    Expects a DataFrame with columns: filepath, lon, lat, img_folder.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        mode: Optional[str] = "both",
        split: str = "train",
    ) -> None:
        assert mode in ["both", "points"]
        self.transform = transform
        self.mode = mode
        self.split = split
        self.filenames = []
        self.points = []
        n_skipped = 0

        for _, row in df.iterrows():
            filepath = os.path.join(row["img_folder"], row["filepath"])
            if not os.path.exists(filepath) or os.path.getsize(filepath) < CHECK_MIN_FILESIZE:
                n_skipped += 1
                continue
            self.filenames.append(filepath)
            self.points.append((row["lon"], row["lat"]))

        print(f"[{split}] loaded {len(self.filenames)} | skipped {n_skipped}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        point = torch.tensor(self.points[index])
        sample: Dict[str, Any] = {"point": point}

        if self.mode == "both":
            with rasterio.open(self.filenames[index]) as src:
                data = src.read().astype(np.float32)  # (4, H, W)
            sample["image"] = data

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Same signature as S2Geo.plot(). Displays bands [2, 1, 0] as RGB."""
        image = np.rollaxis(sample["image"].numpy(), 0, 3)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(image[:, :, [2, 1, 0]])
        ax.axis("off")
        if show_titles:
            ax.set_title(f"({sample['point'][0]:.4f}, {sample['point'][1]:.4f})")
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig

    def visualize_samples(self, n_samples: int = 8, bands=(2, 1, 0)) -> None:
        indices = np.random.choice(len(self), n_samples, replace=False)
        cols, rows = 4, int(np.ceil(n_samples / 4))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
        axes = axes.flatten()
        for i, idx in enumerate(indices):
            s = self[idx]
            img = np.rollaxis(s["image"].numpy(), 0, 3)
            rgb = img[:, :, list(bands)]
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            axes[i].imshow(rgb)
            axes[i].set_title(
                f"lon={s['point'][0]:.3f}\nlat={s['point'][1]:.3f}", fontsize=8
            )
            axes[i].axis("off")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"{self.split.capitalize()} samples", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def visualize_map(self, other=None, other_label="val") -> None:
        lons, lats = zip(*self.points)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_facecolor("#d9eaf7")
        ax.scatter(lons, lats, s=10, alpha=0.6, color="#2176ae",
                   label=f"{self.split} ({len(self)})")
        if other is not None:
            o_lons, o_lats = zip(*other.points)
            ax.scatter(o_lons, o_lats, s=10, alpha=0.8, color="#e84855",
                       label=f"{other_label} ({len(other)})")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Geographic distribution")
        ax.legend(markerscale=2)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    def visualize_coordinate_distribution(self) -> None:
        lons, lats = zip(*self.points)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.hist(lons, bins=30, color="#2176ae", alpha=0.8)
        ax1.set_title(f"{self.split.capitalize()} — Longitude")
        ax2.hist(lats, bins=30, color="#e84855", alpha=0.8)
        ax2.set_title(f"{self.split.capitalize()} — Latitude")
        plt.tight_layout()
        plt.show()
```

- [ ] **Step 2: Run tests — expect them to pass**

```bash
cd /Users/jlando/Documents/GitHub/satclip && python -m pytest tests/test_custom_dataset.py -v
```
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add satclip/datamodules/custom_dataset.py tests/test_custom_dataset.py
git commit -m "feat: add CustomGeoDataset and CustomGeoDataModule for 4-band training"
```

---

## Task 3: Update `__init__.py` to export new classes

**Files:**
- Modify: `satclip/datamodules/__init__.py`

Current contents:
```python
from .transforms import *
from .s2geo_dataset import *
```

- [ ] **Step 1: Add export line**

New contents:
```python
from .transforms import *
from .s2geo_dataset import *
from .custom_dataset import CustomGeoDataset, CustomGeoDataModule
```

- [ ] **Step 2: Verify import works from the satclip package directory**

```bash
cd /Users/jlando/Documents/GitHub/satclip/satclip && python -c "from datamodules import CustomGeoDataModule; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add satclip/datamodules/__init__.py
git commit -m "chore: export CustomGeoDataset and CustomGeoDataModule from datamodules package"
```

---

## Task 4: Update `default.yaml` — model and data sections

**Files:**
- Modify: `satclip/configs/default.yaml`

- [ ] **Step 1: Change `in_channels` from 13 to 4**

In `satclip/configs/default.yaml` line 23, change:
```yaml
  in_channels: 13
```
to:
```yaml
  in_channels: 4
```

- [ ] **Step 2: Replace the `data:` section**

Current `data:` block (lines 33–38):
```yaml
data:
  data_dir: /data/s2
  batch_size: 512
  num_workers: 8
  val_random_split_fraction: 0.1
```

Replace with:
```yaml
data:
  csv1: /path/to/coords1.csv
  csv2: /path/to/coords2.csv
  img_folder1: /path/to/images1/
  img_folder2: /path/to/images2/
  batch_size: 512
  num_workers: 8
  swap_frac: 0.2
```

- [ ] **Step 3: Verify the YAML parses cleanly**

```bash
cd /Users/jlando/Documents/GitHub/satclip && python -c "import yaml; cfg = yaml.safe_load(open('satclip/configs/default.yaml')); assert cfg['model']['in_channels'] == 4; assert 'csv1' in cfg['data']; print('YAML OK')"
```
Expected: `YAML OK`

- [ ] **Step 4: Commit**

```bash
git add satclip/configs/default.yaml
git commit -m "config: set in_channels=4, add dual-CSV data keys for custom dataset"
```

---

## Task 5: Update `main.py` to use `CustomGeoDataModule`

**Files:**
- Modify: `satclip/main.py`

- [ ] **Step 1: Swap the import on line 6**

Change:
```python
from datamodules.s2geo_dataset import S2GeoDataModule
```
to:
```python
from datamodules.custom_dataset import CustomGeoDataModule
```

- [ ] **Step 2: Swap the `datamodule_class` argument on line 119**

Change:
```python
        datamodule_class=S2GeoDataModule,
```
to:
```python
        datamodule_class=CustomGeoDataModule,
```

- [ ] **Step 3: Verify `main.py` imports cleanly (no runtime errors)**

```bash
cd /Users/jlando/Documents/GitHub/satclip/satclip && python -c "import main; print('main.py imports OK')"
```
Expected: `main.py imports OK`

- [ ] **Step 4: Run the full test suite one final time**

```bash
cd /Users/jlando/Documents/GitHub/satclip && python -m pytest tests/test_custom_dataset.py -v
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add satclip/main.py
git commit -m "feat: wire CustomGeoDataModule into LightningCLI trainer entrypoint"
```

---

## Self-Review Checklist

### Spec coverage

| Requirement | Task |
|-------------|------|
| `in_channels: 4` in config | Task 4 Step 1 |
| New file `custom_dataset.py` as drop-in for `s2geo_dataset.py` | Task 2 |
| Subclasses `NonGeoDataset` | Task 2 (class definition) |
| Returns `{"image": Tensor, "point": Tensor}` | Task 2 + test in Task 1 |
| Accepts two CSVs + two image folders | Task 2 (`CustomGeoDataModule.__init__`) |
| Row-swap train/val split | Task 2 (`CustomGeoDataModule.setup`) |
| Skip files < 10kb | Task 2 (`CustomGeoDataset.__init__`) + test |
| 4-band normalization (`/ 10000.0`) | Task 2 (`get_4band_transform`) + tests |
| `plot()` same signature as `S2Geo.plot()` | Task 2 |
| `visualize_samples`, `visualize_map`, `visualize_coordinate_distribution` | Task 2 |
| `__init__.py` export | Task 3 |
| `default.yaml` data section | Task 4 Step 2 |
| `main.py` swap | Task 5 |
| CSV format documented | Task 2 docstring |
| **Watch-out**: guard `get_pretrained_s2_train_transform` import | `__init__.py` does not import it directly; `s2geo_dataset.py` imports it from `transforms` — no change needed, both files coexist |
| **Watch-out**: do not load pretrained checkpoint | Config has no `ckpt_path`; user responsibility, documented in spec |

### Type consistency check

- `CustomGeoDataset.__init__` takes `df: pd.DataFrame` — matches all call sites in `CustomGeoDataModule.setup`
- `CustomGeoDataset.__getitem__` returns `Dict[str, Tensor]` — matches `SatCLIPLightningModule.common_step` which reads `batch["image"]` and `batch["point"]`
- `split` parameter default is `"train"` in both `__init__` and the two `CustomGeoDataset(...)` instantiation sites use explicit `split="train"` / `split="val"`
- `Any` used in `__getitem__` return type annotation and `plot()` signature — must import `Any` from `typing` ✓ (included in imports)
