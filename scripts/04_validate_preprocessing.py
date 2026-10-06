"""
scripts/04_validate_preprocessing.py
====================================

Validasi parameter preprocessing untuk thesis DenseNet121 + Grad-CAM.

Notebook utama `notebooks/01_preprocessing.ipynb` TIDAK menjalankan validasi
ini secara otomatis. Validasi margin konteks dan audit konvensi bbox adalah
eksperimen terpisah yang hasilnya dipakai untuk menyusun metodologi, bukan
langkah pipeline harian.

Yang dikonten script ini (dipindah dari notebook):

  * `evaluate_convention()`      - audit BBox_X=row vs BBox_X=column
  * `build_margin_candidates()`  - kandidat margin per-axis
  * `evaluate_margin_candidates()` - tabel audit margin (reusable)
  * `summarize_by_margin()`      - agregasi per kandidat margin
  * `summarize_by_series()`      - agregasi per seri / pixel spacing

Script ini mandiri (self-contained) karena notebook tidak dapat di-import
sebagai modul Python. Fungsi inti pipeline diulang di sini agar script bisa
dijalankan tanpa membuka notebook.

ATURAN YANG TIDAK BOLEH DIUBAH oleh script ini:
  - HU_MIN = -1350, HU_MAX = 150
  - BBox_X = row, BBox_Y = column
  - middle_index = (n_slices - 1) // 2
  - input (224, 224, 3) channels_last
  - DenseNet121(weights="imagenet")
  - margin konteks: BELUM DIKUNCI (script hanya membandingkan kandidat)

Tidak ada training, split, augmentation, maupun Grad-CAM di file ini.

--------------------------------------------------------------------------
CONTOH PEMAKAIAN
--------------------------------------------------------------------------
    # audit bbox convention saja
    python scripts/04_validate_preprocessing.py --audit bbox

    # audit margin kandidat, 10 seri pertama
    python scripts/04_validate_preprocessing.py --audit margin --max-series 10

    # keduanya, simpan CSV + figure
    python scripts/04_validate_preprocessing.py --audit all --max-series 0

    # pakai anotasi hasil ekstraksi lain
    python scripts/04_validate_preprocessing.py \
        --annotations dataset/00_metadata/ct_annotated_metadata.xlsx
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. KONFIGURASI (nilai terkunci - jangan diubah)
# ---------------------------------------------------------------------------

WINDOW_HU_MIN = -1350.0
WINDOW_HU_MAX = 150.0

TARGET_SIZE = (224, 224)
NUM_CHANNELS = 3
CHANNEL_LAYOUT = "channels_last"          # DenseNet121 Keras: (H, W, C)

FRAMEWORK = "tensorflow_keras"
BBOX_XY_CONVENTION = "transposed"         # BBox_X = baris, BBox_Y = kolom
SLICE_SELECTION = "geometric_lower_middle"

MARGIN_MM_CANDIDATES = (0.0, 3.0, 5.0, 8.0)
MARGIN_PX_CANDIDATES = (0, 16, 32)
MARGIN_STATUS = "BELUM DIKUNCI"

FLAT_STD_THRESHOLD = 0.05
MARGIN_EPS = 1e-6

# Partisi HU khusus audit.
HU_AIRY_MAX = -900.0
HU_LUNG_MAX = -400.0
HU_SOFT_MAX = 150.0

LUNG_AIR = (-2000.0, -400.0)
NODULE_TISSUE = (-700.0, 200.0)

FORBIDDEN_AUDIT_COLUMNS = ("Label", "LabelCode", "MeanMalignancy",
                           "MalignancyScores")

REQUIRED_COLUMNS = (
    "PatientID", "NoduleID", "ClusterIndex", "SeriesInstanceUID",
    "BBox_X_Min", "BBox_X_Max", "BBox_Y_Min", "BBox_Y_Max",
    "BBox_Z_Min", "BBox_Z_Max",
)

DEFAULT_ANNOTATION = Path("dataset/02_annotations/pylidc_nodules_extracted.xlsx")
DEFAULT_RAW_DIR = Path("dataset/01_raw")
DEFAULT_AUDIT_DIR = Path("results/preprocessing_audit")


# ---------------------------------------------------------------------------
# 2. UTILITAS PATH
# ---------------------------------------------------------------------------

def find_project_root(marker: Path, start: Path | None = None) -> Path:
    """Cari folder repository yang memuat `marker`, tanpa path hard-code."""
    base = Path(start).resolve() if start else Path.cwd().resolve()
    for candidate in [base, *base.parents]:
        if (candidate / marker).is_file() or (candidate / marker).is_dir():
            return candidate
    searched = "\n".join(f"  - {p}" for p in [base, *base.parents])
    raise FileNotFoundError(
        f"PROJECT_ROOT tidak ditemukan untuk marker:\n  {marker}\n\n"
        f"Direktori yang diperiksa:\n{searched}\n\n"
        f"Jalankan script dari dalam repository."
    )


def discover_series(raw_dir: Path) -> dict[str, list[Path]]:
    """Petakan SeriesInstanceUID -> daftar file DICOM di bawah raw_dir."""
    import pydicom  # lokal: hanya dipakai bila raw_dir benar-benar ada

    found: dict[str, list[Path]] = {}
    if not raw_dir.is_dir():
        return found
    for dirpath, dirnames, filenames in os.walk(raw_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        files = [Path(dirpath) / f for f in filenames
                 if Path(f).suffix.lower() in {".dcm", ".dicom", ""}]
        if not files:
            continue
        try:
            head = pydicom.dcmread(files[0], stop_before_pixels=True,
                                   force=True)
            uid = str(getattr(head, "SeriesInstanceUID", "") or "").strip()
        except Exception:                                  # noqa: BLE001
            uid = ""
        if not uid:
            uid = Path(dirpath).name
        found.setdefault(uid, []).extend(files)
    return {uid: sorted(v) for uid, v in found.items()}


# ---------------------------------------------------------------------------
# 3. PEMBACA CT SERIES (inti pipeline, dibutuhkan oleh audit)
# ---------------------------------------------------------------------------

class CtSeries:
    """Pembaca satu CT series: header sekali, pixel on-demand."""

    def __init__(self, uid: str, files: list[Path]):
        import pydicom

        self.uid = str(uid)
        self.files = sorted(files)
        self.headers = [pydicom.dcmread(f, stop_before_pixels=True,
                                        force=True) for f in self.files]
        h0 = self.headers[0]
        if str(h0.Modality).strip().upper() != "CT":
            raise ValueError(f"Modality {h0.Modality!r} bukan CT ({uid}).")

        self.n_rows, self.n_cols = int(h0.Rows), int(h0.Columns)
        # DICOM PixelSpacing = [row_spacing, col_spacing]
        self.row_spacing, self.col_spacing = (float(v) for v in h0.PixelSpacing)
        if self.row_spacing <= 0 or self.col_spacing <= 0:
            raise ValueError(f"PixelSpacing tidak valid ({uid}).")
        self.slice_thickness = float(getattr(h0, "SliceThickness", 0.0) or 0.0)
        self.rescale_slope = float(getattr(h0, "RescaleSlope", 1.0))
        self.rescale_intercept = float(getattr(h0, "RescaleIntercept", 0.0))

        iop = np.asarray(h0.ImageOrientationPatient, dtype=float)
        normal = np.cross(iop[0:3], iop[3:6])
        normal = normal / np.linalg.norm(normal)
        positions = np.asarray([h.ImagePositionPatient for h in self.headers],
                               dtype=float)
        # k = urutan ascending proyeksi posisi slice onto normal.
        # k=0 = proyeksi terkecil (BUKAN "inferior" secara anatomis).
        self.slice_normal = normal
        self.projections = positions @ normal
        self.order = np.argsort(self.projections, kind="stable")
        diffs = np.diff(self.projections[self.order])
        if len(diffs) and diffs.min() < -1e-6:
            raise ValueError(f"Urutan slice tidak monoton ({uid}).")
        self.slice_spacing_mm = (float(diffs.mean()) if len(diffs)
                                 else self.slice_thickness)

    @property
    def n_slices(self) -> int:
        return len(self.order)

    @property
    def spacing_is_isotropic(self) -> bool:
        return abs(self.row_spacing - self.col_spacing) < 1e-9

    def file_for_k(self, k: int) -> Path:
        return self.files[int(self.order[k])]

    def read_hu(self, k: int) -> np.ndarray:
        import pydicom

        if not 0 <= k < self.n_slices:
            raise IndexError(f"slice k={k} di luar 0..{self.n_slices - 1}")
        ds = pydicom.dcmread(self.file_for_k(k), force=True)
        arr = ds.pixel_array
        if arr.dtype.itemsize > 2:
            arr = arr.astype(np.int16)
        return (arr.astype(np.float32) * self.rescale_slope
                + self.rescale_intercept)


# ---------------------------------------------------------------------------
# 4. FUNGSI INTI PREPROCESSING (sama dengan notebook utama)
# ---------------------------------------------------------------------------

def map_bbox(x0, x1, y0, y1, convention: str, n_rows: int, n_cols: int):
    """BBox annotation -> (baris_min, baris_max, kolom_min, kolom_max)."""
    if convention == "standard":
        r0, r1, c0, c1 = y0, y1, x0, x1
    elif convention == "transposed":
        r0, r1, c0, c1 = x0, x1, y0, y1
    else:
        raise ValueError(f"konvensi tidak dikenal: {convention!r}")
    return max(0, r0), min(n_rows - 1, r1), max(0, c0), min(n_cols - 1, c1)


def apply_window(hu: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """TAHAP 2+3: lung window lalu scaling ke [0, 1]."""
    if hu_max <= hu_min:
        raise ValueError(f"hu_max harus > hu_min, diterima {hu_min}..{hu_max}")
    return np.clip((hu - hu_min) / (hu_max - hu_min), 0.0, 1.0).astype(np.float32)


def middle_index(n_slices: int) -> int:
    """Aturan slice terkunci: (n-1)//2."""
    if n_slices < 1:
        raise ValueError(f"n_slices harus >= 1, diterima {n_slices}")
    return (n_slices - 1) // 2


def clip_range(lo: int, hi: int, limit: int) -> tuple[int, int]:
    lo = int(max(0, lo))
    hi = int(min(limit - 1, hi))
    if hi < lo:
        raise ValueError(f"Rentang [{lo}, {hi}] tidak valid terhadap {limit}.")
    return lo, hi


def margin_px_from_mm(margin_mm: float, spacing_mm: float, axis: str = "") -> int:
    """Konversi margin mm -> px pada SATU sumbu, dibulatkan ke atas."""
    if spacing_mm <= 0:
        raise ValueError(f"pixel spacing tidak valid: {spacing_mm}")
    return int(np.ceil(margin_mm / spacing_mm))


def margin_px_pair_from_mm(margin_mm: float, row_spacing: float,
                           col_spacing: float) -> tuple[int, int]:
    """Konversi margin mm -> (margin_row_px, margin_col_px), PER SUMBU.

    Rata-rata (row_spacing + col_spacing) / 2 TIDAK dipakai.
    """
    if margin_mm < 0:
        raise ValueError(f"margin_mm harus >= 0, diterima {margin_mm}")
    return (margin_px_from_mm(margin_mm, row_spacing, "row"),
            margin_px_from_mm(margin_mm, col_spacing, "col"))


def build_crop_box(row, series: CtSeries, margin_row_px: int = 0,
                   margin_col_px: int = 0) -> dict:
    """Batas crop in-plane dengan margin terpisah per sumbu."""
    r0, r1, c0, c1 = map_bbox(
        int(row.BBox_X_Min), int(row.BBox_X_Max),
        int(row.BBox_Y_Min), int(row.BBox_Y_Max),
        BBOX_XY_CONVENTION, series.n_rows, series.n_cols)
    core_r0, core_r1, core_c0, core_c1 = r0, r1, c0, c1
    margin_row_px, margin_col_px = int(margin_row_px), int(margin_col_px)
    if margin_row_px < 0 or margin_col_px < 0:
        raise ValueError(f"margin px harus >= 0, "
                         f"row={margin_row_px}, col={margin_col_px}")

    cr0, cr1 = clip_range(r0 - margin_row_px, r1 + margin_row_px, series.n_rows)
    cc0, cc1 = clip_range(c0 - margin_col_px, c1 + margin_col_px, series.n_cols)

    z0 = int(max(0, int(row.BBox_Z_Min)))
    z1 = int(min(series.n_slices - 1, int(row.BBox_Z_Max)))
    if z1 < z0:
        raise ValueError(f"BBox_Z tidak valid: {z0}..{z1}")
    n_z = z1 - z0 + 1

    nominal_h = (core_r1 - core_r0) + 1 + 2 * margin_row_px
    nominal_w = (core_c1 - core_c0) + 1 + 2 * margin_col_px
    actual_h, actual_w = cr1 - cr0 + 1, cc1 - cc0 + 1
    actual_margin_row_px = core_r0 - cr0
    actual_margin_col_px = core_c0 - cc0

    return {
        "core_r0": core_r0, "core_r1": core_r1,
        "core_c0": core_c0, "core_c1": core_c1,
        "r0": cr0, "r1": cr1, "c0": cc0, "c1": cc1,
        "z0": z0, "z1": z1, "n_slices_z": n_z,
        "k_rel": middle_index(n_z), "k_abs": z0 + middle_index(n_z),
        "margin_row_px": margin_row_px, "margin_col_px": margin_col_px,
        "actual_margin_row_px": actual_margin_row_px,
        "actual_margin_col_px": actual_margin_col_px,
        "actual_margin_row_mm": actual_margin_row_px * series.row_spacing,
        "actual_margin_col_mm": actual_margin_col_px * series.col_spacing,
        "clipped_row": bool(actual_h != nominal_h),
        "clipped_col": bool(actual_w != nominal_w),
        "crop_h": actual_h, "crop_w": actual_w,
        "crop_h_mm": actual_h * series.row_spacing,
        "crop_w_mm": actual_w * series.col_spacing,
        "core_h": core_r1 - core_r0 + 1,
        "core_w": core_c1 - core_c0 + 1,
        "clipped": bool(actual_h != nominal_h or actual_w != nominal_w),
        "rel_core_r0": core_r0 - cr0, "rel_core_r1": core_r1 - cr0,
        "rel_core_c0": core_c0 - cc0, "rel_core_c1": core_c1 - cc0,
    }


def resize_to(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from skimage.transform import resize as sk_resize

    return sk_resize(img.astype(np.float32), size, order=1, mode="reflect",
                     anti_aliasing=True, preserve_range=True).astype(np.float32)


def to_three_channel(img01: np.ndarray) -> np.ndarray:
    """TAHAP 4: (H, W) -> (H, W, 3) channels_last."""
    if img01.ndim != 2:
        raise ValueError(f"diharapkan citra 2D, diterima {img01.shape}")
    return np.repeat(img01[:, :, None], NUM_CHANNELS, axis=2).astype(np.float32)


def keras_densenet_preprocess_input(input_3ch_hwc: np.ndarray) -> np.ndarray:
    """TAHAP 5: DenseNet121 preprocess_input resmi dari Keras."""
    import tensorflow as tf

    if input_3ch_hwc.ndim != 3 or input_3ch_hwc.shape[2] != NUM_CHANNELS:
        raise ValueError(f"diharapkan (H, W, {NUM_CHANNELS}), "
                         f"diterima {input_3ch_hwc.shape}")
    x255 = (input_3ch_hwc * 255.0).astype(np.float32)
    return np.asarray(
        tf.keras.applications.densenet.preprocess_input(
            x255, data_format="channels_last"), dtype=np.float32)


def run_pipeline(row, series: CtSeries, margin_row_px: int = 0,
                 margin_col_px: int = 0,
                 volume_hu: np.ndarray | None = None,
                 size=TARGET_SIZE) -> dict:
    box = build_crop_box(row, series, margin_row_px, margin_col_px)
    k = box["k_abs"]
    hu_slice = (volume_hu[k] if volume_hu is not None else series.read_hu(k))
    sl = hu_slice[box["r0"]:box["r1"] + 1, box["c0"]:box["c1"] + 1]
    windowed = apply_window(sl, WINDOW_HU_MIN, WINDOW_HU_MAX)
    resized = resize_to(windowed, size)
    input_3ch = to_three_channel(resized)
    model_input = keras_densenet_preprocess_input(input_3ch)
    assert input_3ch.shape == (size[0], size[1], NUM_CHANNELS)
    assert model_input.shape == (size[0], size[1], NUM_CHANNELS)
    core = sl[box["rel_core_r0"]:box["rel_core_r1"] + 1,
              box["rel_core_c0"]:box["rel_core_c1"] + 1]
    return {"box": box, "hu_slice": sl, "windowed": windowed,
            "resized": resized, "input_3ch": input_3ch,
            "model_input": model_input, "core_hu": core,
            "upsample_factor": max(size) / max(box["crop_h"], box["crop_w"])}


# ---------------------------------------------------------------------------
# 5. AUDIT KONVENSI BBOX  (dipindah dari notebook)
# ---------------------------------------------------------------------------

def build_ring_mask(shape, r0, r1, c0, c1, ring_px: int = 12):
    from scipy import ndimage

    mask = np.zeros(shape, dtype=bool)
    mask[r0:r1 + 1, c0:c1 + 1] = True
    return ndimage.binary_dilation(mask, iterations=ring_px) & ~mask


def nodule_context_score(hu_slice, r0, r1, c0, c1, ring_mask):
    if r1 < r0 or c1 < c0:
        return 0.0, 0.0
    box = hu_slice[r0:r1 + 1, c0:c1 + 1]
    if box.size == 0:
        return 0.0, 0.0
    ring_values = (hu_slice[ring_mask] if ring_mask is not None
                   else np.array([], dtype=np.float32))
    ring_lung = (float(((ring_values > LUNG_AIR[0]) &
                        (ring_values < LUNG_AIR[1])).mean())
                 if ring_values.size else 0.0)
    box_solid = float(((box > NODULE_TISSUE[0]) &
                       (box < NODULE_TISSUE[1])).mean())
    return ring_lung, box_solid


def evaluate_convention(volume_hu, clusters, convention, n_rows, n_cols):
    """Nilai satu konvensi bbox untuk seluruh cluster pada seri ini.

    Skor komposit = 0.5*median + 0.3*min + 0.2*proporsi(cluster > 0.3)
    dihitung dari `ring_in_annZ` (porsi udara paru pada cincin bbox).
    """
    pad, control_stride = 3, 8
    n_slices = volume_hu.shape[0]
    ks = set()
    for _, row in clusters.iterrows():
        for k in range(max(0, int(row.BBox_Z_Min) - pad),
                       min(n_slices - 1, int(row.BBox_Z_Max) + pad) + 1):
            ks.add(k)
    ks.update(range(0, n_slices, control_stride))
    ks = sorted(ks)

    records = []
    for _, row in clusters.iterrows():
        r0, r1, c0, c1 = map_bbox(
            int(row.BBox_X_Min), int(row.BBox_X_Max),
            int(row.BBox_Y_Min), int(row.BBox_Y_Max),
            convention, n_rows, n_cols)
        z0 = max(0, min(int(row.BBox_Z_Min), n_slices - 1))
        z1 = max(z0, min(int(row.BBox_Z_Max), n_slices - 1))
        ring_mask = build_ring_mask(volume_hu.shape[1:], r0, r1, c0, c1)
        in_ann, out_ann = [], []
        for k in ks:
            res = nodule_context_score(volume_hu[k], r0, r1, c0, c1, ring_mask)
            (in_ann if z0 <= k <= z1 else out_ann).append(res)
        records.append({
            "NoduleClusterID": int(row.ClusterIndex),
            "annZ": f"{z0}-{z1}",
            "ring_in_annZ": float(np.mean([q[0] for q in in_ann])) if in_ann else 0.0,
            "box_in_annZ": float(np.mean([q[1] for q in in_ann])) if in_ann else 0.0,
            "ring_out_annZ": float(np.mean([q[0] for q in out_ann])) if out_ann else 0.0,
        })
    result = pd.DataFrame(records)
    result["skor_komposit"] = (
        0.5 * result["ring_in_annZ"].median()
        + 0.3 * result["ring_in_annZ"].min()
        + 0.2 * (result["ring_in_annZ"] > 0.3).mean())
    return result


def audit_bbox_convention(ann, local_series, uids, verbose=True):
    """Bandingkan konvensi standard vs transposed pada seri yang tersedia."""
    cache, tables = {}, {}
    for uid in uids:
        try:
            series = cache.setdefault(uid, CtSeries(uid, local_series[uid]))
            vol = np.stack([series.read_hu(k) for k in range(series.n_slices)])
        except Exception as exc:                            # noqa: BLE001
            if verbose:
                print(f"  [{uid[-12:]}] GAGAL: {type(exc).__name__}: {exc}")
            continue
        cl = ann[ann["SeriesInstanceUID"].astype(str).str.strip() == str(uid)]
        if cl.empty:
            continue
        if verbose:
            print(f"  [{uid[-12:]}] {len(cl)} cluster, {series.n_slices} slice")
        for conv in ("standard", "transposed"):
            df = evaluate_convention(vol, cl, conv, series.n_rows, series.n_cols)
            tables.setdefault(conv, []).append(df)

    if not tables:
        raise RuntimeError("Tidak ada seri yang bisa diaudit.")
    rows = {}
    for conv, dfs in tables.items():
        df = pd.concat(dfs, ignore_index=True)
        rows[conv] = {
            "median ring_lung": df["ring_in_annZ"].median(),
            "min ring_lung": df["ring_in_annZ"].min(),
            "median box_solid": df["box_in_annZ"].median(),
            "ring_lung > 0.3": f"{(df['ring_in_annZ'] > 0.3).sum()}/{len(df)}",
            "SKOR KOMPOSIT": df["skor_komposit"].mean(),
        }
    summary = pd.DataFrame(rows).T
    detected = summary["SKOR KOMPOSIT"].idxmax()
    if verbose:
        print("\nRINGKASAN AUDIT KONVENSI BBOX")
        print(summary.round(3).to_string())
        print(f"\nTerdeteksi     : {detected!r}")
        print(f"Terkonfigurasi : {BBOX_XY_CONVENTION!r}")
    return summary, detected


# ---------------------------------------------------------------------------
# 6. AUDIT MARGIN KONTEKS  (dipindah dari notebook)
# ---------------------------------------------------------------------------

def build_margin_candidates(series, margin_mm_candidates, margin_px_candidates):
    """Kandidat margin PER AXIS.

    Milimeter -> ceil(mm/row_spacing) dan ceil(mm/col_spacing), terpisah.
    Fixed-piksel memakai nilai sama pada dua sumbu.
    """
    out = []
    for m in margin_mm_candidates:
        m = float(m)
        out.append({
            "label": f"{m:g} mm", "kind": "mm", "nilai": m,
            "margin_row_px": margin_px_from_mm(m, series.row_spacing, "row"),
            "margin_col_px": margin_px_from_mm(m, series.col_spacing, "col"),
        })
    for p in margin_px_candidates:
        p = int(p)
        out.append({"label": f"{p} px", "kind": "px", "nilai": float(p),
                    "margin_row_px": p, "margin_col_px": p})
    return out


def evaluate_margin_candidates(annotations, series_index, series_uids=None,
                               margin_mm_candidates=MARGIN_MM_CANDIDATES,
                               margin_px_candidates=MARGIN_PX_CANDIDATES,
                               build_series=None, target_size=TARGET_SIZE,
                               hu_airy_max=HU_AIRY_MAX, hu_lung_max=HU_LUNG_MAX,
                               hu_soft_max=HU_SOFT_MAX, progress=None):
    """Bandingkan kandidat margin konteks. TIDAK menyimpan citra, TIDAK memakai
    label, TIDAK melakukan split. Mengembalikan (table, errors, cache)."""
    build_series = build_series or CtSeries
    if series_uids is None:
        series_uids = list(series_index)

    cache, rows, errors = {}, [], []
    n_total = len(series_uids)
    for i, uid in enumerate(series_uids, start=1):
        if progress:
            progress(i, n_total, uid)
        try:
            series = cache.setdefault(uid, build_series(uid, series_index[uid]))
        except Exception as exc:                            # noqa: BLE001
            errors.append({"SeriesInstanceUID": uid, "Tahap": "buka seri",
                           "Error": f"{type(exc).__name__}: {exc}"})
            continue

        clusters = annotations[
            annotations["SeriesInstanceUID"].astype(str).str.strip() == str(uid)]
        if clusters.empty:
            continue
        row_sp, col_sp = series.row_spacing, series.col_spacing
        candidates = build_margin_candidates(series, margin_mm_candidates,
                                             margin_px_candidates)

        for _, row in clusters.iterrows():
            for cand in candidates:
                try:
                    out = run_pipeline(row, series, cand["margin_row_px"],
                                       cand["margin_col_px"],
                                       size=target_size)
                except Exception as exc:                    # noqa: BLE001
                    errors.append({
                        "SeriesInstanceUID": uid,
                        "NoduleClusterID": int(row.ClusterIndex),
                        "Margin": cand["label"],
                        "Error": f"{type(exc).__name__}: {exc}"})
                    continue

                b, sl, core = out["box"], out["hu_slice"], out["core_hu"]
                img = out["resized"]
                fov = sl[sl > -2000]
                actual_row_mm = b["actual_margin_row_mm"]
                actual_col_mm = b["actual_margin_col_mm"]
                requested = cand["nilai"] if cand["kind"] == "mm" else None
                if requested is not None:
                    ok_row = (actual_row_mm >= requested - MARGIN_EPS
                              or b["clipped_row"])
                    ok_col = (actual_col_mm >= requested - MARGIN_EPS
                              or b["clipped_col"])
                else:
                    ok_row = ok_col = True
                soft = (sl > hu_lung_max) & (sl <= hu_soft_max)
                soft_core = (core > hu_lung_max) & (core <= hu_soft_max)
                n_ctx = max(1, sl.size - core.size)

                rows.append({
                    "SeriesInstanceUID": str(uid),
                    "NoduleClusterID": int(row.ClusterIndex),
                    "row_spacing_mm": round(row_sp, 6),
                    "col_spacing_mm": round(col_sp, 6),
                    "rows": series.n_rows, "cols": series.n_cols,
                    "Margin": cand["label"], "Margin_Jenis": cand["kind"],
                    "Margin_Nilai": cand["nilai"],
                    "margin_row_px": cand["margin_row_px"],
                    "margin_col_px": cand["margin_col_px"],
                    "actual_margin_row_mm": round(actual_row_mm, 3),
                    "actual_margin_col_mm": round(actual_col_mm, 3),
                    "actual_margin_row_px": b["actual_margin_row_px"],
                    "actual_margin_col_px": b["actual_margin_col_px"],
                    "clipped_row": bool(b["clipped_row"]),
                    "clipped_col": bool(b["clipped_col"]),
                    "margin_ok_row": bool(ok_row),
                    "margin_ok_col": bool(ok_col),
                    "crop_h_px": b["crop_h"], "crop_w_px": b["crop_w"],
                    "crop_h_mm": round(b["crop_h_mm"], 2),
                    "crop_w_mm": round(b["crop_w_mm"], 2),
                    "sisi_terpanjang_px": int(max(b["crop_h"], b["crop_w"])),
                    "core_h_px": b["core_h"], "core_w_px": b["core_w"],
                    "core_max_mm": round(max(b["core_h"] * row_sp,
                                             b["core_w"] * col_sp), 2),
                    "upsample_factor": round(out["upsample_factor"], 3),
                    "clipped": bool(b["clipped"]),
                    "n_slices_z": b["n_slices_z"], "k_abs": b["k_abs"],
                    "frac_airy": round(float((sl <= hu_airy_max).mean()), 4),
                    "frac_lung": round(float(
                        ((sl > hu_airy_max) & (sl <= hu_lung_max)).mean()), 4),
                    "frac_soft": round(float(soft.mean()), 4),
                    "frac_dense": round(float((sl > hu_soft_max).mean()), 4),
                    "frac_context_soft": round(
                        float(soft.sum() - soft_core.sum()) / n_ctx, 4),
                    "img_max": round(float(img.max()), 4),
                    "img_std": round(float(img.std()), 4),
                    "img_p99": round(float(np.percentile(img, 99)), 4),
                    "is_empty": bool(img.max() <= 1e-6),
                    "is_flat": bool(img.std() < FLAT_STD_THRESHOLD),
                    "hu_min": float(fov.min()) if fov.size else float("nan"),
                    "hu_max": float(fov.max()) if fov.size else float("nan"),
                })

    table = pd.DataFrame(rows)
    leaked = [c for c in FORBIDDEN_AUDIT_COLUMNS if c in table.columns]
    if leaked:
        raise AssertionError(f"Tabel audit tidak boleh memuat {leaked}.")
    if not table.empty:
        bad = table[~table["margin_ok_row"] | ~table["margin_ok_col"]]
        if len(bad):
            raise AssertionError(
                f"Validasi margin per-axis gagal pada {len(bad)} baris.")
    return table, errors, cache


def summarize_by_margin(df):
    """Agregasi per kandidat margin. Margin aktual per sumbu, tanpa rata-rata."""
    out = []
    for margin, g in df.groupby("Margin", sort=False):
        side_px = g["sisi_terpanjang_px"]
        out.append({
            "Margin": margin,
            "Jenis": g["Margin_Jenis"].iloc[0],
            "Nilai": g["Margin_Nilai"].iloc[0],
            "Jumlah": len(g),
            "margin_row_px": int(g["margin_row_px"].iloc[0]),
            "margin_col_px": int(g["margin_col_px"].iloc[0]),
            "margin_row_mm median": g["actual_margin_row_mm"].median(),
            "margin_col_mm median": g["actual_margin_col_mm"].median(),
            "margin_row_mm maks": g["actual_margin_row_mm"].max(),
            "margin_col_mm maks": g["actual_margin_col_mm"].max(),
            "crop_px min": side_px.min(),
            "crop_px median": side_px.median(),
            "crop_px max": side_px.max(),
            "crop_mm median": g["crop_w_mm"].median(),
            "upsample median": g["upsample_factor"].median(),
            "persen_crop_lt_32px": 100.0 * float((side_px < 32).mean()),
            "persen_crop_lt_48px": 100.0 * float((side_px < 48).mean()),
            "frac_airy median": g["frac_airy"].median(),
            "frac_context_soft median": g["frac_context_soft"].median(),
            "frac_context_soft maks": g["frac_context_soft"].max(),
            "img_std median": g["img_std"].median(),
            f"persen_near_uniform_std_lt_{FLAT_STD_THRESHOLD}":
                100.0 * g["is_flat"].mean(),
            "persen_clipped_tepi": 100.0 * g["clipped"].mean(),
            "jumlah_row_spacing": g["row_spacing_mm"].nunique(),
            "jumlah_col_spacing": g["col_spacing_mm"].nunique(),
        })
    res = pd.DataFrame(out)
    return res.sort_values(["Jenis", "Nilai"],
                           ascending=[True, True]).reset_index(drop=True)


def summarize_by_series(df):
    """Distribusi per seri dan per pasangan pixel spacing."""
    rows = []
    for (uid, rsp, csp, margin), g in df.groupby(
            ["SeriesInstanceUID", "row_spacing_mm", "col_spacing_mm", "Margin"],
            sort=False):
        side_px = g["sisi_terpanjang_px"]
        rows.append({
            "SeriesInstanceUID": uid,
            "row_spacing_mm": rsp, "col_spacing_mm": csp,
            "Margin": margin, "Jenis": g["Margin_Jenis"].iloc[0],
            "Jumlah": len(g),
            "margin_row_px": int(g["margin_row_px"].iloc[0]),
            "margin_col_px": int(g["margin_col_px"].iloc[0]),
            "margin_row_mm median": g["actual_margin_row_mm"].median(),
            "margin_col_mm median": g["actual_margin_col_mm"].median(),
            "crop_px median": side_px.median(),
            "upsample median": g["upsample_factor"].median(),
            "frac_airy median": g["frac_airy"].median(),
            "frac_context_soft median": g["frac_context_soft"].median(),
            "img_std median": g["img_std"].median(),
        })
    res = pd.DataFrame(rows)
    return res.sort_values(["row_spacing_mm", "col_spacing_mm", "Jenis", "Margin"],
                           ascending=[True, True, True, True]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def _progress(label_of):
    def _p(i, total, uid):
        print(f"  [{i}/{total}] {label_of(uid)}")
    return _p


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Validasi preprocessing LIDC (audit terpisah, bukan pipeline).")
    ap.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATION)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    ap.add_argument("--audit", choices=("bbox", "margin", "all"), default="all")
    ap.add_argument("--max-series", type=int, default=0,
                    help="0 = tanpa batas")
    ap.add_argument("--save-figure", action="store_true")
    args = ap.parse_args(argv)

    root = find_project_root(args.annotations)
    ann_file = root / args.annotations
    raw_dir = root / args.raw_dir
    audit_dir = root / args.audit_dir

    if not ann_file.is_file():
        raise FileNotFoundError(f"Annotation tidak ditemukan: {ann_file}")
    print("VALIDASI PREPROCESSING")
    print("=" * 72)
    print(f"PROJECT_ROOT  : {root}")
    print(f"Annotation    : {args.annotations}")
    print(f"Raw DICOM     : {args.raw_dir}")
    print(f"Window (kunci): [{WINDOW_HU_MIN}, {WINDOW_HU_MAX}] HU")
    print(f"Konvensi bbox : {BBOX_XY_CONVENTION}")
    print(f"Margin        : {MARGIN_STATUS} - kandidat mm {MARGIN_MM_CANDIDATES}, "
          f"px {MARGIN_PX_CANDIDATES}")
    print("-" * 72)

    ann = pd.read_excel(ann_file)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in ann.columns]
    if missing_cols:
        raise ValueError(f"Kolom annotation hilang: {missing_cols}")
    print(f"Annotation    : {len(ann)} baris, {len(ann.columns)} kolom")

    local = discover_series(raw_dir)
    ann_uids = set(ann["SeriesInstanceUID"].astype(str).str.strip())
    pool = [u for u in local if u in ann_uids]
    print(f"Seri di disk  : {len(local)}  |  bertannotation: {len(pool)}")
    if args.max_series:
        pool = pool[:args.max_series]
    if not pool:
        print("\nTidak ada seri DICOM yang cocok dengan annotation.")
        print(f"Periksa folder: {raw_dir}")
        return 1

    audit_dir.mkdir(parents=True, exist_ok=True)

    def label_of(uid):
        pid = ann.loc[ann["SeriesInstanceUID"].astype(str).str.strip() == uid,
                      "PatientID"]
        return f"{pid.iloc[0] if not pid.empty else '?'} ({uid[-12:]})"

    if args.audit in ("bbox", "all"):
        print("\n[AUDIT 1] Konvensi bbox: standard vs transposed")
        summary, detected = audit_bbox_convention(ann, local, pool[:5])
        summary.to_csv(audit_dir / "bbox_convention_audit.csv")
        print(f"\nDisimpan: "
              f"{(audit_dir / 'bbox_convention_audit.csv').relative_to(root)}")
        if detected != BBOX_XY_CONVENTION:
            print(f"\nPERINGATAN: data mendukung {detected!r}, "
                  f"pipeline memakai {BBOX_XY_CONVENTION!r}.")
            print("Laporan ke penulis skripsi; JANGAN diubah otomatis.")
        else:
            print(f"\nKonvensi {BBOX_XY_CONVENTION!r} cocok dengan data.")

    if args.audit in ("margin", "all"):
        print("\n[AUDIT 2] Kandidat margin konteks (belum dikunci)")
        table, errors, cache = evaluate_margin_candidates(
            ann, local, series_uids=pool, progress=_progress(label_of))
        if errors:
            print(f"\n{len(errors)} baris GAGAL:")
            print(pd.DataFrame(errors).head(20).to_string(index=False))
        if table.empty:
            raise RuntimeError("Tabel audit margin kosong.")
        out_csv = audit_dir / "margin_audit_table.csv"
        table.to_csv(out_csv, index=False)
        s = summarize_by_margin(table)
        s.to_csv(audit_dir / "margin_audit_summary.csv", index=False)
        print("\nRINGKASAN PER KANDIDAT MARGIN")
        print(s.round(3).to_string(index=False))
        print(f"\nDisimpan: {out_csv.relative_to(root)}")

        mm = table[table["Margin_Jenis"] == "mm"]
        nz = mm[mm["Margin_Nilai"] > 0]
        if len(nz) and (nz["margin_row_px"] == nz["margin_col_px"]).all():
            print("\nCATATAN: semua CT isotropic in-plane, sehingga margin "
                  "kedua sumbu kebetulan sama.")
        print(f"\nMargin final: {MARGIN_STATUS}. "
              f"Script ini tidak memilih nilai margin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
