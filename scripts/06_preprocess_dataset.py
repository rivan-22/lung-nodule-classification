"""
scripts/06_preprocess_dataset.py
================================

Batch preprocessing dataset penelitian LIDC-IDRI.

Membaca `dataset/05_splits/nodule_manifest.csv` (524 kandidat, sudah
ber-Fold), lalu untuk setiap sample:

    DICOM -> HU -> lung window [-1350, 150] -> [0,1]
          -> crop bbox + margin 5 mm (per-axis)
          -> resize 224x224              -> dataset/03_crops/*.png
          -> grayscale -> 3 kanal HWC
          -> densenet.preprocess_input   -> dataset/04_preprocessed/*.npy

Fungsi preprocessing diimpor dari `04_validate_preprocessing.py` agar
pipeline identik dengan yang diaudit, bukan diduplikasi.

KONSEKUENSI PEMROSESAN GANDA
----------------------------
File di `dataset/04_preprocessed/` sudah berupa OUTPUT
`tf.keras.applications.densenet.preprocess_input`. Nilai tersebut HARUS
diberikan langsung ke model. Jangan memanggil `preprocess_input` lagi.
Simpan `.npy` float32 (bukan PNG) karena output preprocessing bernilai
negatif dan tidak dapat disimpan sebagai uint8 tanpa kehilangan informasi.

`.npy` dipilih karena:
  * menyimpan float32 apa adanya -> tidak ada kuantisasi yang mengubah nilai;
  * tidak memerlukan decode gambar saat training;
  * secara eksplisit "sudah diproses" lewat nama file dan metadata.
PNG uint16 atau JPEG TIDAK dipakai: keduanya akan menguji ulang preprocessing
atau kehilangan presisi.

--------------------------------------------------------------------------
CONTOH PEMAKAIAN
--------------------------------------------------------------------------
    # dry-run: cek mapping tanpa menulis file
    python scripts/06_preprocess_dataset.py --dry-run

    # generate dengan margin terkunci
    python scripts/06_preprocess_dataset.py

    # hanya QC dataset yang sudah ada
    python scripts/06_preprocess_dataset.py --qc-only
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

# ---------------------------------------------------------------------------
# 0. KONFIGURASI TERKUNCI - jangan diubah tanpa persetujuan metodologis
# ---------------------------------------------------------------------------

WINDOW_HU_MIN = -1350.0
WINDOW_HU_MAX = 150.0
TARGET_SIZE = (224, 224)
NUM_CHANNELS = 3
CHANNEL_LAYOUT = "channels_last"
BBOX_XY_CONVENTION = "transposed"          # BBox_X = baris, BBox_Y = kolom
SLICE_SELECTION = "geometric_lower_middle"  # (n-1)//2

# MARGIN FINAL - hasil audit empiris 524 sample, disetujui penulis skripsi.
# Bukan placeholder. Bukan hasil optimasi model (model belum dilatih).
MARGIN_MM_FINAL = 5.0
MARGIN_DECISION_REASON = (
    "Nodul median 45% dari luas crop; upsample median x7.7 ke 224; hanya 5.9% "
    "sample butuh >=10x upsample (dibanding 47.7% pada 3 mm dan 79.0% pada "
    "0 mm). Konteks parenkim dan pembuluh darah sekitar nodul tetap terlihat. "
    "Clipping 0% karena jarak minimum bbox ke tepi FOV 21.9 mm."
)

DENSENET_PREPROCESS = "tf.keras.applications.densenet.preprocess_input"

DEFAULT_MANIFEST = Path("dataset/05_splits/nodule_manifest.csv")
CROP_RELDIR = Path("dataset/03_crops")
PREP_RELDIR = Path("dataset/04_preprocessed")

MANIFEST_OUT_NAME = "preprocessed_manifest.csv"
CONFIG_OUT_NAME = "preprocessing_config.json"
QC_OUT_NAME = "preprocessing_qc.csv"

IDENTITY_COLUMNS = ("PatientID", "SeriesInstanceUID", "NoduleClusterID")


def _load_audit_module(project_root: Path):
    """Impor 04_validate_preprocessing.py sebagai modul."""
    path = project_root / "scripts" / "04_validate_preprocessing.py"
    if not path.is_file():
        raise FileNotFoundError(f"Modul audit tidak ditemukan: {path}")
    spec = importlib.util.spec_from_file_location("lidc_audit_v04", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lidc_audit_v04"] = mod
    spec.loader.exec_module(mod)
    return mod


def find_project_root(marker: Path, start: Path | None = None) -> Path:
    base = Path(start).resolve() if start else Path.cwd().resolve()
    for candidate in [base, *base.parents]:
        if (candidate / marker).exists():
            return candidate
    searched = "\n".join(f"  - {p}" for p in [base, *base.parents])
    raise FileNotFoundError(
        f"PROJECT_ROOT tidak ditemukan untuk marker: {marker}\n\n"
        f"Direktori yang diperiksa:\n{searched}\n\n"
        f"Jalankan script dari dalam repository."
    )


# ---------------------------------------------------------------------------
# 1. INPUT
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={
        "PatientID": str, "SeriesInstanceUID": str, "NoduleClusterID": str,
        "GroupID": str, "Label": str})
    required = [*IDENTITY_COLUMNS, "Label", "LabelBinary", "Fold",
                "BBox_X_Min", "BBox_X_Max", "BBox_Y_Min", "BBox_Y_Max",
                "BBox_Z_Min", "BBox_Z_Max", "BBox_Z_Slices"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Kolom manifest hilang: {missing}")
    for c in IDENTITY_COLUMNS:
        if df[c].isna().any() or (df[c].astype(str).str.strip() == "").any():
            raise ValueError(f"Kolom identitas '{c}' memiliki nilai kosong.")
    if not set(df["LabelBinary"].unique()) <= {0, 1}:
        raise ValueError("LabelBinary harus hanya 0 atau 1.")
    if not set(df["Fold"].unique()) <= {1, 2, 3, 4, 5}:
        raise ValueError("Fold di luar 1..5.")
    return df


def index_dicom(raw_dir: Path) -> dict[str, list[Path]]:
    """SeriesInstanceUID -> daftar file DICOM (dibaca dari header)."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Folder DICOM tidak ditemukan: {raw_dir}")
    out: dict[str, list[Path]] = {}
    for p in raw_dir.rglob("*.dcm"):
        try:
            h = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            uid = str(getattr(h, "SeriesInstanceUID", "")).strip()
        except Exception:                                    # noqa: BLE001
            continue
        if uid:
            out.setdefault(uid, []).append(p)
    return {k: sorted(v) for k, v in out.items()}


def validate_mapping(df: pd.DataFrame, series_index: dict) -> pd.DataFrame:
    """Cek setiap baris manifest punya DICOM. TIDAK membuang sample."""
    rows = []
    for _, r in df.iterrows():
        uid = str(r.SeriesInstanceUID)
        files = series_index.get(uid)
        if not files:
            rows.append({
                "PatientID": r.PatientID, "SeriesInstanceUID": uid,
                "NoduleClusterID": r.NoduleClusterID,
                "masalah": "DICOM tidak ditemukan",
            })
            continue
        try:
            h = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
            n_rows, n_cols = int(h.Rows), int(h.Columns)
        except Exception as exc:                             # noqa: BLE001
            rows.append({"PatientID": r.PatientID, "SeriesInstanceUID": uid,
                         "NoduleClusterID": r.NoduleClusterID,
                         "masalah": f"header tidak terbaca: {exc}"})
            continue
        if not (0 <= int(r.BBox_X_Min) <= int(r.BBox_X_Max) < n_rows):
            rows.append({"PatientID": r.PatientID, "SeriesInstanceUID": uid,
                         "NoduleClusterID": r.NoduleClusterID,
                         "masalah": f"BBox_X di luar 0..{n_rows - 1}"})
        if not (0 <= int(r.BBox_Y_Min) <= int(r.BBox_Y_Max) < n_cols):
            rows.append({"PatientID": r.PatientID, "SeriesInstanceUID": uid,
                         "NoduleClusterID": r.NoduleClusterID,
                         "masalah": f"BBox_Y di luar 0..{n_cols - 1}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. OUTPUT
# ---------------------------------------------------------------------------

def safe_key(row) -> str:
    """Nama file yang dapat ditelusuri ke identitas sample."""
    return (f"{row.PatientID}__{row.SeriesInstanceUID}"
            f"__{row.NoduleClusterID}")


def write_png_gray(path: Path, img01: np.ndarray) -> None:
    """Simpan grayscale 2D dalam [0,1] sebagai PNG uint8."""
    from PIL import Image

    u8 = np.clip(np.rint(img01 * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path, optimize=True)


def rel_to_root(path: Path, project_root: Path) -> str:
    """Path RELATIF terhadap project root, separator '/' (portabel antar mesin).

    Output WAJIB berada di dalam project root agar path dapat ditelusuri.
    Direktori di luar root ditolak, bukan dipaksa jadi nama file basename.
    """
    p = Path(path).resolve()
    root = Path(project_root).resolve()
    try:
        rel = p.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Output di luar PROJECT_ROOT: {p}\n"
            f"PROJECT_ROOT: {root}\n"
            f"Path manifest harus relatif dan dapat ditelusuri. "
            f"Gunakan --crop-dir/--prep-dir di dalam repository."
        ) from exc
    return rel.as_posix()


# ---------------------------------------------------------------------------
# 3. BATCH PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess_all(df: pd.DataFrame, series_index: dict, audit,
                   crop_dir: Path, prep_dir: Path, margin_mm: float,
                   project_root: Path, write: bool = True,
                   progress_every: int = 50) -> dict:
    crop_dir.mkdir(parents=True, exist_ok=True)
    prep_dir.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    densenet_pre = tf.keras.applications.densenet.preprocess_input

    records, failures = [], []
    t0 = time.time()
    done = 0

    for uid, g in df.groupby("SeriesInstanceUID", sort=False):
        try:
            series = audit.CtSeries(uid, series_index[uid])
        except Exception as exc:                             # noqa: BLE001
            for _, r in g.iterrows():
                failures.append({"PatientID": r.PatientID,
                                 "SeriesInstanceUID": uid,
                                 "NoduleClusterID": r.NoduleClusterID,
                                 "Tahap": "buka seri",
                                 "Error": f"{type(exc).__name__}: {exc}"})
            continue

        mr, mc = audit.margin_px_pair_from_mm(
            margin_mm, series.row_spacing, series.col_spacing)

        # HU di-cache per slice supaya slice sama tidak dibaca 2x
        hu_cache: dict[int, np.ndarray] = {}
        for _, r in g.iterrows():
            nid = str(r.NoduleClusterID)
            try:
                box = audit.build_crop_box(r, series, mr, mc)
                k = box["k_abs"]
                if k not in hu_cache:
                    hu_cache[k] = series.read_hu(k)
                hu = hu_cache[k]
                crop_hu = hu[box["r0"]:box["r1"] + 1,
                              box["c0"]:box["c1"] + 1]
                if crop_hu.size == 0:
                    raise ValueError("crop kosong")
                win01 = audit.apply_window(crop_hu, WINDOW_HU_MIN,
                                          WINDOW_HU_MAX)
                res01 = audit.resize_to(win01, TARGET_SIZE)
                hwc = audit.to_three_channel(res01)
                x255 = (hwc * 255.0).astype(np.float32)
                model_input = np.asarray(
                    densenet_pre(x255, data_format="channels_last"),
                    dtype=np.float32)

                if hwc.shape != (TARGET_SIZE[0], TARGET_SIZE[1],
                                 NUM_CHANNELS):
                    raise AssertionError(f"shape 3 kanal {hwc.shape}")
                if model_input.shape != (TARGET_SIZE[0], TARGET_SIZE[1],
                                          NUM_CHANNELS):
                    raise AssertionError(f"shape model {model_input.shape}")
                if not np.isfinite(model_input).all():
                    raise ValueError("NaN/Inf pada model input")

                chans = [hwc[..., c] for c in range(NUM_CHANNELS)]
                kanal_ok = bool(np.array_equal(chans[0], chans[1])
                                and np.array_equal(chans[1], chans[2]))

                key = safe_key(r)
                crop_name = f"{key}__crop{TARGET_SIZE[0]}.png"
                prep_name = f"{key}__densenet_input.npy"
                if write:
                    write_png_gray(crop_dir / crop_name, res01)
                    np.save(prep_dir / prep_name, model_input)

                records.append({
                    "PatientID": r.PatientID,
                    "SeriesInstanceUID": str(uid),
                    "NoduleClusterID": nid,
                    "MeanMalignancy": r.MeanMalignancy,
                    "NReaders": r.NReaders,
                    "Label": r.Label,
                    "LabelBinary": int(r.LabelBinary),
                    "Fold": int(r.Fold),
                    "CropPath": rel_to_root(crop_dir / crop_name, project_root),
                    "PreprocessedPath": rel_to_root(prep_dir / prep_name,
                                                     project_root),
                    "SelectedSliceIndex": int(k),
                    "MarginMM": float(margin_mm),
                    "MarginRowPx": int(mr),
                    "MarginColPx": int(mc),
                    "ActualMarginRowMM": round(box["actual_margin_row_mm"], 3),
                    "ActualMarginColMM": round(box["actual_margin_col_mm"], 3),
                    "CropHeight": int(box["crop_h"]),
                    "CropWidth": int(box["crop_w"]),
                    "CropHeightMM": round(box["crop_h_mm"], 2),
                    "CropWidthMM": round(box["crop_w_mm"], 2),
                    "OriginalHeight": int(series.n_rows),
                    "OriginalWidth": int(series.n_cols),
                    "RowSpacingMM": float(series.row_spacing),
                    "ColSpacingMM": float(series.col_spacing),
                    "SpacingIsIsotropic": bool(series.spacing_is_isotropic),
                    "Clipped": bool(box["clipped_row"] or box["clipped_col"]),
                    "ClippedRow": bool(box["clipped_row"]),
                    "ClippedCol": bool(box["clipped_col"]),
                    "UpsampleFactor": round(
                        max(TARGET_SIZE) / max(box["crop_h"], box["crop_w"]), 3),
                    "NSlicesZ": int(box["n_slices_z"]),
                    "SelectedSliceRelIndex": int(box["k_rel"]),
                    "ChannelsIdentical": kanal_ok,
                    "ModelInputMin": float(model_input.min()),
                    "ModelInputMax": float(model_input.max()),
                    "ModelInputDtype": str(model_input.dtype),
                    "HasNaN": bool(not np.isfinite(model_input).all()),
                    "HasInf": bool(np.isinf(model_input).any()),
                    "IsEmptyCrop": bool(res01.max() <= 1e-6),
                })
            except Exception as exc:                         # noqa: BLE001
                failures.append({"PatientID": r.PatientID,
                                 "SeriesInstanceUID": uid,
                                 "NoduleClusterID": nid,
                                 "Tahap": "preprocess",
                                 "Error": f"{type(exc).__name__}: {exc}"})
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"  {done}/{len(df)} sample "
                      f"({time.time() - t0:.0f}s, {len(failures)} gagal)")

    return {"records": pd.DataFrame(records),
            "failures": pd.DataFrame(failures),
            "elapsed_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# 4. QC
# ---------------------------------------------------------------------------

def run_qc(manifest: pd.DataFrame, df_input: pd.DataFrame,
           crop_dir: Path, prep_dir: Path, project_root: Path) -> pd.DataFrame:
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"cek": name, "status": "OK" if ok else "GAGAL",
                       "detail": str(detail)})

    # Baris acuan SELALU manifest input penuh, bukan subset hasil --limit.
    n_expected = len(df_input)
    n_out = len(manifest)
    exp_label = df_input["LabelBinary"].value_counts().to_dict()
    got_label = (manifest["LabelBinary"].value_counts().to_dict()
                 if n_out else {})

    chk("jumlah input manifest penuh", n_expected == 524, n_expected)
    chk("jumlah output = jumlah input penuh", n_out == n_expected,
        f"{n_out}/{n_expected}")
    chk("tidak ada sample duplikat di manifest",
        manifest["NoduleClusterID"].duplicated().sum() == 0,
        int(manifest["NoduleClusterID"].duplicated().sum()))
    chk("setiap NoduleClusterID punya tepat 1 output",
        manifest["NoduleClusterID"].value_counts().max() == 1
        if n_out else False,
        manifest["NoduleClusterID"].value_counts().max() if n_out else 0)
    chk("semua Fold tetap 1..5",
        set(manifest["Fold"].unique()) <= {1, 2, 3, 4, 5},
        sorted(manifest["Fold"].unique().tolist()))
    chk("distribusi label sesuai input (381/143)", got_label == exp_label,
        f"input {exp_label} -> output {got_label}")
    chk("LabelBinary hanya 0/1",
        set(manifest["LabelBinary"].unique()) <= {0, 1},
        sorted(manifest["LabelBinary"].unique().tolist()))
    chk("tidak ada NaN pada model input",
        (not manifest["HasNaN"].any()) if n_out else False,
        int(manifest["HasNaN"].sum()) if n_out else "-")
    chk("tidak ada Inf pada model input",
        (not manifest["HasInf"].any()) if n_out else False,
        int(manifest["HasInf"].sum()) if n_out else "-")
    chk("tidak ada crop kosong",
        (not manifest["IsEmptyCrop"].any()) if n_out else False,
        int(manifest["IsEmptyCrop"].sum()) if n_out else "-")
    chk("kanal 1/2/3 identik (replikasi)",
        bool(manifest["ChannelsIdentical"].all()) if n_out else False,
        int((~manifest["ChannelsIdentical"]).sum()) if n_out else "-")
    chk("dtype model input float32",
        set(manifest["ModelInputDtype"]) == {"float32"} if n_out else False,
        sorted(manifest["ModelInputDtype"].unique().tolist())
        if n_out else "-")

    # file benar-benar ada?
    if n_out:
        miss_crop = sum(1 for p in manifest["CropPath"]
                        if not (project_root / p).is_file())
        miss_prep = sum(1 for p in manifest["PreprocessedPath"]
                        if not (project_root / p).is_file())
        chk("semua CropPath ada", miss_crop == 0, f"hilang {miss_crop}")
        chk("semua PreprocessedPath ada", miss_prep == 0,
            f"hilang {miss_prep}")

        # path relatif: tanpa drive letter, tanpa separator awal
        all_paths = list(manifest["CropPath"]) + list(manifest["PreprocessedPath"])
        abs_paths = [p for p in all_paths
                     if Path(p).is_absolute() or (len(p) > 2 and p[1] == ":")
                     or p.startswith("/") or "\\" in p]
        chk("semua path relatif", not abs_paths,
            f"absolut/backslash {len(abs_paths)} {abs_paths[:1]}")

        # tanpa output yatim: file di disk tidak terdaftar di manifest
        on_disk_c = {p.name for p in crop_dir.glob("*.png")} if crop_dir.is_dir() else set()
        on_disk_p = {p.name for p in prep_dir.glob("*.npy")} if prep_dir.is_dir() else set()
        listed_c = {Path(p).name for p in manifest["CropPath"]}
        listed_p = {Path(p).name for p in manifest["PreprocessedPath"]}
        chk("tidak ada file crop tanpa manifest",
            not (on_disk_c - listed_c), f"yatim {len(on_disk_c - listed_c)}")
        chk("tidak ada file preprocessed tanpa manifest",
            not (on_disk_p - listed_p), f"yatim {len(on_disk_p - listed_p)}")

        chk("margin seragam di semua sample",
            manifest["MarginMM"].nunique() == 1,
            sorted(manifest["MarginMM"].unique().tolist()))
        chk("tidak ada crop ter-clipping",
            not manifest["Clipped"].any(),
            f"{int(manifest['Clipped'].sum())} sample ter-clipping")

        # verifikasi shape nyata dari file .npy di disk
        probe = min(n_out, 40)
        bad_shape = 0
        for p in manifest["PreprocessedPath"].head(probe):
            arr = np.load(project_root / p, mmap_mode="r")
            if (arr.shape != (TARGET_SIZE[0], TARGET_SIZE[1], NUM_CHANNELS)
                    or arr.dtype != np.float32):
                bad_shape += 1
        chk(f"shape .npy (224,224,3) float32 [{probe} dicek]",
            bad_shape == 0, f"shape salah {bad_shape}")

    return pd.DataFrame(checks)


# ---------------------------------------------------------------------------
# 5. CONFIG JSON
# ---------------------------------------------------------------------------

def write_config(path: Path, project_root: Path, margin_mm: float,
                 manifest: pd.DataFrame, failures: pd.DataFrame) -> None:
    import keras
    import skimage
    import matplotlib
    import sklearn

    cfg = {
        "source_manifest": DEFAULT_MANIFEST.as_posix(),
        "dicom_root": "dataset/01_raw",
        "window_center_hu": (WINDOW_HU_MIN + WINDOW_HU_MAX) / 2.0,
        "window_width_hu": WINDOW_HU_MAX - WINDOW_HU_MIN,
        "window_hu_min": WINDOW_HU_MIN,
        "window_hu_max": WINDOW_HU_MAX,
        "window_adaptive": False,
        "scaling": "ke [0, 1] setelah window",
        "margin_mm": float(margin_mm),
        "margin_unit": "mm (dikonversi per-axis ke pixel)",
        "margin_conversion": ("ceil(mm / row_spacing) dan "
                              "ceil(mm / col_spacing) - TIDAK memakai rata-rata"),
        "margin_decision_reason": MARGIN_DECISION_REASON,
        "margin_selected_by": "audit empiris 524 sample, bukan performa model",
        "resize_size": list(TARGET_SIZE),
        "resize_method": "skimage.transform.resize order=1, mode=reflect, "
                         "anti_aliasing=True, preserve_range=True",
        "channel_layout": CHANNEL_LAYOUT,
        "num_channels": NUM_CHANNELS,
        "channel_mode": "replikasi grayscale (kanal 0 = 1 = 2)",
        "densenet_preprocessing": DENSENET_PREPROCESS,
        "densenet_preprocessing_data_format": "channels_last",
        "densenet_preprocessing_arg": "input_3ch * 255.0",
        "WARNING_preprocessing": (
            "File di dataset/04_preprocessed/ SUDAH berupa output "
            "densenet.preprocess_input. Berikan langsung ke model. "
            "JANGAN memanggil preprocess_input lagi."),
        "imagenet_resize_center_crop": "tidak dipakai (disengaja)",
        "slice_selection": f"{SLICE_SELECTION} -> (n-1)//2",
        "slice_selection_note": ("indeks lebih rendah pada urutan ascending "
                                 "proyeksi ImagePositionPatient onto normal; "
                                 "BUKAN klaim anatomis inferior"),
        "coordinate_convention": (f"{BBOX_XY_CONVENTION} "
                                  "(BBox_X=baris/row, BBox_Y=kolom/column)"),
        "crop_storage": "PNG uint8 grayscale, nilai [0,1] diskalakan ke 0..255",
        "preprocessed_storage": (".npy float32 (224,224,3) - nilai output "
                                 "preprocess_input apa adanya"),
        "random_seed": None,
        "random_seed_note": ("preprocessing deterministik, tidak ada RNG; "
                             "Fold dari manifest tidak diubah"),
        "leakage_controls": [
            "tidak ada oversampling/undersampling",
            "tidak ada augmentation",
            "tidak ada normalisasi berbasis statistik global",
            "tidak ada image selection berdasarkan label",
            "Fold dipertahankan dari nodule_manifest.csv",
        ],
        "counts": {
            "input": int(len(manifest)),
            "ok": int(len(manifest)),
            "failed": int(len(failures)),
            "benign": int((manifest["LabelBinary"] == 0).sum()) if len(manifest) else 0,
            "malignant": int((manifest["LabelBinary"] == 1).sum()) if len(manifest) else 0,
        },
        "library_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pydicom": pydicom.__version__,
            "scikit-image": skimage.__version__,
            "matplotlib": matplotlib.__version__,
            "scikit-learn": sklearn.__version__,
            "tensorflow": __import__("tensorflow").__version__,
            "keras": keras.__version__,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Batch preprocessing dataset LIDC-IDRI (crops + DenseNet input).")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--raw-dir", type=Path, default=Path("dataset/01_raw"))
    ap.add_argument("--crop-dir", type=Path, default=CROP_RELDIR)
    ap.add_argument("--prep-dir", type=Path, default=PREP_RELDIR)
    ap.add_argument("--margin-mm", type=float, default=MARGIN_MM_FINAL)
    ap.add_argument("--dry-run", action="store_true",
                    help="validasi mapping saja, tidak menulis file")
    ap.add_argument("--qc-only", action="store_true",
                    help="hanya jalankan QC pada output yang sudah ada")
    ap.add_argument("--limit", type=int, default=0,
                    help="proses N sample saja (0 = semua)")
    args = ap.parse_args(argv)

    project_root = find_project_root(args.manifest)
    audit = _load_audit_module(project_root)
    manifest_file = project_root / args.manifest
    raw_dir = project_root / args.raw_dir
    crop_dir = project_root / args.crop_dir
    prep_dir = project_root / args.prep_dir

    print("BATCH PREPROCESSING LIDC-IDRI")
    print("=" * 78)
    print(f"PROJECT_ROOT     : {project_root}")
    print(f"Manifest         : {args.manifest}")
    print(f"Raw DICOM        : {args.raw_dir}")
    print(f"Output crop      : {args.crop_dir}")
    print(f"Output preproc   : {args.prep_dir}")
    print(f"Window (terkunci): [{WINDOW_HU_MIN}, {WINDOW_HU_MAX}] HU")
    print(f"Margin final     : {args.margin_mm} mm (per-axis)")
    print(f"Konvensi bbox    : {BBOX_XY_CONVENTION}")
    print(f"Channel layout   : {CHANNEL_LAYOUT} {TARGET_SIZE} x {NUM_CHANNELS}")
    print("-" * 78)

    df = load_manifest(manifest_file)
    df_input_full = df.copy()          # acuan QC, tidak pernah dipotong --limit
    print(f"Input            : {len(df)} sample, "
          f"{df['PatientID'].nunique()} pasien, "
          f"{df['SeriesInstanceUID'].nunique()} seri")
    print(f"Label            : {df['Label'].value_counts().to_dict()}")
    print(f"Fold             : {sorted(df['Fold'].unique().tolist())}")

    if args.qc_only:
        out = prep_dir / MANIFEST_OUT_NAME
        if not out.is_file():
            raise FileNotFoundError(f"Manifest output tidak ada: {out}")
        m = pd.read_csv(out, dtype={"PatientID": str,
                                    "SeriesInstanceUID": str,
                                    "NoduleClusterID": str})
        qc = run_qc(m, df, crop_dir, prep_dir, project_root)
        print()
        print(qc.to_string(index=False))
        (prep_dir / QC_OUT_NAME).write_text(qc.to_csv(index=False),
                                            encoding="utf-8")
        return 0 if (qc["status"] == "OK").all() else 1

    t0 = time.time()
    series_index = index_dicom(raw_dir)
    print(f"\nDICOM ditemukan  : {len(series_index)} seri "
          f"({time.time() - t0:.0f}s)")

    print("\nVALIDASI MAPPING")
    problems = validate_mapping(df, series_index)
    if len(problems):
        print(problems.to_string(index=False))
        print(f"\n{len(problems)} sample bermasalah. PROSES DIHENTIKAN.")
        print("Sample TIDAK dibuang diam-diam. Perbaiki sumber data dulu.")
        return 1
    print(f"  semua {len(df)} sample terpetakan ke DICOM  [OK]")
    print(f"  seri unik dipakai : {df['SeriesInstanceUID'].nunique()}")
    print(f"  pasien unik       : {df['PatientID'].nunique()}")

    if args.dry_run:
        print("\nDRY RUN: tidak ada file yang ditulis.")
        return 0

    if args.limit:
        df = df.head(args.limit).copy()

    print(f"\nGENERATE (margin {args.margin_mm} mm)")
    res = preprocess_all(df, series_index, audit, crop_dir, prep_dir,
                         args.margin_mm, project_root, write=True)
    man, failures = res["records"], res["failures"]

    print(f"\nSelesai dalam {res['elapsed_s']}s")
    print(f"  berhasil : {len(man)}")
    print(f"  gagal    : {len(failures)}")
    if len(failures):
        print("\nSAMPLE GAGAL (tidak dibuang diam-diam):")
        print(failures.to_string(index=False))

    if len(man):
        man.to_csv(prep_dir / MANIFEST_OUT_NAME, index=False)
        write_config(prep_dir / CONFIG_OUT_NAME, project_root,
                     args.margin_mm, man, failures)
        qc = run_qc(man, df_input_full, crop_dir, prep_dir, project_root)
        qc.to_csv(prep_dir / QC_OUT_NAME, index=False)
        print("\nQUALITY CONTROL")
        print(qc.to_string(index=False))
        n_bad = int((qc["status"] != "OK").sum())
        if n_bad:
            print(f"\n{n_bad} pemeriksaan QC GAGAL.")
        return 0 if n_bad == 0 and not len(failures) else 1

    print("\nTidak ada output yang berhasil. Periksa daftar gagal di atas.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
