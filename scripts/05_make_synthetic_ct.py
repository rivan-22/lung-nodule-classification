"""
scripts/05_make_synthetic_ct.py
==============================

Membuat CT DICOM sintetis untuk menguji implementasi preprocessing tanpa
memerlukan dataset LIDC.

**PENTING:** fixture ini BUKAN data LIDC dan BUKAN validasi metodologi.
Phantom ini hanya memastikan kode pipeline berjalan (bentuk tensor, rentang
nilai, urutan slice, margin per-axis, forward pass DenseNet121).

Phantom dirancang agar:
  * parenkim paru off-center sehingga audit konvensi bbox dapat membedakan
    BBox_X=baris dari BBox_X=kolom secara empiris;
  * `PixelSpacing` ANISOTROPIC (row != col) sehingga margin per-axis
    benar-benar berbeda dan rata-rata spacing terbukti salah.

--------------------------------------------------------------------------
    python scripts/05_make_synthetic_ct.py --out <folder>
--------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

ROW_SP, COL_SP = 0.70, 0.80          # ANISOTROPIC
N_SLICES, H, W = 9, 256, 256
UID = "1.2.826.0.1.3680043.9.7272.1.999.1.1.1"
IOP = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # normal = (0, 0, 1)
Z0, Z1 = 2, 4


def build(out_dir: Path) -> Path:
    raw = out_dir / "dataset" / "01_raw"
    ann_dir = out_dir / "dataset" / "02_annotations"
    raw.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = W / 2.0
    ly = H / 2.0 + 30.0                       # parenkim digeser ke bawah

    lung = (((yy - ly) / 40.0) ** 2 + ((xx - cx) / 35.0) ** 2) < 1.0
    body = (((yy - ly) / 105.0) ** 2 + ((xx - cx) / 110.0) ** 2) < 1.0
    nodule = (((yy - ly) / 8.0) ** 2 + ((xx - cx) / 8.0) ** 2) < 1.0

    series_dir = raw / UID
    series_dir.mkdir(parents=True, exist_ok=True)
    for old in series_dir.glob("*.dcm"):
        old.unlink()

    for k in range(N_SLICES):
        hu = np.full((H, W), -1000.0, np.float32)
        hu[body] = 40.0
        hu[lung] = -820.0
        if Z0 <= k <= Z1:
            hu[nodule] = 25.0

        px = np.clip(hu, -1024, 3071).astype(np.int16)
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = "CT"
        ds.PatientID = "SYNTH-001"
        ds.SeriesInstanceUID = UID
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesDescription = "SYNTHETIC CT (fixture, bukan LIDC)"
        ds.Rows, ds.Columns = H, W
        ds.PixelSpacing = [ROW_SP, COL_SP]
        ds.SliceThickness = 2.5
        ds.ImagePositionPatient = [0.0, 0.0, float(k) * 2.5]
        ds.ImageOrientationPatient = IOP
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.InstanceNumber = k + 1
        ds.PixelData = px.tobytes()
        ds.save_as(series_dir / f"syn{k:03d}.dcm", enforce_file_format=True)

    ann = pd.DataFrame([{
        "PatientID": "SYNTH-001", "NoduleID": "SYNTH-001.n1",
        "ClusterIndex": 0, "NReaders": 4, "MeanMalignancy": 1.0,
        "Label": "Benign", "SeriesInstanceUID": UID,
        "BBox_X_Min": int(ly - 10), "BBox_X_Max": int(ly + 10),   # baris
        "BBox_Y_Min": int(cx - 10), "BBox_Y_Max": int(cx + 10),   # kolom
        "BBox_Z_Min": Z0, "BBox_Z_Max": Z1,
        "BBox_Z_Slices": Z1 - Z0 + 1,
    }])
    ann_file = ann_dir / "pylidc_nodules_extracted.xlsx"
    ann.to_excel(ann_file, index=False)
    return ann_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True,
                    help="folder fixture (akan dibuat)")
    a = ap.parse_args()
    ann_file = build(a.out.resolve())
    print(f"DICOM sintetis : {a.out / 'dataset' / '01_raw' / UID}")
    print(f"Annotation     : {ann_file}")
    print(f"Spacing        : row={ROW_SP} col={COL_SP} (anisotropic)")
    print("CATATAN        : fixture sintetis, BUKAN data LIDC.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
