from pathlib import Path

import pandas as pd


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ORIGINAL_METADATA = (
    BASE_DIR / "ct_annotated_metadata.xlsx"
)

IDC_AUDIT = (
    BASE_DIR / "series_metadata_audit.csv"
)

OUTPUT_METADATA = (
    BASE_DIR / "ct_only_metadata.xlsx"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MEMBUAT METADATA CT ONLY")
print("=" * 70)


# ============================================================
# BACA DATA
# ============================================================

print("\nMembaca metadata awal...")

df_original = pd.read_excel(
    ORIGINAL_METADATA
)

print(
    f"Baris metadata awal : {len(df_original)}"
)


print("\nMembaca hasil audit IDC...")

df_idc = pd.read_csv(
    IDC_AUDIT
)

print(
    f"Baris hasil audit IDC : {len(df_idc)}"
)


# ============================================================
# NORMALISASI UID
# ============================================================

df_original["Series Instance UID"] = (
    df_original["Series Instance UID"]
    .astype(str)
    .str.strip()
)

df_idc["SeriesInstanceUID"] = (
    df_idc["SeriesInstanceUID"]
    .astype(str)
    .str.strip()
)


# ============================================================
# AMBIL HANYA CT DARI IDC
# ============================================================

df_ct_idc = df_idc[
    df_idc["Modality"]
    .astype(str)
    .str.strip()
    .str.upper()
    == "CT"
].copy()


print(
    f"\nSeries CT terverifikasi IDC : "
    f"{len(df_ct_idc)}"
)


# ============================================================
# UID CT
# ============================================================

ct_uids = set(
    df_ct_idc["SeriesInstanceUID"]
)


# ============================================================
# FILTER METADATA AWAL
# ============================================================

df_ct = df_original[
    df_original["Series Instance UID"]
    .isin(ct_uids)
].copy()


# ============================================================
# VALIDASI
# ============================================================

print("\n" + "=" * 70)
print("VALIDASI")
print("=" * 70)

print(
    f"Metadata awal          : {len(df_original)}"
)

print(
    f"CT terverifikasi IDC   : {len(df_ct_idc)}"
)

print(
    f"CT hasil filter        : {len(df_ct)}"
)

print(
    f"UID unik hasil         : "
    f"{df_ct['Series Instance UID'].nunique()}"
)


if len(df_ct) != 306:

    raise ValueError(
        "Jumlah CT hasil filter tidak sama dengan "
        "306. Periksa kembali metadata."
    )


# ============================================================
# TAMBAHKAN METADATA IDC
# ============================================================

df_idc_small = df_ct_idc[
    [
        "SeriesInstanceUID",
        "PatientID",
        "StudyInstanceUID",
        "Modality",
        "SeriesDescription",
        "SOPClassUID",
        "series_size_MB",
    ]
].copy()


df_idc_small = df_idc_small.rename(
    columns={
        "SeriesInstanceUID":
            "IDC SeriesInstanceUID",

        "PatientID":
            "IDC PatientID",

        "StudyInstanceUID":
            "IDC StudyInstanceUID",

        "Modality":
            "IDC Modality",

        "SeriesDescription":
            "IDC SeriesDescription",

        "SOPClassUID":
            "IDC SOPClassUID",

        "series_size_MB":
            "IDC Series Size MB",
    }
)


# ============================================================
# MERGE
# ============================================================

df_ct = df_ct.merge(
    df_idc_small,
    left_on="Series Instance UID",
    right_on="IDC SeriesInstanceUID",
    how="left"
)


# ============================================================
# VALIDASI AKHIR
# ============================================================

non_ct = df_ct[
    df_ct["IDC Modality"]
    .astype(str)
    .str.upper()
    != "CT"
]


if len(non_ct) > 0:

    raise ValueError(
        "Masih terdapat metadata non-CT!"
    )


# ============================================================
# SIMPAN
# ============================================================

df_ct.to_excel(
    OUTPUT_METADATA,
    index=False
)


# ============================================================
# HASIL
# ============================================================

print("\n" + "=" * 70)
print("SELESAI")
print("=" * 70)

print(
    f"\nFile metadata CT-only:"
)

print(
    OUTPUT_METADATA
)

print(
    f"\nJumlah CT : {len(df_ct)}"
)

print(
    f"UID unik  : "
    f"{df_ct['Series Instance UID'].nunique()}"
)

print(
    "\nIDC Modality:"
)

print(
    df_ct["IDC Modality"].value_counts()
)

print("=" * 70)