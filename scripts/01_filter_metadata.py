from pathlib import Path
import pandas as pd


# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

METADATA_FILE = BASE_DIR / "metadata.xlsx"
OUTPUT_FILE = BASE_DIR / "ct_annotated_metadata.xlsx"


# ============================================================
# 2. CEK FILE
# ============================================================

print("=" * 60)
print("CEK METADATA LIDC-IDRI")
print("=" * 60)

print(f"Folder kerja : {BASE_DIR}")
print(f"Metadata     : {METADATA_FILE}")

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"File metadata tidak ditemukan:\n{METADATA_FILE}"
    )


# ============================================================
# 3. BACA EXCEL
# ============================================================

df = pd.read_excel(METADATA_FILE)

print("\nMetadata berhasil dibaca.")
print(f"Jumlah baris awal : {len(df):,}")

print("\nKolom:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i:2}. {col}")


# ============================================================
# 4. NORMALISASI KOLOM YANG DIPAKAI
# ============================================================

required_columns = [
    "Patient ID",
    "Series Instance UID",
    "Modality",
    "Annotations Flag",
    "Image Count",
    "File Size",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nKolom berikut tidak ditemukan:\n{missing_columns}"
    )


# ============================================================
# 5. FILTER MODALITY CT
# ============================================================

df_ct = df[
    df["Modality"]
    .astype(str)
    .str.strip()
    .str.upper()
    == "CT"
].copy()

print("\n" + "=" * 60)
print("FILTER MODALITY CT")
print("=" * 60)

print(f"Jumlah data CT : {len(df_ct):,}")


# ============================================================
# 6. FILTER DATA YANG MEMILIKI ANNOTATION
# ============================================================

print("\nNilai asli pada kolom 'Annotations Flag':")
print(df_ct["Annotations Flag"].value_counts(dropna=False))

# Karena pandas membaca TRUE dari Excel sebagai 1.0
df_ct_annotated = df_ct[
    df_ct["Annotations Flag"] == 1.0
].copy()

print("\n" + "=" * 60)
print("FILTER CT + ANNOTATION")
print("=" * 60)

print(
    f"Jumlah CT dengan annotation : "
    f"{len(df_ct_annotated):,}"
)

# ============================================================
# 7. CEK INFORMASI DATA
# ============================================================

print("\nInformasi dataset:")

print(
    f"Jumlah Patient ID unik     : "
    f"{df_ct_annotated['Patient ID'].nunique():,}"
)

print(
    f"Jumlah Series UID unik     : "
    f"{df_ct_annotated['Series Instance UID'].nunique():,}"
)

print(
    f"Total Image Count          : "
    f"{pd.to_numeric(df_ct_annotated['Image Count'], errors='coerce').sum():,}"
)


# ============================================================
# 8. SIMPAN HASIL FILTER
# ============================================================

df_ct_annotated.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("SELESAI")
print("=" * 60)

print(f"File hasil disimpan di:")
print(OUTPUT_FILE)