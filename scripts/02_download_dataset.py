from pathlib import Path

import pandas as pd
from idc_index import IDCClient


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

METADATA_FILE = BASE_DIR / "ct_only_metadata.xlsx"

DOWNLOAD_DIR = BASE_DIR / "raw" / "lidc_idri"


# ------------------------------------------------------------
# True  = hanya cek dan hitung ukuran, TIDAK download
# False = download CT
# ------------------------------------------------------------

DRY_RUN = False


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DOWNLOAD CT ONLY - LIDC-IDRI")
print("=" * 70)


# ============================================================
# BACA METADATA
# ============================================================

print("\nMembaca metadata...")

df = pd.read_excel(METADATA_FILE)

print(f"Jumlah baris Excel : {len(df)}")


# ============================================================
# AMBIL SERIES UID
# ============================================================

series_uids = (
    df["Series Instance UID"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)


print(
    f"Jumlah Series UID unik : {len(series_uids)}"
)


if len(series_uids) != 306:

    raise ValueError(
        f"Series UID yang ditemukan bukan 385, "
        f"tetapi {len(series_uids)}."
    )


# ============================================================
# CEK KOLOM MODALITY DI EXCEL
# ============================================================

if "Modality" in df.columns:

    excel_modalities = (
        df["Modality"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .value_counts()
    )

    print("\nModality dari Excel:")

    for modality, count in excel_modalities.items():

        print(
            f"  {modality:10s}: {count}"
        )

else:

    print(
        "\nWARNING: Kolom 'Modality' tidak ditemukan "
        "di Excel."
    )


# ============================================================
# FOLDER DOWNLOAD
# ============================================================

DOWNLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(
    f"\nFolder download : {DOWNLOAD_DIR}"
)


# ============================================================
# IDC CLIENT
# ============================================================

print("\nMembuka IDC Index...")

client = IDCClient.client()


# ============================================================
# QUERY IDC
# ============================================================

print("\nMemeriksa Series UID ke IDC...")
print("Filter tambahan:")
print("  collection_id = lidc_idri")
print("  Modality      = CT")


# Membuat daftar UID untuk SQL
uid_sql = ", ".join(
    f"'{uid}'"
    for uid in series_uids
)


query = f"""
SELECT
    collection_id,
    PatientID,
    StudyInstanceUID,
    SeriesInstanceUID,
    Modality,
    SeriesDescription,
    series_size_MB
FROM index
WHERE collection_id = 'lidc_idri'
  AND Modality = 'CT'
  AND SeriesInstanceUID IN ({uid_sql})
"""


# ============================================================
# JALANKAN QUERY
# ============================================================

result = client.sql_query(query)


print(
    f"\nSeries CT yang ditemukan di IDC : {len(result)}"
)


# ============================================================
# CEK APAKAH SEMUA 385 TERSEDIA
# ============================================================

found_uids = set(
    result["SeriesInstanceUID"]
    .astype(str)
    .str.strip()
    .tolist()
)

requested_uids = set(series_uids)


missing_uids = sorted(
    requested_uids - found_uids
)


# ============================================================
# HASIL VALIDASI
# ============================================================

print("\n" + "=" * 70)
print("VALIDASI CT")
print("=" * 70)

print(
    f"UID dari Excel       : {len(requested_uids)}"
)

print(
    f"CT di IDC             : {len(found_uids)}"
)

print(
    f"UID tidak ditemukan   : {len(missing_uids)}"
)


# ============================================================
# JIKA ADA UID HILANG
# ============================================================

if missing_uids:

    missing_file = (
        BASE_DIR / "missing_ct_series_uids.txt"
    )

    with open(
        missing_file,
        "w",
        encoding="utf-8"
    ) as f:

        for uid in missing_uids:

            f.write(uid + "\n")


    print(
        "\nERROR: Ada Series UID yang tidak "
        "teridentifikasi sebagai CT di IDC."
    )

    print(
        f"Daftar UID disimpan di:\n{missing_file}"
    )

    print(
        "\nDownload DIBATALKAN."
    )

    raise SystemExit(1)


# ============================================================
# PASTIKAN SEMUANYA CT
# ============================================================

unique_modalities = (
    result["Modality"]
    .astype(str)
    .str.strip()
    .str.upper()
    .unique()
    .tolist()
)


print(
    f"\nModality dari hasil query IDC : "
    f"{unique_modalities}"
)


if unique_modalities != ["CT"]:

    raise ValueError(
        "Hasil query mengandung modality selain CT."
    )


# ============================================================
# TOTAL UKURAN
# ============================================================

total_size_mb = (
    result["series_size_MB"]
    .fillna(0)
    .sum()
)

total_size_gb = (
    total_size_mb / 1024
)


print("\n" + "=" * 70)
print("UKURAN DATA")
print("=" * 70)

print(
    f"Total CT : {total_size_mb:,.2f} MB"
)

print(
    f"         : {total_size_gb:,.2f} GB"
)


# ============================================================
# SIMPAN HASIL QUERY
# ============================================================

output_metadata = (
    BASE_DIR / "ct_download_selection.csv"
)

result.to_csv(
    output_metadata,
    index=False
)


print(
    f"\nDaftar CT yang akan didownload disimpan di:"
)

print(
    output_metadata
)


# ============================================================
# DOWNLOAD / DRY RUN
# ============================================================

print("\n" + "=" * 70)

if DRY_RUN:

    print("DRY RUN - TIDAK ADA FILE YANG DIDOWNLOAD")

else:

    print("DOWNLOAD CT")

print("=" * 70)


# ============================================================
# DOWNLOAD HANYA SERIES CT
# ============================================================

client.download_dicom_series(
    seriesInstanceUID=(
        result["SeriesInstanceUID"]
        .astype(str)
        .tolist()
    ),
    downloadDir=str(DOWNLOAD_DIR),
    dry_run=DRY_RUN,
    quiet=False,
    show_progress_bar=True,
    source_bucket_location="gcs",
)


# ============================================================
# SELESAI
# ============================================================

print("\n" + "=" * 70)

if DRY_RUN:

    print("DRY RUN SELESAI")

    print(
        "\nTidak ada file yang didownload."
    )

    print(
        "Jika ukuran sudah sesuai, ubah:"
    )

    print(
        "    DRY_RUN = True"
    )

    print(
        "menjadi:"
    )

    print(
        "    DRY_RUN = False"
    )

    print(
        "\nKemudian jalankan kembali script."
    )

else:

    print("DOWNLOAD SELESAI")

    print(
        f"\nData CT berada di:"
    )

    print(
        DOWNLOAD_DIR
    )

print("=" * 70)