import os
import numpy as np
import pandas as pd
import pylidc as pl


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\230016_rivan\source-code2020"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

METADATA_FILE = os.path.join(
    DATASET_DIR,
    "ct_only_metadata.xlsx"
)

OUTPUT_FILE = os.path.join(
    DATASET_DIR,
    "pylidc_nodules_extracted.xlsx"
)

# ============================================================
# LABEL RULE
# ============================================================
#
# Mean Consensus Malignancy
#
# <= 2.5       -> Benign (0)
# > 2.5 < 3.5  -> Exclude / Uncertain
# >= 3.5       -> Malignant (1)
#
# Jumlah reader TIDAK digunakan sebagai filter.
# ============================================================

BENIGN_MAX = 2.5
MALIGNANT_MIN = 3.5


# ============================================================
# COMPATIBILITY PATCH
# ============================================================

np.int = int


# ============================================================
# START
# ============================================================

print("=" * 70)
print("EXTRACT NODULES FROM PYLIDC")
print("=" * 70)


# ============================================================
# LOAD METADATA
# ============================================================

print("\n[1] Membaca metadata CT...")

if not os.path.exists(METADATA_FILE):
    raise FileNotFoundError(
        f"\nFile tidak ditemukan:\n{METADATA_FILE}\n\n"
        "Pastikan ct_only_metadata.xlsx berada di folder dataset."
    )

metadata = pd.read_excel(METADATA_FILE)

print(f"Jumlah baris metadata : {len(metadata)}")


# ============================================================
# FIND SERIES UID COLUMN
# ============================================================

uid_candidates = [
    "Series Instance UID",
    "SeriesInstanceUID",
    "Series Instance UID",
    "seriesInstanceUID",
    "series_instance_uid",
]

uid_column = None

for col in uid_candidates:
    if col in metadata.columns:
        uid_column = col
        break

if uid_column is None:
    raise ValueError(
        "\nKolom Series Instance UID tidak ditemukan.\n"
        f"Kolom tersedia:\n{metadata.columns.tolist()}"
    )

series_uids = (
    metadata[uid_column]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

print(f"Kolom UID yang digunakan : {uid_column}")
print(f"Series UID unik          : {len(series_uids)}")


# ============================================================
# LOAD PYLIDC
# ============================================================

print("\n[2] Membaca database pylidc...")

scans = pl.query(pl.Scan).all()

print(
    f"Total Scan di database pylidc : {len(scans)}"
)


# ============================================================
# CREATE UID -> SCAN MAPPING
# ============================================================

scan_map = {
    str(scan.series_instance_uid): scan
    for scan in scans
}


# ============================================================
# MATCH CT WITH PYLIDC
# ============================================================

print("\n[3] Mencocokkan CT dengan pylidc...")

matched_scans = []
missing_uids = []

for uid in series_uids:

    if uid in scan_map:
        matched_scans.append(
            scan_map[uid]
        )
    else:
        missing_uids.append(uid)


print(
    f"CT cocok dengan pylidc : {len(matched_scans)}"
)

print(
    f"CT tidak ditemukan     : {len(missing_uids)}"
)


if missing_uids:

    print("\nContoh UID yang tidak ditemukan:")

    for uid in missing_uids[:10]:
        print(uid)


# ============================================================
# EXTRACT NODULE CLUSTERS
# ============================================================

print("\n[4] Mengekstrak cluster nodul...")

records = []

warning_scans = []


for scan_idx, scan in enumerate(
    matched_scans,
    start=1
):

    print(
        f"\rProcessing "
        f"{scan_idx}/{len(matched_scans)} "
        f"| Patient: {scan.patient_id}",
        end="",
        flush=True
    )

    try:

        # ====================================================
        # GET ANNOTATIONS
        # ====================================================

        annotations = scan.annotations

        if annotations is None:
            continue

        if len(annotations) == 0:
            continue


        # ====================================================
        # CLUSTER ANNOTATIONS
        # ====================================================

        clusters = scan.cluster_annotations()


        # ====================================================
        # PROCESS EACH CLUSTER
        # ====================================================

        for cluster_idx, cluster in enumerate(
            clusters,
            start=1
        ):

            # =================================================
            # MALIGNANCY SCORE
            # =================================================

            scores = []

            for ann in cluster:

                malignancy = ann.malignancy

                if malignancy is not None:

                    scores.append(
                        float(malignancy)
                    )


            # =================================================
            # MEAN CONSENSUS + LABEL
            # =================================================

            if len(scores) == 0:

                mean_malignancy = np.nan
                label = "NO_SCORE"
                label_code = np.nan

            else:

                mean_malignancy = float(
                    np.mean(scores)
                )

                # -----------------------------
                # BENIGN
                # -----------------------------

                if mean_malignancy <= BENIGN_MAX:

                    label = "Benign"
                    label_code = 0


                # -----------------------------
                # MALIGNANT
                # -----------------------------

                elif mean_malignancy >= MALIGNANT_MIN:

                    label = "Malignant"
                    label_code = 1


                # -----------------------------
                # UNCERTAIN
                # -----------------------------

                else:

                    label = "Exclude"
                    label_code = np.nan


            # =================================================
            # BBOX
            # =================================================
            #
            # Kita ambil bbox dari seluruh annotation
            # dalam cluster.
            #
            # bbox_matrix() menghasilkan koordinat voxel
            # dari contour annotation.
            #
            # Axis:
            #   X -> kolom
            #   Y -> baris
            #   Z -> slice
            #
            # Kita ambil min/max seluruh titik contour.
            # =================================================

            # =================================================
            # BBOX DARI CONTOUR PYLIDC
            # =================================================

            x_values = []
            y_values = []
            z_values = []


            for ann in cluster:

                try:

                    # ---------------------------------------------
                    # Setiap annotation memiliki beberapa contour
                    # pada beberapa slice.
                    # ---------------------------------------------

                    for contour in ann.contours:

                        # -----------------------------------------
                        # Koordinat contour pada slice
                        #
                        # to_matrix() menghasilkan koordinat
                        # contour dalam bentuk:
                        # [x, y]
                        # -----------------------------------------

                        coords = np.asarray(
                            contour.to_matrix()
                        )

                        if coords.size == 0:
                            continue

                        if coords.ndim != 2:
                            continue

                        if coords.shape[1] < 2:
                            continue

                        # X
                        x_values.extend(
                            coords[:, 0].tolist()
                        )

                        # Y
                        y_values.extend(
                            coords[:, 1].tolist()
                        )

                        # -----------------------------------------
                        # Z
                        #
                        # image_k_position = index slice pada
                        # volume pylidc
                        # -----------------------------------------

                        if contour.image_k_position is not None:

                            z_values.append(
                                float(
                                    contour.image_k_position
                                )
                            )

                except Exception:
                    continue


            # =================================================
            # HITUNG BBOX
            # =================================================

            if len(x_values) > 0:

                x_min = int(
                    np.floor(
                        np.min(x_values)
                    )
                )

                x_max = int(
                    np.ceil(
                        np.max(x_values)
                    )
                )

            else:

                x_min = None
                x_max = None


            if len(y_values) > 0:

                y_min = int(
                    np.floor(
                        np.min(y_values)
                    )
                )

                y_max = int(
                    np.ceil(
                        np.max(y_values)
                    )
                )

            else:

                y_min = None
                y_max = None


            if len(z_values) > 0:

                z_min = int(
                    np.floor(
                        np.min(z_values)
                    )
                )

                z_max = int(
                    np.ceil(
                        np.max(z_values)
                    )
                )

            else:

                z_min = None
                z_max = None


            # =================================================
            # JUMLAH SLICE
            # =================================================

            if (
                z_min is not None
                and z_max is not None
            ):

                z_slices = (
                    z_max - z_min + 1
                )

            else:

                z_slices = None


            # =================================================
            # ANNOTATION IDs
            # =================================================

            annotation_ids = [
                int(ann.id)
                for ann in cluster
            ]


            # =================================================
            # MALIGNANCY SCORE STRING
            # =================================================

            score_string = ",".join(
                (
                    str(int(score))
                    if score.is_integer()
                    else str(score)
                )
                for score in scores
            )


            # =================================================
            # SAVE RECORD
            # =================================================

            records.append({

                "PatientID":
                    str(scan.patient_id),

                "StudyInstanceUID":
                    str(scan.study_instance_uid),

                "SeriesInstanceUID":
                    str(scan.series_instance_uid),

                "NoduleID":
                    (
                        f"{scan.series_instance_uid}"
                        f"_N{cluster_idx:03d}"
                    ),

                "ClusterIndex":
                    cluster_idx,

                "NReaders":
                    len(cluster),

                "MalignancyScores":
                    score_string,

                "MeanMalignancy":
                    mean_malignancy,

                "Label":
                    label,

                "LabelCode":
                    label_code,

                "AnnotationIDs":
                    ",".join(
                        map(
                            str,
                            annotation_ids
                        )
                    ),

                "BBox_X_Min":
                    x_min,

                "BBox_X_Max":
                    x_max,

                "BBox_Y_Min":
                    y_min,

                "BBox_Y_Max":
                    y_max,

                "BBox_Z_Min":
                    z_min,

                "BBox_Z_Max":
                    z_max,

                "BBox_Z_Slices":
                    z_slices,
            })


    except Exception as e:

        warning_scans.append({
            "PatientID":
                str(scan.patient_id),

            "SeriesInstanceUID":
                str(scan.series_instance_uid),

            "Error":
                f"{type(e).__name__}: {e}"
        })

        print(
            f"\n\n[WARNING] Gagal memproses "
            f"{scan.patient_id}"
        )

        print(
            f"Series UID: "
            f"{scan.series_instance_uid}"
        )

        print(
            f"Error: {type(e).__name__}: {e}"
        )

        continue


print("\n")


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# ============================================================
# RESULT
# ============================================================

print("=" * 70)
print("HASIL EKSTRAKSI")
print("=" * 70)

print(
    f"\nTotal cluster nodul : {len(df)}"
)


if len(df) > 0:

    print("\nDistribusi label:")

    print(
        df["Label"]
        .value_counts(
            dropna=False
        )
    )


    print("\nDistribusi jumlah reader:")

    print(
        df["NReaders"]
        .value_counts()
        .sort_index()
    )


    print("\nStatistik Mean Malignancy:")

    print(
        df["MeanMalignancy"]
        .describe()
    )


    print("\nStatistik BBox Z:")

    print(
        df["BBox_Z_Slices"]
        .describe()
    )


    print("\nJumlah BBox Z kosong:")

    print(
        df["BBox_Z_Min"]
        .isna()
        .sum()
    )


# ============================================================
# SAVE MAIN OUTPUT
# ============================================================

print("\n[5] Menyimpan hasil...")

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nFile berhasil disimpan:"
)

print(OUTPUT_FILE)


# ============================================================
# SAVE WARNING
# ============================================================

if warning_scans:

    warning_file = os.path.join(
        DATASET_DIR,
        "pylidc_extraction_warnings.xlsx"
    )

    pd.DataFrame(
        warning_scans
    ).to_excel(
        warning_file,
        index=False
    )

    print(
        f"\nWarning disimpan ke:"
    )

    print(warning_file)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)


if len(df) > 0:

    total = len(df)

    benign = (
        df["Label"] == "Benign"
    ).sum()

    malignant = (
        df["Label"] == "Malignant"
    ).sum()

    exclude = (
        df["Label"] == "Exclude"
    ).sum()

    no_score = (
        df["Label"] == "NO_SCORE"
    ).sum()


    print(
        f"Total cluster       : {total}"
    )

    print(
        f"Benign              : {benign}"
    )

    print(
        f"Malignant           : {malignant}"
    )

    print(
        f"Exclude / Uncertain : {exclude}"
    )

    print(
        f"No score            : {no_score}"
    )


    print("\nPersentase:")

    print(
        f"Benign     : "
        f"{benign / total * 100:.2f}%"
    )

    print(
        f"Malignant  : "
        f"{malignant / total * 100:.2f}%"
    )

    print(
        f"Exclude    : "
        f"{exclude / total * 100:.2f}%"
    )


print("\nSelesai.")