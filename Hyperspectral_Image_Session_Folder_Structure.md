# Hyperspectral Image Session Folder Structure

This folder contains all data for a single hyperspectral imaging session.

---

## 1. Root Folder

- Contains the main metadata file:

  **`result_table_general.txt`**
  
  - Summarizes all images and reference measurements for the session.
  - Columns include:

    ```
    Session | Name_picture_RGB | Name_picture_HYP | Temperature_HYP |
    Reference 0 | Temperature_HYP_Ref_0 | Reference 50 | Temperature_HYP_Ref_50 |
    Reference 90 | Temperature_HYP_Ref_90 | Shell | Individual_number | Individual_name |
    Weight(g) | Array_shape | Ref_Array_shape
    ```
  - Example row:

    ```
    13_06_22cm_muestras_1, RGB_SEED_13_06_22cm_muestras_1_2024-06-13-08-10-58_2.jpg,
    HYP_SEED_13_06_22cm_muestras_1_2024-06-13-08-10-58_0, 55.5,
    C:/.../REFERENCES/Reference_0_2024-06-13-08-02-35, 54.875, ...
    SEED, 0, E_85_A, 75, (3527,1280,425), (581,1280,425)
    ```

---

## 2. HYP Folder

Contains all hyperspectral data for the session.

- **Subfolders:**
  
  - **`RAW`**
    - Compressed hyperspectral images in **LZ4** format.
    - Each file corresponds to a sample.
  
  - **`REFERENCES`**
    - Compressed reference images in **LZ4** format.
    - Includes references for calibration:
      - `Reference_0`, `Reference_50`, `Reference_90`

---

## 3. Notes

- The folder structure ensures automated scripts can locate the hyperspectral images and reference files.
- **`Array_shape`** in the metadata refers to the dimensions of the hyperspectral cube `(Height, Width, Bands)`.
- **`Ref_Array_shape`** refers to the dimensions of the calibration references.
- File naming conventions:
  - HYP: `HYP_SEED_[Session]_[Timestamp]_0`


## 4. Example Folder Structure

Session_samples_1/
├── result_table_general.txt
└── HYP/
├── RAW/
│ ├── HYP_SEED_13_06_22cm_muestras_1_2024-06-13-08-10-58_0.lz4
│ ├── HYP_SEED_13_06_22cm_muestras_1_2024-06-13-08-10-58_1.lz4
│ └── ...
└── REFERENCES/
├── Reference_0_2024-06-13-08-02-35.lz4
├── Reference_50_2024-06-13-08-04-27.lz4
└── Reference_90_2024-06-13-08-03-30.lz4
