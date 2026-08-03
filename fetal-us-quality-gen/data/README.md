# Data setup

This project does not include the dataset itself. Follow these steps:

1. Download the dataset from Zenodo: https://zenodo.org/record/3904280
   (Burgos-Artizzu et al., 2020 — FETAL_PLANES_DB)
2. Unzip it so you end up with this structure inside `data/`:

3. **Note:** the CSV uses a **semicolon (`;`)** delimiter, not a comma —
   `src/dataset.py` is already configured for this
   (`pd.read_csv(csv_path, sep=";")`). If you're using a different release
   of the dataset with a comma-delimited CSV, update that line accordingly.

4. Confirmed columns in the release used for this project:
   `Image_name`, `Patient_num`, `Plane`, `Brain_plane`, `Operator`,
   `US_Machine`, `Train`. These match `src/dataset.py` exactly — no
   changes needed if you're using the same Zenodo release.

5. This folder (`data/`) is listed in `.gitignore` — the raw images and
   CSV are intentionally excluded from version control, both for repo
   size and to respect the dataset's distribution terms. Only code is
   tracked here.