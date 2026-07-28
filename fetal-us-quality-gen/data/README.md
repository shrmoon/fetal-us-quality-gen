# Data setup

This project does not include the dataset itself. Follow these steps:

1. Download the dataset from Zenodo: https://zenodo.org/record/3904280
   (Burgos-Artizzu et al., 2020 — FETAL_PLANES_DB)
2. Unzip it so you end up with this structure inside `data/`:

```
data/
├── FETAL_PLANES_DB_data.csv
└── Images/
    ├── Image1.png
    ├── Image2.png
    └── ...
```

3. Confirm the CSV has (at least) these columns — the exact column names
   in the original release are: `Image_name`, `Patient_num`, `Plane`,
   `Brain_plane`, `Operator`, `US_Machine`, `Train`. If the version you
   download has slightly different column names, update the column names
   at the top of `src/dataset.py` to match.

4. This folder (`data/`) is listed in `.gitignore` — do not commit the raw
   images or CSV to GitHub. Only the code should be pushed; the dataset
   stays local, both for repo size reasons and to respect the dataset's
   distribution terms.
