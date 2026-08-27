# Fetal Ultrasound Plane Classification: A Cross-Domain Generalization Study

A data pipeline and modeling project analyzing a 12,400-image medical
imaging dataset, focused on building a reproducible pipeline to measure
how well models generalize across different data sources — and how much
of that generalization gap can be closed through better model design.

## Overview

Most machine learning models are evaluated on data that looks like their
training data. In practice, data pulled from a different source, device,
or time period often behaves differently — a common failure mode in any
production data system. This project builds a small, reproducible pipeline
to measure that gap directly: structuring a dataset by source, engineering
a deliberate train/test split along that source boundary (rather than a
random split), and quantifying how much model performance drops as a
result.

## Dataset

- **Source:** FETAL_PLANES_DB (Burgos-Artizzu et al., 2020), a public
  dataset of 12,400 medical images from 1,792 subjects, collected across
  multiple imaging devices at two hospitals. https://zenodo.org/record/3904280
- **Structure:** a CSV of metadata (`FETAL_PLANES_DB_data.csv`) joined
  against image files, with fields including `Image_name`, `Patient_num`,
  `Plane` (target label), `US_Machine` (data source), and `Operator`.
- Raw data is not redistributed in this repository, in line with the
  source dataset's terms of use. Setup instructions: `data/README.md`.

## Data Pipeline Design

The core engineering problem this project solves: build a pipeline that
tests generalization across data sources, not just overall accuracy.

- **Ingestion:** load and join image files against CSV metadata using
  pandas, validating expected columns and image paths.
- **Split strategy:** rather than a random train/test split, the pipeline
  deliberately partitions data by `US_Machine` (data source) —
  holding one source out entirely from training. This produces two
  distinct evaluation sets from the same pipeline:
  - **In-distribution (ID):** held-out data from sources seen during training.
  - **Out-of-distribution (OOD):** data from a source never seen during training.
- **Reproducibility:** fixed random seeds, deterministic splits, and a
  config-driven CLI (`argparse`) so any run can be repeated exactly.
- **Version control:** the full pipeline, from raw data ingestion through
  evaluation, is tracked in Git with a clear module structure (see below).

## Modeling & Evaluation

To test whether the pipeline's generalization measurement was meaningful,
two models were run through it:

1. A **ResNet18** backbone pretrained on ImageNet, fine-tuned on the target
   dataset.
2. A **simple CNN baseline** trained from scratch, for comparison.

Both were evaluated on the same ID and OOD splits produced by the pipeline,
with accuracy and a **generalization gap** (ID − OOD accuracy) computed
for each as the key output metric — not just raw accuracy.

## Project Structure

```
fetal-us-quality-gen/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md          # dataset setup instructions
├── src/
│   ├── dataset.py          # data ingestion, joining, and ID/OOD split logic
│   ├── model.py             # ResNet18 and simple CNN baseline
│   ├── train.py              # training pipeline (CLI-driven)
│   ├── evaluate.py           # ID vs OOD evaluation + generalization gap
│   └── utils.py               # seeding, transforms, plotting helpers
└── results/                # run outputs: checkpoints, metrics, plots
    ├── best_model.pt
    ├── generalization_gap.png
    └── simple_cnn/
```

## How to Run

```bash
pip install -r requirements.txt

# Run pipeline + train ResNet18
python src/train.py --data_dir data/ --epochs 15 --model resnet18 --output_dir results/

# Run pipeline + train CNN baseline
python src/train.py --data_dir data/ --epochs 15 --model simple_cnn --output_dir results/simple_cnn/

# Evaluate a given run's ID vs OOD generalization gap
python src/evaluate.py --data_dir data/ --checkpoint results/best_model.pt --output_dir results/
```

## Results

| Model | ID Accuracy | OOD Accuracy | Generalization Gap |
|---|---|---|---|
| Simple CNN (from scratch) | 87.2% | 78.4% | 8.9 points |
| **ResNet18 (pretrained)** | **96.5%** | **94.7%** | **1.7 points** |

![Generalization gap](results/generalization_gap.png)

The same pipeline, applied to two different models, produced very
different generalization behavior — the pretrained model's gap was roughly
5x smaller. This illustrates why measuring performance split by data
source matters: aggregate accuracy alone would not have surfaced this
difference. Per-class breakdown also showed uneven degradation — one
category ("Other") dropped more sharply under the OOD split than overall
accuracy suggested, reinforcing the value of source-aware evaluation over
a single aggregate number.

## Skills Demonstrated

- Data ingestion and joining of structured metadata with an unstructured
  (image) dataset using **pandas**
- Designing a **non-random, source-aware data split** to test for
  distribution shift, rather than relying on a default random split
- Building a **reproducible, CLI-driven pipeline** (Python, argparse,
  fixed seeds) rather than a one-off notebook script
- **Version-controlled** pipeline code (Git/GitHub)
- Model training and evaluation (**PyTorch**, ResNet18, CNNs) as a
  downstream test of the pipeline's output
- Clear metric selection: reporting a **derived comparison metric**
  (generalization gap) rather than a single surface-level number

## Limitations

- Single public dataset from one geographic region; a production data
  pipeline would need to handle more heterogeneous, larger-scale, and
  messier real-world sources.
- Single-run experiments; no automated pipeline monitoring, retraining,
  or orchestration (e.g. Airflow) layered on top yet — a natural next
  extension.

## References

Burgos-Artizzu, X.P., Coronado-Gutiérrez, D., Valenzuela-Alcaraz, B. et al.
Evaluation of deep convolutional neural networks for automatic
classification of common maternal fetal ultrasound planes. *Sci Rep* 10,
10200 (2020).
