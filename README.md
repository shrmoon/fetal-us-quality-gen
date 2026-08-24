# Fetal Ultrasound Plane Classification: A Cross-Domain Generalization Study for Low-Resource Clinical Settings

A small scoped research study applying deep learning to fetal ultrasound
image classification, with a specific focus on measuring how well models
generalize across different ultrasound machines and operators. This project reflects my personal research interest in cross-domain generalization for deep learning-based fetal ultrasound image quality assessment.

## Motivation

Most deep learning models for fetal ultrasound are trained and evaluated on
data from a single hospital or machine, and often perform worse when applied
to images from a different device, operator, or clinical setting. This
project tests that idea directly: train classifiers on one subset of a
public dataset, then explicitly measure how much accuracy drops when
evaluating on a held-out subset that differs by acquisition machine, and
compare whether pretraining changes how large that drop is.

## Dataset

This project uses the **FETAL_PLANES_DB** dataset (Burgos-Artizzu et al.,
2020), a public dataset of 12,400 fetal ultrasound images from 1,792
patients, collected at two hospitals in Barcelona using multiple ultrasound
machines and operators.

- Source: https://zenodo.org/record/3904280
- The dataset includes a CSV of metadata (`FETAL_PLANES_DB_data.csv`) with
  columns including `Image_name`, `Patient_num`, `Plane` (the anatomical
  plane label), `US_Machine`, `Operator`, and a provided `Train` split.

**Setup:** download the dataset from Zenodo, unzip the images into
`data/Images/`, and place the CSV at `data/FETAL_PLANES_DB_data.csv`. See
`data/README.md` for exact expected structure.

I have not redistributed the dataset itself in this repository — only code
that operates on it — in line with the original dataset's terms of use.

## Method

1. **Task:** multi-class classification of the ultrasound `Plane` (e.g.
   abdomen, brain, femur, thorax, other).
2. **Models:** a ResNet18 backbone (pretrained on ImageNet), fine-tuned on
   fetal ultrasound images, compared against a simple from-scratch CNN
   baseline — to isolate how much pretraining contributes versus a basic
   architecture with no transfer learning.
3. **Generalization evaluation (the core idea of this project):**
   - **In-distribution (ID) test set:** held-out images from the *same*
     `US_Machine` values seen during training.
   - **Out-of-distribution (OOD) test set:** held-out images from a
     `US_Machine` (or `Operator`) *not* seen during training.
   - I report accuracy on both and compute the **generalization gap**
     (ID accuracy − OOD accuracy) as the key metric of interest, rather
     than just overall accuracy.

This ID/OOD split design is a small-scale version of the evaluation
approach used in published work on fetal ultrasound generalization to
low-resource settings (see Related Work), and reflects the kind of question I'm interested in exploring further.

## Related Work

This project builds on a growing body of work on fetal ultrasound deep
learning. Burgos-Artizzu et al. (2020) introduced the FETAL_PLANES_DB
dataset used here, evaluating CNN classifiers for standard-plane
detection. A 2022 systematic review (arXiv:2201.12260) surveyed 145 papers
on fetal ultrasound deep learning and identified generalization across
imaging sites and devices as a persistent open problem. Sendra-Balcells et
al. (2023) directly evaluated fetal ultrasound model generalization to
low-resource settings across five African countries, using held-out-device
evaluation similar in spirit to the ID/OOD design used here, though at
larger scale and with real cross-country data rather than a single public
dataset. This project is a small, self-contained test of the same
underlying question — how much do pretraining and architecture choice
affect generalization under domain shift — rather than a replication of
either study.

## Project structure

```
fetal-us-quality-gen/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md          # dataset download/setup instructions
├── src/
│   ├── dataset.py          # PyTorch Dataset + ID/OOD split logic
│   ├── model.py             # ResNet18 and simple CNN baseline
│   ├── train.py              # training loop
│   ├── evaluate.py           # ID vs OOD evaluation + generalization gap
│   └── utils.py               # seeding, transforms, plotting helpers
└── results/                # training logs, saved plots
    ├── best_model.pt        # ResNet18 checkpoint
    ├── generalization_gap.png
    └── simple_cnn/           # simple CNN baseline checkpoint + plot
```

## How to run

```bash
pip install -r requirements.txt

# Train ResNet18 (pretrained)
python src/train.py --data_dir data/ --epochs 15 --model resnet18 --output_dir results/

# Train simple CNN baseline
python src/train.py --data_dir data/ --epochs 15 --model simple_cnn --output_dir results/simple_cnn/

# Evaluate either model's ID vs OOD generalization gap
python src/evaluate.py --data_dir data/ --checkpoint results/best_model.pt --output_dir results/
python src/evaluate.py --data_dir data/ --checkpoint results/simple_cnn/best_model.pt --output_dir results/simple_cnn/
```

## Results

Both models were trained for 15 epochs, holding out one ultrasound machine
entirely as the out-of-distribution (OOD) test set.

| Model | ID Accuracy | OOD Accuracy | Generalization Gap |
|---|---|---|---|
| Simple CNN (from scratch) | 87.2% | 78.4% | 8.9 points |
| **ResNet18 (pretrained)** | **96.5%** | **94.7%** | **1.7 points** |

![Generalization gap](results/generalization_gap.png)

The pretrained ResNet18 not only achieved higher accuracy overall but showed
a substantially smaller generalization gap than the from-scratch CNN
(1.7 points vs. 8.9 points) — suggesting that ImageNet pretraining improves
not just raw accuracy but specifically the model's robustness to domain
shift across imaging devices. This is a more informative comparison than
accuracy alone: a model could achieve high overall accuracy while still
degrading sharply under distribution shift, which the CNN baseline
illustrates directly.

Overall accuracy also masks per-class differences: for both models, the
"Other" class showed noticeably weaker precision than the overall accuracy
suggests, indicating that specific anatomical categories may be more
vulnerable to domain shift than others even when aggregate accuracy looks
strong.

This result is a useful but limited signal: FETAL_PLANES_DB was collected
across only two hospitals, so the diversity of machines/settings is
narrower than a true cross-country domain shift (e.g. testing on real
low-resource clinical data, as in Sendra-Balcells et al., 2023). A small
gap for the pretrained model here does not rule out much larger
generalization failures under more extreme distribution shifts, which is
the open question I aim to investigate further in my proposed thesis.

## Limitations

- Single public dataset from one geographic region (Spain); the real
  motivation for my thesis — generalization to genuinely low-resource
  settings such as Bangladesh — would need real out-of-region clinical
  data, which this project does not include.
- Small-scale, single-run experiments; no hyperparameter search or
  statistical significance testing.
- Scoped as a focused study of one specific question (pretraining's effect
  on generalization), rather than a comprehensive evaluation across
  multiple architectures, datasets, or domain shift types.

## Why this project

In many low-resource clinical settings, only a few expert doctor may be
available to review ultrasound scans, so pregnant women often face long
waits outside the ultrasound room, and image quality issues sometimes lead
to inaccurate or inconclusive results. This motivates my interest in AI
tools that could help make fetal ultrasound image quality assessment more
consistent and accessible in settings where specialist availability is
limited.

## References

Burgos-Artizzu, X.P., Coronado-Gutiérrez, D., Valenzuela-Alcaraz, B. et al.
Evaluation of deep convolutional neural networks for automatic
classification of common maternal fetal ultrasound planes. *Sci Rep* 10,
10200 (2020).

Sendra-Balcells, C. et al. Generalisability of fetal ultrasound deep
learning models to low-resource imaging settings in five African
countries. *Sci Rep* 13, 2728 (2023).

A Review on Deep-Learning Algorithms for Fetal Ultrasound-Image Analysis.
arXiv:2201.12260 (2022).
