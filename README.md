# Sensor Match Dominates Pretraining Source in Geospatial Foundation Model Transfer

Code, data and figures for the paper of the same name, submitted to *PFG – Journal of
Photogrammetry, Remote Sensing and Geoinformation Science*.

The study crosses two backbones (ResNet-50, Swin-B), three pretraining sources
(SatlasPretrain, ImageNet, random initialization), two input configurations
(3-band RGB, 9-band multispectral), six label budgets and five seeds, on EuroSAT
and NWPU-RESISC45 — 510 runs under one frozen protocol.

Every number in the paper can be recomputed from the files here. Nothing needs retraining.

## Quick start

Regenerate the paper's figures from the released results:

```bash
pip install numpy pandas matplotlib
python make_figures.py .        # writes figures/fig1..fig8
```

To re-run the study itself, open `SensorMatchStudy.ipynb` in Colab on a GPU runtime
and run the cells top to bottom. Only cell 1 needs editing.

## What is here

| Path | Contents |
|---|---|
| `SensorMatchStudy.ipynb` | The full study: config, source modules, test suite, grid runner, analysis |
| `make_figures.py` | Rebuilds every figure in the paper from `results/` |
| `results/` | One row per run for all 510 runs, plus the statistical tests and derived analyses |
| `derived/` | Aggregated tables used to typeset the paper |
| `splits/` | The fixed NWPU-RESISC45 70/15/15 split (seed 42) |
| `confusion/` | Confusion matrices for the two configurations discussed in the error analysis |
| `environment.txt` | Exact software versions the grid ran under |

### results/

- `eurosat_results.csv`, `resisc45_results.csv` — every run: all six metrics at both
  stages, epochs and optimiser steps actually taken, split hashes, GPU, timings
- `seed_tests.csv` — 396 seed-level comparisons (Wilcoxon, paired t, Holm)
- `mcnemar_pvalues.csv` — McNemar and Jaccard for all 66 pairs at every budget, seed 42
- `per_class_f1.csv` — per-class F1 for every run and stage
- `factorial_eurosat.csv`, `label_efficiency.csv`, `synthesis_deltas.csv`,
  `frozen_vs_finetuned.csv` — the derived analyses
- `min_steps_sweep.csv` — the pilot sweep that fixed the 300/200 optimiser-step floor
- `dataset_summary.csv`, `dataset_summary.md` — per-class split counts

## Notes

Per-epoch validation histories were not archived, so Fig. 6 of the paper is the one
figure that cannot be rebuilt from these files. Every other figure is produced by
`make_figures.py`.

EuroSAT uses the dataset's own released split; the hash of each split is recorded on
every result row. NWPU-RESISC45 uses the split in `splits/`.

## Datasets

- EuroSAT — Helber et al. (2019)
- NWPU-RESISC45 — Cheng et al. (2017)
- SatlasPretrain checkpoints — Bastani et al. (2023)

## Citation

A citation will be added here once the paper has a DOI.
