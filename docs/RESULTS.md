# Results Guide

This document explains the result artifacts and how to reproduce them.

## Result Files

| File | Purpose |
|---|---|
| `experiments/results/results.csv` | Tabular benchmark output |
| `experiments/results/results.json` | Same results in JSON format |
| `experiments/results/stats.json` | Statistical test output |
| `paper/results.tex` | LaTeX macros used by the IEEE paper |
| `paper/figures/accuracy_curve.png` | Accuracy visualization |
| `paper/figures/detection_delay.png` | Detection-delay visualization |

## Columns In `results.csv`

Common columns:

- `model`
- `detector`
- `stream`
- `seed`
- `accuracy`
- `retrains`
- `n_drift_events`
- `throughput_samples_per_s`
- `elapsed_s`
- `latency_*`
- `n_samples`

Synthetic streams also include:

- `true_drifts`
- `true_positives`
- `false_positives`
- `missed`
- `mean_detection_delay`
- `penalized_detection_delay`
- `false_positive_rate`
- `miss_rate`

## Smoke Run

Use this for quick validation:

```bash
python -m src.pipelines.experiment \
  --no-mlflow \
  --seeds 42 \
  --models sgd_logistic \
  --detectors ddm hybrid \
  --streams sea \
  --n-samples 1000
```

## Full Benchmark

Use this for final research results:

```bash
python -m src.pipelines.experiment \
  --seeds 42 1337 2024 \
  --models sgd_logistic hoeffding_tree \
  --detectors adwin ddm eddm kswin page_hinkley hybrid \
  --streams elec2 sea hyperplane
```

## Statistical Analysis

The experiment harness computes:

- Friedman test over detector ranks.
- Nemenyi critical difference for post-hoc comparison.
- Penalized detection-delay comparison, where a missed drift is assigned the
  stream length as a bounded worst-case delay. This keeps missed detections
  visible in statistical tests instead of silently dropping them as missing
  delay values.

Implementation:

```text
src/pipelines/experiment.py
```

## Important Interpretation Note

The checked-in artifacts are intentionally small enough for fast review. If you need publication-grade final numbers, run the full benchmark and then regenerate paper macros:

```bash
python scripts/fill_paper_numbers.py
```

Then commit:

```text
experiments/results/results.csv
experiments/results/results.json
experiments/results/stats.json
paper/results.tex
paper/figures/*.png
```
