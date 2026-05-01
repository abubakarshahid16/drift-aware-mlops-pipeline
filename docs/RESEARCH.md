# Research Notes

## Research Problem

Streaming ML models degrade when the real data distribution changes after deployment. This project studies how an MLOps pipeline can detect concept drift, expose it through monitoring, and trigger adaptive retraining.

## Research Questions

RQ1: Can a hybrid detector reduce false positives compared with single-signal drift detectors?

RQ2: Can monitoring metrics serve both dashboard observability and experimental measurement?

RQ3: What is the trade-off between detection delay, accuracy, latency, and retraining frequency?

RQ4: Can the full system remain reproducible through Docker and CI/CD?

## Proposed Detector: HybridDD

HybridDD combines:

- DDM for performance drift.
- KSWIN for distribution drift.
- Consensus within a configurable time window.
- Confidence override when error rate becomes high enough.
- Cooldown to avoid retraining storms.

## Literature Review Coverage

The bibliography includes 19 entries covering:

- Concept drift detection.
- Streaming data mining.
- Online learning.
- MLOps and production ML systems.
- Experiment tracking.
- Monitoring and observability.
- Statistical comparison of classifiers.

The bibliography file is `paper/refs.bib`.

## Experimental Setup

Datasets and streams:

- ELEC2 electricity market dataset.
- SEA synthetic abrupt drift stream.
- Rotating hyperplane synthetic gradual drift stream.

Models:

- SGD logistic classifier.
- Hoeffding tree.
- Batch logistic baseline.

Detectors:

- ADWIN
- DDM
- EDDM
- KSWIN
- Page-Hinkley
- HybridDD

Metrics:

- Accuracy
- Detection delay
- False positive rate
- Miss rate
- Retrain count
- Throughput
- Inference latency
- Online update latency

Statistical tests:

- Friedman test
- Nemenyi post-hoc critical difference

## Reproducibility Commands

Smoke run:

```bash
python -m src.pipelines.experiment --no-mlflow --seeds 42 --models sgd_logistic --detectors ddm hybrid --streams sea --n-samples 1000
```

Full run:

```bash
python -m src.pipelines.experiment \
  --seeds 42 1337 2024 \
  --models sgd_logistic hoeffding_tree \
  --detectors adwin ddm eddm kswin page_hinkley hybrid \
  --streams elec2 sea hyperplane
```

Refresh paper numbers:

```bash
python scripts/fill_paper_numbers.py
```
