# movement-classifier

Classifies human movement (walking, walking upstairs, walking downstairs, sitting, standing, lying) from smartphone accelerometer and gyroscope data at 50 Hz. It covers the whole path from raw sensor streams to a served, monitored model:

- windowing of raw recordings and a train/validation/test split by subject, so no person appears in both training and test
- a hand-crafted-feature baseline with gradient boosting
- a 1D-CNN in PyTorch, with experiment tracking in MLflow
- evaluation over several seeds and leave-subjects-out cross-validation
- ONNX export, served with FastAPI and ONNX Runtime
- drift checks on the input signals

Data: the UCI "Smartphone-Based Recognition of Human Activities and Postural Transitions" (HAPT) raw recordings, from the same 30-subject experiment as the UCI HAR dataset. The dataset is downloaded by a script and is not stored in this repo.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python scripts/download_data.py   # 79.6 MB zip from the UCI repository, checksum-verified
python scripts/prepare_data.py    # cuts 128-sample windows, caches data/windows.npz
pytest
```

## Data pipeline

- 6 channels: accelerometer x/y/z (g) and gyroscope x/y/z (rad/s), raw and unfiltered, 50 Hz.
- Each labeled activity segment is cut into 128-sample windows (2.56 s) with 50% overlap. A window never crosses a label boundary, and postural transitions are dropped.
- Split by subject: the dataset's own 9 test subjects are the test set, 4 of the 21 training subjects are held out for validation, and 17 train. Tests check that no subject, recording or overlapping window is shared across splits.

| split | subjects | windows |
|---|---|---|
| train | 17 | 5,928 |
| val | 4 | 1,487 |
| test | 9 | 2,996 |
| total | 30 | 10,411 |

The dataset (UCI HAPT, Reyes-Ortiz et al.) is licensed for non-commercial use.

## Baseline: features + gradient boosting

`python scripts/train_baseline.py` extracts 99 features per window: for each of 8 channels (the 6 axes plus accelerometer and gyroscope magnitude) the mean, std, min, max, energy, dominant frequency and power in 6 FFT bands, plus the 3 accelerometer axis correlations. It trains scikit-learn's `HistGradientBoostingClassifier` on them. The number of boosting rounds (86) is chosen on the validation subjects; the test subjects are scored once. Every run is logged to MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

Test subjects, 5 seeds, CPU: macro-F1 **0.9479 ± 0.0000**, accuracy **0.9479 ± 0.0000**. The std is 0 because this model has no randomness without subsampling, so every seed gives the same trees. The weakest classes are sitting (F1 0.900) and standing (0.913): 65 sitting windows were predicted as standing and 33 standing as sitting.

## 1D-CNN

`python scripts/train_cnn.py --seeds 0` trains a 3-block 1D-CNN (32,550 parameters) directly on the raw windows. The training set's per-channel mean and std are stored inside the model, so it takes raw sensor values. Adam, batch 64, up to 60 epochs, early stopping on validation macro-F1 with patience 10, best-epoch weights kept. Every epoch's train loss and validation F1 go to MLflow.

First run (seed 0, MPS, 10 s): best epoch 5 of 15, validation macro-F1 0.9640, **test macro-F1 0.9197**, accuracy 0.9176. That is below the baseline on test, even though it is above it on validation. Most test errors come from one subject (subject 10: 64.1% accuracy, 46 of 160 static windows predicted as a walking class), whose phone orientation while lying down differs from most training subjects. Validation F1 also moves between 0.91 and 0.96 from one epoch to the next, so early stopping on only 4 subjects is noisy.

### Choosing augmentation without touching the test set

Changing the model because of a test result would be tuning on the test set, so the fix was judged by cross-validation over the 21 **non-test** subjects (`scripts/select_cnn.py`). The decision rule was written into the script and committed (`9fe5e74`) before it was first run. Five folds of 4–5 held-out subjects each; in each fold 4 other subjects are used for early stopping. Three seeds per config, MPS.

| config | held-out macro-F1 (15 runs) | per-fold means |
|---|---|---|
| HGB baseline (reference) | 0.9200 ± 0.0350 | 0.924, 0.944, 0.962, 0.888, 0.881 |
| CNN v1, no augmentation | 0.9155 ± 0.0368 | 0.937, 0.920, 0.964, 0.876, 0.881 |
| **CNN v2**, random rotation ≤ 30° + ±10% gain | **0.9308 ± 0.0387** | 0.937, 0.965, 0.971, 0.883, 0.899 |

v2 beat v1 on 4 of 5 folds (+0.0154 mean), which meets the rule, so v2 is now the default. Caveat: the idea for v2 came from looking at test errors (subject 10), so its later test score is slightly test-informed. The v1 test result above stays in the record.

## Evaluation

`python scripts/evaluate.py --device mps` (CNN on MPS, baseline on CPU; all runs logged to MLflow experiment `evaluation`).

**Fixed split, test subjects, seeds 0–4** (mean ± sample std over seeds):

| model | macro-F1 | accuracy | per-seed macro-F1 |
|---|---|---|---|
| Features + HGB | **0.9479 ± 0.0000** | 0.9479 ± 0.0000 | identical for every seed |
| 1D-CNN (v2) | 0.9324 ± 0.0207 | 0.9308 ± 0.0205 | 0.9229, 0.9095, 0.9217, 0.9482, 0.9594 |

**Leave-subjects-out CV over all 30 subjects** (5 folds of 6 subjects; mean ± std over the 5 fold means, CNN fold means average 5 seeds): CNN **0.9353 ± 0.0268**, HGB 0.9391 ± 0.0295. Per fold, CNN / HGB: 0.908 / 0.926, 0.938 / 0.949, 0.958 / 0.937, 0.965 / 0.982, 0.908 / 0.902.

On this data the two models are about even. The baseline is ahead on the official test subjects and by 0.004 in CV, which is well inside the fold-to-fold spread. They fail differently:

| per-class F1 (test, mean over seeds) | walking | upstairs | downstairs | sitting | standing | laying |
|---|---|---|---|---|---|---|
| HGB | 0.976 | 0.950 | 0.948 | 0.900 | 0.913 | 1.000 |
| CNN | 0.982 | 0.976 | 0.987 | **0.798** | 0.851 | 0.999 |

The CNN is better on the three walking classes and worse on sitting. **The hardest pair is sitting vs standing for both models**: the CNN labeled 27.4% of sitting windows as standing (696 of 2,540 over 5 seeds) and 7.1% of standing as sitting (197 of 2,780); for HGB it was 12.8% and 5.9%. Augmentation removed most of the static-as-walking errors seen in v1: test subject 10 went from 64.1% accuracy (v1, seed 0) to 80.9% (v2, mean of 5 seeds).

![CNN confusion matrix](docs/confusion_cnn.png)

The served model is the seed-0 CNN (test macro-F1 0.9229), fixed in `scripts/evaluate.py` before the run, not the best seed.

## Serving

```bash
python scripts/export_onnx.py                                   # artifacts/model.onnx + parity check
uvicorn movement.serve:create_app --factory --port 8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"windows": [[[0.9, -0.1, 0.5, 0.0, 0.0, 0.0], ... 128 rows ...]]}'
```

- **Export:** `torch.onnx.export` (dynamo exporter, dynamic batch dimension). The model's normalization is inside the graph, so clients send raw sensor values (acc in g, gyro in rad/s).
- **Parity:** on all 2,996 test windows, ONNX Runtime vs PyTorch on CPU: max |logit difference| **2.384e-06**, max |probability difference| 4.768e-07, same predicted class for 100% of windows (vs PyTorch on MPS: 2.861e-06). Tests also check batch sizes 1, 7, 32 and 300 against a 1e-4 tolerance.
- **Model:** 32,550 parameters, `model.onnx` 151,572 bytes (0.152 MB).
- **API:** `POST /predict` takes 1–256 windows of shape 128 × 6 and returns a label and class probabilities for each, plus a model version (first 12 hex characters of the ONNX file's SHA-256). Wrong shapes, NaN or infinite values, and non-JSON bodies get a 422 with a message; more than 256 windows get a 413. The body is parsed with `json` + NumPy instead of a nested pydantic model because a batch of 32 is 24,576 numbers. ONNX Runtime runs with 1 thread per call in FastAPI's thread pool, so concurrent requests can use several cores.

Work in progress; results will be added as each milestone lands.
