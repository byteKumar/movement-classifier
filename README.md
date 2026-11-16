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

v2 beat v1 on 4 of 5 folds (+0.0154 mean), which meets the rule, so v2 is now the default. Caveat: the idea for v2 came from looking at test errors (subject 10), so its later test score is slightly test-informed. The v1 test result above stays in the record. Results over 5 seeds and cross-validation on the test protocol follow in the evaluation milestone.

Work in progress; results will be added as each milestone lands.
