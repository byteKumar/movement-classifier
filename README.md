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

Work in progress; results will be added as each milestone lands.
