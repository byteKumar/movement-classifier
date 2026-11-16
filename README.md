# movement-classifier

Classifies human movement (walking, walking upstairs, walking downstairs, sitting, standing, lying) from smartphone accelerometer and gyroscope data at 50 Hz. It covers the whole path from raw sensor streams to a served, monitored model:

- windowing of raw recordings and a train/validation/test split by subject, so no person appears in both training and test
- a hand-crafted-feature baseline with gradient boosting
- a 1D-CNN in PyTorch, with experiment tracking in MLflow
- evaluation over several seeds and leave-subjects-out cross-validation
- ONNX export, served with FastAPI and ONNX Runtime
- drift checks on the input signals

Data: the UCI "Smartphone-Based Recognition of Human Activities and Postural Transitions" (HAPT) raw recordings, from the same 30-subject experiment as the UCI HAR dataset. The dataset is downloaded by a script and is not stored in this repo.

Work in progress; setup steps and results will be added as each milestone lands.
