# Results

All numbers below were measured once on the machine listed, unless a row says it is a mean over seeds or folds. Nothing is rounded beyond what is shown. The test subjects were used only for the final scores, never for choosing models, epochs or thresholds.

## Spec results template

```
MACHINE: chip=Apple M3 RAM=16 GB, Python=3.12.10, PyTorch=2.15.0
DATA: subjects train/val/test=17/4/9 ; windows=10,411
MACRO-F1 (mean ± std, 5 seeds): baseline=0.9479 ± 0.0000 CNN=0.9324 ± 0.0207 ; accuracy: baseline=0.9479 ± 0.0000 CNN=0.9308 ± 0.0205
LEAVE-SUBJECTS-OUT CV macro-F1 (CNN)=0.9353 ± 0.0268 (5 folds of 6 subjects, 5 seeds per fold; HGB 0.9391 ± 0.0295)
HARDEST CLASSES: SITTING vs STANDING (F1 0.798 / 0.851, CNN mean over 5 seeds; HGB 0.900 / 0.913)
ONNX parity max abs diff=2.384e-06 (logits, 2,996 test windows, ONNX Runtime CPU vs PyTorch CPU) ; model size=0.152 MB, params=32,550
SERVING: single p50/p99=0.449/0.714 ms ; batch-32 p50/p99=7.079/12.604 ms ; req/s at p99<50 ms=500 (single-window requests; the 1,000 req/s step had p99 45.67 ms but 83 dropped iterations, so it fails the pre-declared rule)
DRIFT: shifts detected 6 of 6 ; false alarms on clean data 0 of 200 (calibrated check, test subjects; the first KS p-value check had 47 of 200)
ANYTHING THAT FAILED OR SURPRISED YOU: the CNN did not beat the feature baseline (0.9324 vs 0.9479 on test, 0.9353 vs 0.9391 in CV); the first drift check raised alarms on 47 of 200 clean test batches and 200 of 200 clean val batches and had to be recalibrated; a 528 KB JSON body makes a batch of 32 cost about 16x a single window; p99 under steady load (29.85 ms at 250 req/s) was far above p99 with one client (0.714 ms)
REPO pushed? yes
```

## Devices

- CNN training and evaluation: PyTorch on MPS (Apple M3 GPU).
- HGB baseline: scikit-learn on CPU.
- ONNX parity: ONNX Runtime CPU vs PyTorch CPU (2.384e-06) and vs PyTorch MPS (2.861e-06).
- Serving benchmark: ONNX Runtime on CPU, 1 thread per call, 1 uvicorn worker. k6 ran on the same machine, so it competed with the server for CPU.

## Model quality (test subjects 2, 4, 9, 10, 12, 13, 18, 20, 24)

| model | macro-F1 (5 seeds) | accuracy (5 seeds) |
|---|---|---|
| Features + HistGradientBoosting | 0.9479 ± 0.0000 | 0.9479 ± 0.0000 |
| 1D-CNN v2 (rotation + gain augmentation) | 0.9324 ± 0.0207 | 0.9308 ± 0.0205 |

- **HGB std is 0** because the model has no randomness without subsampling, so every seed gives the same trees.
- **CNN per-seed macro-F1:** 0.9229, 0.9095, 0.9217, 0.9482, 0.9594.
- **Served model:** the seed-0 CNN (0.9229), fixed before the run, not the best seed.
- **v1 (no augmentation):** seed 0 reached 0.9197. Adopting v2 was decided by cross-validation on the non-test subjects, with a rule committed before that run. The idea for it came from looking at test errors, so v2's test score is slightly test-informed.
- **Leave-subjects-out CV over all 30 subjects:** CNN 0.9353 ± 0.0268, HGB 0.9391 ± 0.0295.

## Serving benchmark (k6 v2.4.0, 60 s per run, measured once)

Bodies are real test-subject windows: 64 single-window bodies of 16,135 bytes and 16 batch-32 bodies of 527,989 bytes. The 1-minute load average was 2.66–4.12 before the runs, from normal desktop use plus the benchmark itself.

| run | requests | achieved req/s | p50 ms | p90 ms | p99 ms | max ms | failed | dropped |
|---|---|---|---|---|---|---|---|---|
| latency, 1 window, 1 client | 115,684 | 1,928.0 | 0.449 | 0.474 | 0.714 | 62.526 | 0 | 0 |
| latency, 32 windows, 1 client | 7,164 | 119.4 | 7.079 | 7.472 | 12.604 | 74.433 | 0 | 0 |
| rate 250 req/s, 1 window | 15,001 | 250.0 | 1.544 | 2.322 | 29.851 | 140.472 | 0 | 0 |
| rate 500 req/s, 1 window | 30,000 | 500.0 | 0.954 | 1.206 | 5.970 | 137.223 | 0 | 0 |
| rate 1,000 req/s, 1 window | 59,918 | 998.6 | 0.600 | 0.650 | 45.670 | 177.108 | 0 | 83 |

The ladder stopped at 1,000 req/s, as the plan says it stops at the first failing step. The p99 at 250 req/s is higher than at 500, so a single 60 s run gives a noisy p99 tail; each step ran once. The server recorded 247,238 requests and 469,322 windows over the whole run, warm-up included, with no errors in its log.

## Drift check (200 batches of 256 windows each)

| data | KS p-value check (first) | calibrated check (served) |
|---|---|---|
| clean, val subjects | 200 alarms | 0 |
| clean, test subjects | 47 alarms | 0 |
| noise 0.1× channel std | 200 | 200 |
| noise 0.5× channel std | 200 | 200 |
| accelerometer gain ×1.1 | 200 | 200 |
| accelerometer gain ×1.5 | 200 | 200 |
| acc_z axis reads 0 | 200 | 200 |
| gyro_y axis reads 0 | 200 | 200 |

- **Calibration:** 5,000 clean batches from groups of 4 training subjects held out from the reference. The alarm threshold is a min-p of 8.0e-4, and the check fires on 0.92% of those null batches.
- **Order of decisions:** adoption was decided on the validation subjects before the test stage ran (commit 8f78d6a).
- **Limit:** every shift is caught on every batch, so this does not show the smallest fault the check can detect.

## Not verified

- CI has not run: GitHub Actions is refused for this account. The test suite was run locally only.
- The serving numbers come from one run per setting on a shared laptop, with k6 on the same machine.
- The drift evaluation covers 6 synthetic shifts. Real sensor faults were not tested.
