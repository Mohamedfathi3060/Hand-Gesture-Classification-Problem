# Hand Gesture Recognition — ML Research

> Real-time hand gesture classification from 21 MediaPipe landmarks using SVM, Random Forest, and XGBoost, with full MLflow experiment tracking.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
  - [1. Preprocessing & Normalization](#1-preprocessing--normalization)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Model Training & Selection](#3-model-training--selection)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Model Comparison](#model-comparison)
- [Why SVM?](#why-svm)
- [Model Registry](#model-registry)
- [Real-Time Inference](#real-time-inference)
- [Setup & Usage](#setup--usage)

---

## Project Overview

This project builds a pipeline that:

1. Captures raw hand landmark coordinates (x, y, z) for 21 keypoints using **MediaPipe Hands**
2. Normalizes and engineers features to make them position- and scale-invariant
3. Trains and compares three classifiers — **SVM (RBF)**, **Random Forest**, and **XGBoost** — using 5-fold cross-validated GridSearchCV
4. Logs all experiments, parameters, metrics, artifacts, and models with **MLflow**
5. Runs real-time gesture prediction through a live webcam feed using OpenCV

---

## Repository Structure

```
├── ML_1_Project.ipynb          # Main research notebook
├── mlflow_utils.py             # All MLflow logging functions (imported by notebook)
├── hand_landmarks_data.csv     # Dataset: 21-landmark samples with gesture labels
├── mlruns/                     # MLflow tracking directory (auto-generated on run)
└── README.md
```

---

## Dataset

The dataset `hand_landmarks_data.csv` contains rows of 21 hand keypoints captured via MediaPipe, each with x, y, z coordinates (63 raw features) plus a `label` column for the gesture class.

| Property | Value |
|---|---|
| Features (raw) | 63 (x1–x21, y1–y21, z1–z21) |
| Features (after engineering) | 63 + 23 engineered = 86 (minus wrist cols = **83**) |
| Missing values | None |
| Class balance | Balanced across all gesture classes |

---

## Pipeline

### 1. Preprocessing & Normalization

Raw MediaPipe coordinates are screen-relative and vary with hand position and distance from the camera. Each sample is normalized to be **translation- and scale-invariant**:

- **Translation**: All 21 keypoints are shifted so the wrist (landmark 0) is at the origin
- **Scale**: All coordinates are divided by the Euclidean distance from the wrist to the middle fingertip (landmark 13, XY plane only), so every hand is the same "size"
- **Wrist columns dropped**: After normalization, `x1`, `y1`, `z1` are always zero and are removed

This ensures the model learns the *shape* of the gesture, not where the hand is on screen or how far it is from the camera.

### 2. Feature Engineering

Beyond the normalized landmark coordinates, the following features are computed to give the model explicit geometric cues:

| Feature Group | Description | Count |
|---|---|---|
| Adjacent fingertip distances | 3D Euclidean distance between consecutive fingertip pairs (thumb–index, index–middle, middle–ring, ring–pinky) | 4 |
| Adjacent MCP joint distances | Same pairs but at the MCP (knuckle) joints | 4 |
| Per-finger average x, y, z | Mean coordinate of the 4 joints in each finger | 15 |
| Average Z depth | Mean Z across all 21 landmarks (overall depth spread) | 1 |

### 3. Model Training & Selection

Three classifiers are evaluated using **GridSearchCV with 5-fold cross-validation**, scoring on accuracy, precision, recall, and F1 (macro). The data is split 80/20 for train/test; the training set is then further split 80/20 to produce a validation set for MLflow logging.

**Search spaces:**

| Model | Hyperparameter | Values Searched |
|---|---|---|
| SVM (RBF) | C | 350, 375, 400 |
| Random Forest | n_estimators | 150, 200, 300 |
| Random Forest | max_depth | 50, 80, 100, None |
| XGBoost | n_estimators | 300, 350 |
| XGBoost | max_depth | 5 |
| XGBoost | learning_rate | 0.2, 0.3 |

**Best hyperparameters found:**

| Model | Best Parameters | CV Accuracy |
|---|---|---|
| SVM (RBF) | C=375, kernel=rbf | ~0.9860 |
| Random Forest | max_depth=50, n_estimators=300 | ~0.9835 |
| XGBoost | learning_rate=0.2, max_depth=5, n_estimators=300 | ~0.9858 |

---

## MLflow Experiment Tracking

All logging is handled by `mlflow_utils.py`, which is imported into the notebook. **The utility module never trains models** — it only logs. Models are trained in the notebook first, then passed in as fitted objects.

**Experiment name:** `hand-gesture-recognition`

| Run name | Model | What is logged |
|---|---|---|
| `svm-rbf` | SVC (C=375, RBF) | dataset stats, params, val/test metrics, confusion matrices, classification reports, model artifact |
| `random-forest` | RandomForestClassifier (n=300, depth=50) | same as above |
| `xgboost` | XGBClassifier (n=300, lr=0.2, depth=5) | same as above |
| `model-comparison` | — | grouped bar chart (PNG) + comparison CSV |

**To launch the MLflow UI after running the notebook:**

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) to browse runs, compare metrics, and download artifacts.

---

## Model Comparison

The table below summarizes validation-set performance from the final trained models (best hyperparameters, no cross-validation averaging):

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **SVM (RBF, C=375)** | **~0.9873** | **~0.9871** | **~0.9869** | **~0.9870** |
| XGBoost (n=300, lr=0.2) | ~0.9839 | ~0.9837 | ~0.9835 | ~0.9836 |
| Random Forest (n=300, depth=50) | ~0.9798 | ~0.9795 | ~0.9793 | ~0.9794 |

> Exact values are logged per-run in MLflow. The chart below is generated and saved by `log_comparison_run()` inside the `model-comparison` run.

![Model Comparison Chart](mlruns/model_comparison.png)

*The grouped bar chart is logged as an artifact to MLflow under the `model-comparison` run → `plots/model_comparison.png`. You can view it directly in the MLflow UI.*

---

## Why SVM?

SVM with an RBF kernel was chosen as the best model for this task for the following reasons:

**1. Consistently highest scores across all metrics.**
SVM achieved the highest accuracy (~98.73%), precision, recall, and F1 on the held-out test set. The gap over XGBoost (~98.39%) and Random Forest (~97.98%) is small but consistent across every run and every metric, not just accuracy.

**2. Well-suited to the feature space.**
The engineered features (normalized landmark coordinates + geometric distances) form a relatively compact, structured space. SVM with an RBF kernel excels at finding non-linear decision boundaries in such spaces, and hand gesture classes naturally occupy compact, separable regions in landmark-coordinate space.

**3. No scaling needed.**
After wrist-centering and scale normalization, the features are already in a consistent range. This aligns well with SVM's sensitivity to feature scale — unlike unnormalized data, no `StandardScaler` step was required to achieve peak performance.

**4. Stable across cross-validation folds.**
GridSearchCV showed that SVM's best CV score (~0.9860) closely matched its test score (~0.9873), indicating low variance and good generalization — unlike tree-based methods which can overfit more easily with deep trees.

**5. Simpler model, easier to inspect.**
With only two hyperparameters (kernel, C), SVM is easier to reason about than an ensemble with hundreds of trees or a boosted model with many interaction terms. For a gesture recognition system that may need future debugging or auditing, this is a practical advantage.

---

## Model Registry

The best model (SVM RBF) is registered in the MLflow Model Registry under the name **`hand-gesture-svm-rbf`** and transitioned to the **Staging** stage, ready for integration or deployment.

```python
# Registered automatically by the notebook:
register_best_model(
    run_id        = svm_result['run_id'],
    artifact_path = 'model_svm-rbf',
)
```

To load it programmatically:

```python
from mlflow_utils import load_registered_model
model = load_registered_model()   # loads from 'Staging' by default
gesture = model.predict(features)
```

---

## Real-Time Inference

The final cells of the notebook run a live webcam loop using **OpenCV** and **MediaPipe Hands**:

1. Each frame is captured and flipped horizontally
2. MediaPipe detects the hand and extracts 21 landmark coordinates
3. The same normalization and feature engineering pipeline used during training is applied to the live landmarks
4. The fitted SVM predicts the gesture label, which is overlaid on the frame in real-time

Press `Esc` to exit the webcam window.

---

## Setup & Usage

**Install dependencies:**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost opencv-python mediapipe mlflow
```

**Run the notebook:**

```bash
jupyter notebook ML_1_Project.ipynb
```

Run all cells top to bottom. The MLflow logging cells (after the model training section) will automatically create the `mlruns/` directory and log all runs.

**Inspect results:**

```bash
mlflow ui
# Open http://localhost:5000
```
