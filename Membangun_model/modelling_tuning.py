import json
import os
from datetime import datetime

import numpy as np
import mlflow
from mlflow.models.signature import infer_signature

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt


def load_data(prep_dir: str):
    X_train = np.load(os.path.join(prep_dir, "X_train.npy"))
    X_test = np.load(os.path.join(prep_dir, "X_test.npy"))
    y_train = np.load(os.path.join(prep_dir, "y_train.npy"))
    y_test = np.load(os.path.join(prep_dir, "y_test.npy"))

    label_mapping_path = os.path.join(prep_dir, "label_mapping.json")
    if os.path.exists(label_mapping_path):
        with open(label_mapping_path, "r") as f:
            label_mapping = json.load(f)
    else:
        label_mapping = None

    return X_train, X_test, y_train, y_test, label_mapping


def save_confusion_matrix_png(cm, out_path, labels=None, title="Confusion Matrix"):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.colorbar()

    # Optional: ticks
    if labels is not None:
        plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
        plt.yticks(range(len(labels)), labels)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    # ====== SETTING PATH ======
    PREP_DIR = os.path.join(os.path.dirname(__file__), "namadataset_preprocessing")
    ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts_tuning")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # ====== MLFLOW SETUP ======
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Student_Academic_Success_Tuning")

    X_train, X_test, y_train, y_test, label_mapping = load_data(PREP_DIR)

    # label order untuk plot (opsional)
    inv_map = None
    label_names = None
    if label_mapping:
        # label_mapping biasanya {"Dropout":0,"Enrolled":1,"Graduate":2}
        inv_map = {int(v): k for k, v in label_mapping.items()}
        label_names = [inv_map[i] for i in sorted(inv_map.keys())]

    # ====== MODEL + TUNING ======
    # Fokus: improve kelas minoritas -> class_weight
    base_model = LogisticRegression(
        max_iter=2000,
        multi_class="auto",
        n_jobs=None,  # macOS kadang lebih aman None
    )

    param_grid = {
        "C": [0.1, 0.5, 1.0, 2.0, 5.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "class_weight": [None],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="accuracy",     # penting buat multi-class imbalanced
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    with mlflow.start_run(run_name=f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # ====== LOG PARAM GRID INFO (optional) ======
        mlflow.log_param("tuning_scoring", "accuracy")
        mlflow.log_param("cv_folds", 5)

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_cv_score = grid.best_score_

        # ====== LOG BEST PARAMS ======
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)
        mlflow.log_metric("best_cv_accuracy", float(best_cv_score))

        # ====== EVALUATION ======
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracy = f1_score(y_test, y_pred, average="macro")
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)

        mlflow.log_metric("test_accuracy", float(acc))
        mlflow.log_metric("test_accuracy", float(accuracy))
        mlflow.log_metric("test_precision_macro", float(prec_macro))
        mlflow.log_metric("test_recall_macro", float(rec_macro))

        # ====== ARTIFACTS ======
        report_path = os.path.join(ARTIFACT_DIR, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, y_pred))
        mlflow.log_artifact(report_path, artifact_path="reports")

        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
        save_confusion_matrix_png(cm, cm_path, labels=label_names)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # ====== LOG MODEL ======
        signature = infer_signature(X_test, best_model.predict(X_test))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_test[:5],
        )

        print("Done.")
        print("Best params:", best_params)
        print("Best CV accuracy:", best_cv_score)
        print("Test acc:", acc)
        print("Test accuracy:", accuracy)


if __name__ == "__main__":
    main()