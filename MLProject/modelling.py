import os
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def load_data(data_dir: str):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def main():
    # JANGAN set_tracking_uri di sini (biar CI set via ENV)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "Student_Academic_Success_Basic"))

    data_dir = os.path.join(os.path.dirname(__file__), "namadataset_preprocessing")
    X_train, X_test, y_train, y_test = load_data(data_dir)

    # Jika sudah ada run dari MLflow Project, pakai itu. Kalau tidak ada, buat run.
    started_here = False
    if mlflow.active_run() is None:
        mlflow.start_run()
        started_here = True

    try:
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")

        mlflow.log_metric("test_accuracy", float(acc))
        mlflow.log_metric("test_f1_macro", float(f1m))

        # WAJIB untuk Kriteria 4: pastikan ada artifact "model"
        mlflow.sklearn.log_model(model, artifact_path="model")

        # artifacts
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Pred")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()

        os.makedirs("artifacts", exist_ok=True)
        cm_path = "artifacts/confusion_matrix.png"
        plt.savefig(cm_path, dpi=200)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="plots")

        report_path = "artifacts/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, y_pred))
        mlflow.log_artifact(report_path, artifact_path="reports")

        print("Done. acc=", acc, "f1_macro=", f1m)

    finally:
        if started_here:
            mlflow.end_run()


if __name__ == "__main__":
    main()
