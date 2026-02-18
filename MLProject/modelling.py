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
    # Tracking lokal di folder project ini
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Student_Academic_Success_Basic")

    data_dir = os.path.join(os.path.dirname(__file__), "namadataset_preprocessing")
    X_train, X_test, y_train, y_test = load_data(data_dir)

    mlflow.sklearn.autolog(log_models=True)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    mlflow.log_metric("test_accuracy_manual", float(acc))
    mlflow.log_metric("test_f1_macro_manual", float(f1m))

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
    mlflow.log_artifact(cm_path)

    report_path = "artifacts/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact(report_path)

    print("Done. acc=", acc, "f1_macro=", f1m)

if __name__ == "__main__":
    main()
