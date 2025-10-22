import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42
TEST_SIZE = 0.2

@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    test_acc: float

def load_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    iris = datasets.load_iris(as_frame=False)
    X, y = iris.data, iris.target
    target_names = list(iris.target_names)
    return X, y, target_names

def train_models() -> Tuple[TrainedModel, List[TrainedModel], List[str]]:
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    candidates = [
        ("LogisticRegression",
         Pipeline([("scaler", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])),
        ("SVM",
         Pipeline([("scaler", StandardScaler()),
                   ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))])),
        ("RandomForest",
         Pipeline([("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))])),
    ]

    trained: List[TrainedModel] = []
    for name, pipe in candidates:
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        trained.append(TrainedModel(name=name, pipeline=pipe, test_acc=acc))

    best = max(trained, key=lambda tm: tm.test_acc)

    SHOW_REPORT = False
    if SHOW_REPORT:
        y_pred_best = best.pipeline.predict(X_test)
        print(f"[Best] {best.name} accuracy: {best.test_acc:.4f}", file=sys.stderr)
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_best), file=sys.stderr)
        print("Classification report:\n",
              classification_report(y_test, y_pred_best, target_names=target_names),
              file=sys.stderr)

    return best, trained, target_names

_BEST_MODEL, _ALL, _TARGET_NAMES = train_models()

def predict_flower(features: List[float]) -> str:
    """
    features = [sepal_length, sepal_width, petal_length, petal_width]
    return: species name (str)
    """
    arr = np.array(features, dtype=float).reshape(1, -1)
    idx = int(_BEST_MODEL.pipeline.predict(arr)[0])
    return _TARGET_NAMES[idx]

def main():
    line = sys.stdin.readline().strip()
    try:
        vals = [float(x) for x in line.split()]
        if len(vals) != 4:
            raise ValueError("Need exactly 4 numbers")
    except Exception:
        return
    print(predict_flower(vals))

if __name__ == "__main__":
    main()
