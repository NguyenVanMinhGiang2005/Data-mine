import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from cau1 import main, PATH

df = pd.read_csv(PATH)

def main():
    label_col = "Class"  # tên nhãn trong file của bạn
    feature_cols = [
    '1) Alcohol','2) Malic acid','3) Ash','4) Alcalinity of ash  ',
    '5) Magnesium','6) Total phenols','7) Flavanoids','8) Nonflavanoid phenols',
    '9) Proanthocyanins','10)Color intensity','11)Hue','12)OD280/OD315 of diluted wines'
    ]
    X = df[feature_cols].copy()
    y = df[label_col].copy()

    X = X.fillna(X.mean(numeric_only=True))      # điền thiếu (nếu có)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)              # z-score

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_std)              # nhận PC1, PC2
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    viz = pd.DataFrame({"PC1": X_2d[:,0], "PC2": X_2d[:,1], label_col: y})
    viz.to_csv("C:\\Users\\nguye\\Documents\\Khai thac du lieu\\DOAN\\csv\\wine_2d_pca.csv", index=False)   # lưu lại nếu cần

    plt.figure()
    for cls in sorted(viz[label_col].unique()):
        mask = viz[label_col] == cls
        plt.scatter(viz.loc[mask, "PC1"], viz.loc[mask, "PC2"], s=18, label=f"{label_col}={cls}")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA to 2D")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()