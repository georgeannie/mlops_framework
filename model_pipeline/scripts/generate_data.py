# data_gen.py
import numpy as np
import pandas as pd

def generate_classification_data(n_samples=1000, n_features=10, random_state=42):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                               n_informative=6, n_redundant=2, n_classes=2,
                               random_state=random_state)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    df.to_csv("./data/train_data.csv", index=False)
    print("Data saved to train_data.csv")

if __name__ == "__main__":
    generate_classification_data()
