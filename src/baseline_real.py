import numpy as np
from sklearn.ensemble import RandomForestClassifier
from train_gnn_real import load_elliptic
from baseline import evaluate


def main():
    X, y, _, train, test = load_elliptic()
    y = np.where(y < 0, 0, y)

    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                 n_jobs=-1, random_state=0)
    clf.fit(X[train], y[train])
    pred = clf.predict(X[test])

    evaluate("Tabular baseline (RandomForest) on REAL Elliptic (temporal test)",
             y[test], pred)


if __name__ == "__main__":
    main()