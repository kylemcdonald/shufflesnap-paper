"""Download MNIST / Fashion-MNIST from OpenML and compute 2D UMAP embeddings.

Outputs data/<name>_umap.npz with arrays `xy` (float64, n x 2) and `labels` (int).
Deterministic given the fixed UMAP random_state (single-threaded UMAP when seeded).
"""
import sys
import time

import numpy as np


def main(name):
    from sklearn.datasets import fetch_openml
    import umap

    openml = {"mnist": "mnist_784", "fashion": "Fashion-MNIST"}[name]
    t = time.time()
    X, y = fetch_openml(openml, version=1, return_X_y=True, as_frame=False, data_home="data/openml")
    print(name, X.shape, "download/load", round(time.time() - t, 1), "s", flush=True)
    X = X.astype(np.float32) / 255.0
    t = time.time()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    xy = reducer.fit_transform(X)
    print(name, "umap", round(time.time() - t, 1), "s", flush=True)
    np.savez_compressed(f"data/{name}_umap.npz", xy=xy.astype(np.float64), labels=y.astype(int),
                        params=np.array(["n_neighbors=15,min_dist=0.1,random_state=42,umap-learn"]))


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
