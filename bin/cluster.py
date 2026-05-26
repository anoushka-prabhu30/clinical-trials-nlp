import json

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_top_keywords(model, feature_names, n=15):
    keywords = {}
    centers = model.cluster_centers_
    for cluster_id, center in enumerate(centers):
        top_indices = np.argsort(center)[::-1][:n]
        keywords[cluster_id] = [feature_names[i] for i in top_indices]
    return keywords


def main():
    print("Loading data/tfidf_matrix.npz...")
    matrix = scipy.sparse.load_npz("data/tfidf_matrix.npz")
    print(f"  Matrix shape: {matrix.shape}")

    print("Loading nct_ids from data/trials_clean.csv...")
    nct_ids = pd.read_csv("data/trials_clean.csv", usecols=["nct_id"])["nct_id"]

    print("Loading feature names from data/feature_names.txt...")
    with open("data/feature_names.txt", encoding="utf-8") as f:
        feature_names = [line.rstrip("\n") for line in f]

    best_k = None
    best_score = -1
    best_model = None

    print("Running KMeans for k = 5 through 15...")
    for k in range(5, 16):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(matrix)
        score = silhouette_score(matrix, labels, sample_size=2000, random_state=42)
        print(f"  k={k:2d}  silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    print(f"\nBest k={best_k} with silhouette score={best_score:.4f}")

    labels = best_model.labels_

    labels_path = "data/cluster_labels.csv"
    pd.DataFrame({"nct_id": nct_ids, "cluster": labels}).to_csv(labels_path, index=False)
    print(f"Saved cluster labels to {labels_path}")

    keywords = get_top_keywords(best_model, feature_names, n=15)
    keywords_path = "data/cluster_keywords.json"
    with open(keywords_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in keywords.items()}, f, indent=2)
    print(f"Saved cluster keywords to {keywords_path}")


if __name__ == "__main__":
    main()
