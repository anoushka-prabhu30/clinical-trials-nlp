import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    print("Reading data/trials_clean.csv...")
    df = pd.read_csv("data/trials_clean.csv")
    print(f"  {len(df):,} rows loaded")

    print("Fitting TF-IDF vectorizer (max_features=5000, min_df=2)...")
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2)
    matrix = vectorizer.fit_transform(df["brief_description"])
    print(f"  Matrix shape: {matrix.shape}")

    matrix_path = "data/tfidf_matrix.npz"
    scipy.sparse.save_npz(matrix_path, matrix)
    print(f"Saved sparse matrix to {matrix_path}")

    feature_names = vectorizer.get_feature_names_out()
    names_path = "data/feature_names.txt"
    with open(names_path, "w", encoding="utf-8") as f:
        f.write("\n".join(feature_names))
    print(f"Saved {len(feature_names):,} feature names to {names_path}")


if __name__ == "__main__":
    main()
