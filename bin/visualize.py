import json

import pandas as pd
import plotly.express as px
import scipy.sparse
from sklearn.manifold import TSNE


def main():
    print("Loading data...")
    matrix = scipy.sparse.load_npz("data/tfidf_matrix.npz")
    labels_df = pd.read_csv("data/cluster_labels.csv")
    trials_df = pd.read_csv("data/trials_clean.csv", usecols=["nct_id", "official_title"])

    with open("data/cluster_keywords.json", encoding="utf-8") as f:
        cluster_keywords: dict = json.load(f)

    print(f"  Matrix shape: {matrix.shape}")

    print("Running TSNE (n_components=2, perplexity=30)...")
    reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    embedding = reducer.fit_transform(matrix.toarray())
    print("  TSNE done")

    df = labels_df.copy()
    df["x"] = embedding[:, 0]
    df["y"] = embedding[:, 1]
    df = df.merge(trials_df, on="nct_id", how="left")

    def top5_keywords(cluster_id: int) -> str:
        keywords = cluster_keywords.get(str(cluster_id), [])
        return ", ".join(keywords[:5])

    df["keywords"] = df["cluster"].apply(top5_keywords)
    df["hover"] = (
        "<b>" + df["official_title"].fillna("(no title)") + "</b>"
        + "<br>Cluster " + df["cluster"].astype(str)
        + "<br><i>" + df["keywords"] + "</i>"
    )

    print("Building plotly figure...")
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=df["cluster"].astype(str),
        custom_data=["hover"],
        title="Clinical Trial Clusters (TSNE + KMeans)",
        labels={"color": "Cluster"},
    )
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        marker=dict(size=5, opacity=0.7),
    )
    fig.update_layout(
        xaxis_title="TSNE-1",
        yaxis_title="TSNE-2",
        legend_title="Cluster",
        hoverlabel=dict(bgcolor="white", font_size=12),
    )

    out_path = "output/clusters.html"
    fig.write_html(out_path)
    print(f"Saved interactive plot to {out_path}")


if __name__ == "__main__":
    main()
