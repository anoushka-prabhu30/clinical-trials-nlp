# Clinical Trials NLP Clustering

## Background

ClinicalTrials.gov registers over 400,000 studies and publishes the full dataset as a bulk export. Each trial has structured fields (phase, status, condition, intervention) plus a free-text brief description written by the investigators. The structured fields are useful but coarse, a trial listed under "Type 2 Diabetes" might actually be about cardiovascular endpoints, gut microbiome changes, or health disparities depending on what the protocol is really doing. The brief descriptions tend to capture that finer-grained intent. This project runs NLP clustering on those descriptions to see whether the language organizes into themes that cut across or subdivide the official condition categories.

## Research Question

Can k-means clustering on TF-IDF embeddings of trial descriptions surface thematic subgroups that don't show up in the official condition labels?

## Methods

The pipeline runs in five sequential stages:

1. **Download** — joins `studies.txt` and `brief_summaries.txt` from the ClinicalTrials.gov bulk export on `nct_id`, filters to trials with non-null descriptions, and samples 5,000 trials for analysis.
2. **Preprocess** — lowercases text, removes punctuation, digits, and English stopwords (NLTK), and drops any trial whose cleaned description falls below 20 characters.
3. **TF-IDF embedding** — fits a `TfidfVectorizer` with `max_features=5000` and `min_df=2` on the cleaned descriptions, producing a sparse document term matrix.
4. **K-means clustering** — sweeps `k` from 5 to 15, selects the value with the highest silhouette score, and records the top-15 TF-IDF centroid keywords per cluster.
5. **t-SNE visualization** — reduces the TF-IDF matrix to 2D (`n_neighbors=15`, `min_dist=0.1`) and renders an interactive Plotly scatter plot colored by cluster, with per-point hover text showing the trial title and top-5 cluster keywords.

## Key Finding

K-means (k=14) on TF-IDF embeddings split oncology trials into 3 mechanistically distinct clusters, chemotherapy toxicity (cluster 0), tumor biology (cluster 2), and Phase I dosee scalation (cluster 4), subgroups that do not map to any single MeSH condition category. Metabolic trials (cluster 7: diabetes/insulin) and lifestyle intervention trials (cluster 5: exercise/rehabilitation) also emerged as coherent independent themes. This suggests that free text trial descriptions carry thematic signal beyond official condition labels.

## How to Reproduce

Install [Nextflow](https://www.nextflow.io/) and [Docker](https://www.docker.com/), place the ClinicalTrials.gov bulk export files (`studies.txt`, `brief_summaries.txt`) in `data/`, then run:

```bash
nextflow run main.nf
```

The interactive visualization is written to `output/clusters.html`.

## Repository Structure

```
clinical-trials-nlp/
├── bin/
│   ├── download.py        # join & sample raw trial data
│   ├── preprocess.py      # clean brief descriptions
│   ├── embed.py           # fit TF-IDF vectorizer
│   ├── cluster.py         # k-means sweep + silhouette selection
│   └── visualize.py       # t-SNE + interactive Plotly scatter
├── data/                  # input & intermediate files (gitignored)
├── output/                # clusters.html (gitignored)
├── main.nf                # Nextflow pipeline definition
├── nextflow.config        # Docker execution profile
├── requirements.txt       # Python dependencies
└── .gitignore
```
