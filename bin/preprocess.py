import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    text = " ".join(tokens)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("Reading data/trials_raw.csv...")
    df = pd.read_csv("data/trials_raw.csv")
    print(f"  {len(df):,} rows loaded")

    print("Cleaning brief_description...")
    df["brief_description"] = df["brief_description"].astype(str).apply(clean_text)

    before = len(df)
    df = df[df["brief_description"].str.len() >= 20].reset_index(drop=True)
    dropped = before - len(df)
    print(f"  Dropped {dropped:,} rows with cleaned text under 20 characters")
    print(f"  {len(df):,} rows remain after cleaning")

    out_path = "data/trials_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
