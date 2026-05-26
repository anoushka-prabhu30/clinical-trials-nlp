import pandas as pd


def main():
    print("Reading studies.txt...")
    studies = pd.read_csv(
        "data/studies.txt",
        sep="|",
        usecols=["nct_id", "official_title", "overall_status", "phase"],
        low_memory=False,
    )
    print(f"  Loaded {len(studies):,} study rows")

    print("Reading brief_summaries.txt...")
    summaries = pd.read_csv(
        "data/brief_summaries.txt",
        sep="|",
        usecols=["nct_id", "description"],
        low_memory=False,
    )
    summaries = summaries.rename(columns={"description": "brief_description"})
    print(f"  Loaded {len(summaries):,} summary rows")

    print("Joining on nct_id...")
    merged = studies.merge(summaries, on="nct_id", how="inner")
    print(f"  {len(merged):,} rows after join")

    print("Filtering rows with missing brief_description...")
    merged = merged[merged["brief_description"].notna()]
    print(f"  {len(merged):,} rows after filter")

    print("Sampling up to 5000 rows (random_state=42)...")
    n = min(5000, len(merged))
    sampled = merged.sample(n=n, random_state=42)
    print(f"  {len(sampled):,} rows sampled")

    cols = ["nct_id", "official_title", "overall_status", "phase", "brief_description"]
    sampled = sampled[cols]

    out_path = "data/trials_raw.csv"
    sampled.to_csv(out_path, index=False)
    print(f"Saved {len(sampled):,} rows to {out_path}")


if __name__ == "__main__":
    main()
