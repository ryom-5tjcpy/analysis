import argparse
import polars as pl
from polars import col
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=6)
    parser.add_argument("--input_file", type=str, default="coarsened.csv")
    parser.add_argument("--algorithm", type=str, default="kmeans", choices=["kmeans", "spectral"])
    args = parser.parse_args()

    lf = pl.scan_csv(args.input_file)
    df = lf.select([col("so"), col("s2"), col("o2"), col("eps")]).collect()

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    if args.algorithm == "kmeans":
        model = KMeans(n_clusters=args.n_clusters, random_state=42)
    elif args.algorithm == "spectral":
        model = SpectralClustering(n_clusters=args.n_clusters, random_state=42)
    model.fit(df_scaled)

    df = df.with_columns(pl.Series("cluster", model.labels_))
    output_file = f"clustered_{args.algorithm}_{str.replace(args.input_file, '.csv', '')}_{args.n_clusters}"
    df.write_csv(output_file)


if __name__ == "__main__":
    main()
