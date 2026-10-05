import polars as pl
from polars import col
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def main():
    n_cluster = 6

    lf = pl.scan_csv("coarsened.csv")
    df = lf.select([col("so"), col("s2"), col("o2"), col("eps")]).collect()

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    model = KMeans(n_clusters=n_cluster, random_state=42)
    model.fit(df_scaled)

    df = df.with_columns(pl.Series("cluster", model.labels_))
    df.write_csv("clustered_kmeans.csv")


if __name__ == "__main__":
    main()
