import argparse
import plotly.express as px
import polars as pl
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, default=6)
    parser.add_argument("--input_file", type=str, default="coarsened.csv")
    parser.add_argument("--algorithm", type=str, default="kmeans", choices=["kmeans", "spectral"])
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=1.0)
    args = parser.parse_args()

    lf = pl.scan_csv(args.input_file)
    df = lf.collect()
    print(df.select(["so", "s2", "o2", "eps"]).describe())

    x = df.select(["so", "s2", "o2", "eps"])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    if args.algorithm == "kmeans":
        model = KMeans(n_clusters=args.n_clusters, random_state=42)
    elif args.algorithm == "spectral":
        model = SpectralClustering(n_clusters=args.n_clusters, random_state=42)
    model.fit(x_scaled)

    lf_clustered = df.with_columns(pl.Series("cluster", model.labels_)).lazy()

    def padded_range(series: pl.Series, padding: float = 0) -> list:
        min_val = series.min()
        max_val = series.max()
        range_val = max_val - min_val
        return [min_val - padding * range_val, max_val + padding * range_val]

    for c in range(args.n_clusters):
        print("-" * 50)
        print(f"Cluster {c}")
        df_clustered = lf_clustered.filter(pl.col("cluster") == c).select(["i", "j", "k"]).collect()
        print(df_clustered.describe())
        if df_clustered.count() <= 10:
            print(f"\nCluster {c} has 10 or fewer points.")
            for row in df_clustered.iter_rows():
                print(row)

    df_filtered = lf_clustered.filter(pl.col("eps") > args.threshold).collect()

    fig = px.scatter_3d(
        df_filtered,
        x='i',
        y='j',
        z='k',
        color='eps',
        symbol='cluster',
        range_color=padded_range(df_filtered["eps"])
    )
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(range=[0, args.grid_size], autorange=False),
            yaxis=dict(range=[0, args.grid_size], autorange=False),
            zaxis=dict(range=[0, args.grid_size], autorange=False),
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            title_text='Cluster'
        )
    )

    name = str.replace(args.input_file, '.csv', '')

    html_output_file = f"clustered_{args.algorithm}_{name}_{args.n_clusters}.html"
    fig.write_html(html_output_file)

    df = lf_clustered.collect()

    fig2 = px.scatter_3d(
        df,
        x="so",
        y="s2",
        z="o2",
        color="eps",
        symbol="cluster",
        range_color=padded_range(df["eps"])
    )
    fig2.update_layout(
        scene=dict(
            xaxis=dict(range=padded_range(df["so"], padding=0.1)),
            yaxis=dict(range=padded_range(df["s2"], padding=0.1)),
            zaxis=dict(range=padded_range(df["o2"], padding=0.1)),
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            title_text='Cluster'
        )
    )

    html_output_file = f"clustered_{args.algorithm}_{name}_{args.n_clusters}_2.html"
    fig2.write_html(html_output_file)


if __name__ == "__main__":
    main()
