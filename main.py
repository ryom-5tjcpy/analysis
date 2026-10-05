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

    for c in range(args.n_clusters):
        print("-" * 50)
        print(f"Cluster {c}")
        df_clustered = lf_clustered.filter(pl.col("cluster") == c).select(["so", "s2", "o2", "eps"]).collect()
        print(df_clustered.describe())

    fig = px.scatter_3d(
        lf_clustered.filter(pl.col("eps") > args.threshold).collect(),
        x='i',
        y='j',
        z='k',
        color='eps',
        symbol='cluster',
        range_color=[0, 30]
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
    fig.update_traces(marker=dict(cmin=0, cmax=30))

    html_output_file = f"clustered_{args.algorithm}_{str.replace(args.input_file, '.csv', '')}_{args.n_clusters}.html"
    fig.write_html(html_output_file)


if __name__ == "__main__":
    main()
