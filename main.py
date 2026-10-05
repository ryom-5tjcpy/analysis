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
    args = parser.parse_args()

    lf = pl.scan_csv(args.input_file)
    df = lf.collect()

    x = df.select(["so", "s2", "o2", "eps"])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    if args.algorithm == "kmeans":
        model = KMeans(n_clusters=args.n_clusters, random_state=42)
    elif args.algorithm == "spectral":
        model = SpectralClustering(n_clusters=args.n_clusters, random_state=42)
    model.fit(x_scaled)

    df = df.with_columns(pl.Series("cluster", model.labels_))

    fig = px.scatter_3d(
        df,
        x='i',
        y='j',
        z='k',
        color='eps',
        symbol='cluster'
    )
    fig.update_layout(
        scene=dict(
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

    html_output_file = f"clustered_{args.algorithm}_{str.replace(args.input_file, '.csv', '')}_{args.n_clusters}.html"
    fig.write_html(html_output_file)


if __name__ == "__main__":
    main()
