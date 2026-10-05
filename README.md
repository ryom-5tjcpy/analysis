# CSVクラスタリング可視化

CSVファイルのデータをクラスタリングし、3次元散布図をHTMLファイルとして出力するスクリプトです。クラスタリングには `so`、`s2`、`o2`、`eps` の各値を標準化して使用します。可視化では `i`、`j`、`k` を座標、`eps` を色、クラスタ番号をマーカーの形で表示します。

## 必要環境

- Python 3.12 以上
- [uv](https://docs.astral.sh/uv/)

## セットアップ

リポジトリのルートディレクトリで依存パッケージをインストールします。

```bash
uv sync
```

## 入力CSV

入力CSVには、次の列を含めてください。

| 列 | 用途 |
| --- | --- |
| `i`、`j`、`k` | 3次元散布図の座標 |
| `so`、`s2`、`o2`、`eps` | クラスタリングに使う特徴量 |

## 実行方法

デフォルトでは、カレントディレクトリの `coarsened.csv` を読み込み、クラスタ数6、`kmeans` アルゴリズムで処理します。

```bash
uv run python main.py
```

入力ファイルやクラスタ数、アルゴリズムを指定する例です。

```bash
uv run python main.py --input_file data.csv --n_clusters 4 --algorithm spectral
```

### オプション

| オプション | デフォルト | 説明 |
| --- | --- | --- |
| `--input_file` | `coarsened.csv` | 入力CSVファイル |
| `--n_clusters` | `6` | クラスタ数 |
| `--algorithm` | `kmeans` | クラスタリング手法。`kmeans` または `spectral` |
| `--grid_size` | `64` | 3次元グラフの各軸の表示範囲（0から指定値まで） |

利用可能な引数は次のコマンドでも確認できます。

```bash
uv run python main.py --help
```

## 出力

実行すると、インタラクティブな3次元散布図を含むHTMLファイルが生成されます。ファイル名は次の形式です。

```text
clustered_<アルゴリズム>_<入力ファイル名>_<クラスタ数>.html
```

デフォルト設定では `clustered_kmeans_coarsened_6.html` が出力されます。HTMLファイルをブラウザーで開くと、散布図を操作して確認できます。