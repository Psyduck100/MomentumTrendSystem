"""Plot stitched walk-forward equity curves and drawdowns for quick diagnostics."""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _load_stitched_returns(csv_path: Path) -> Tuple[pd.Series, pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)
    if "returns_path" not in df.columns:
        raise ValueError(f"Column 'returns_path' missing in {csv_path}")

    segments: List[pd.Series] = []
    for raw in df["returns_path"]:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Missing return file: {path}")
        series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        series.name = "return"
        segments.append(series)

    combined = pd.concat(segments).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    equity = (1 + combined).cumprod()
    drawdown = equity / equity.cummax() - 1
    return combined, equity, drawdown


def plot_walkforward_curves(csv_paths: List[Path], output: Path) -> None:
    n = len(csv_paths)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.0 * n), sharex="col")
    if n == 1:
        axes = axes.reshape(1, 2)

    for row, csv_path in enumerate(csv_paths):
        returns, equity, drawdown = _load_stitched_returns(csv_path)
        label = csv_path.stem.replace("walkforward_", "")

        ax_eq = axes[row][0]
        ax_dd = axes[row][1]

        ax_eq.plot(equity.index, equity.values, label=f"Equity ({label})", color="tab:blue")
        ax_eq.set_ylabel("Growth (×)")
        ax_eq.set_title(f"{label} cumulative return")
        ax_eq.grid(True, alpha=0.3)

        ax_dd.fill_between(drawdown.index, drawdown.values * 100, color="tab:red", alpha=0.3)
        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.set_title(f"{label} drawdown")
        ax_dd.grid(True, alpha=0.3)
        ax_dd.set_ylim(min(drawdown.min() * 100 * 1.1, -1), 5)

        max_dd_pct = drawdown.min() * 100
        ax_dd.annotate(
            f"Min DD: {max_dd_pct:.1f}%",
            xy=(drawdown.idxmin(), max_dd_pct),
            xytext=(10, -10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="tab:red"),
            fontsize=8,
        )

    axes[-1][0].set_xlabel("Date")
    axes[-1][1].set_xlabel("Date")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot walk-forward stitched equity curves")
    parser.add_argument("csv_files", nargs="+", help="Paths to walk-forward CSV files")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/walkforward_curves.png"),
        help="Destination image path",
    )
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv_files]
    plot_walkforward_curves(csv_paths, args.output)


if __name__ == "__main__":
    main()
