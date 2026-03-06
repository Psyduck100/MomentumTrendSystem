from pathlib import Path

import pandas as pd

from momentum_program.config import AppConfig
from momentum_program.pipeline.runner import MomentumPipeline


def load_tickers_from_csv(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "ticker" in df.columns:
        return df["ticker"].dropna().astype(str).tolist()
    # fallback: first column if no header
    return df.iloc[:, 0].dropna().astype(str).tolist()


def main() -> None:
    cfg = AppConfig()
    csv_path = Path("etfs.csv")
    tickers = load_tickers_from_csv(csv_path)
    if tickers:
        cfg.data.tickers = tickers
    pipeline = MomentumPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
