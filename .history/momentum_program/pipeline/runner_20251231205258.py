import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from momentum_program.config import AppConfig
from momentum_program.strategy.ranking import build_targets
from momentum_program.universe.base import UniverseProvider
from momentum_program.data_providers.base import DataProvider


class MomentumPipeline:
    def __init__(
        self,
        cfg: AppConfig,
        universe_provider: UniverseProvider,
        price_provider: DataProvider,
    ) -> None:
        self.cfg = cfg
        self.universe_provider = universe_provider
        self.price_provider = price_provider
        self.state_path = Path(self.cfg.data.data_dir) / "last_portfolio.json"

    def run(self) -> None:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.cfg.data.lookback_days)
        tickers = self.universe_provider.get_tickers()
        bucket_map = self.universe_provider.get_bucket_map()
        if not tickers:
            tickers = self.cfg.data.tickers
            bucket_map = {}
        price_series = self.price_provider.fetch(tickers, start, end)
        prev_selection = self._load_prev_selection()
        targets = build_targets(
            price_series,
            self.cfg.strategy,
            end,
            bucket_map,
            prev_bucket_selection=prev_selection,
        )
        self._emit_targets(targets)
        self._save_selection(targets)

    def _emit_targets(self, targets) -> None:
        # Skeleton placeholder; wire into broker or reporting later.
        print(f"Target weights as of {targets.as_of:%Y-%m-%d} -> {targets.weights}")
        if targets.bucket_rankings:
            print("Bucket rankings (top picks per bucket):")
            for bucket, symbols in targets.bucket_rankings.items():
                scores = (
                    targets.bucket_scores.get(bucket) if targets.bucket_scores else None
                )
                if scores:
                    scored = [f"{sym} ({score:.4f})" for sym, score in scores]
                    print(f"  {bucket}: {', '.join(scored)}")
                else:
                    print(f"  {bucket}: {symbols}")

    def _load_prev_selection(self) -> dict[str, str | None]:
        if not self.state_path.exists():
            return {}
        try:
            with self.state_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("bucket_selection", {}) or {}
        except Exception:
            return {}

    def _save_selection(self, targets) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "as_of": targets.as_of.isoformat(),
                "bucket_selection": targets.bucket_selection or {},
            }
            with self.state_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            # Persistence failures should not break live signal generation
            pass
