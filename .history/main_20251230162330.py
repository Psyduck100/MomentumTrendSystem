from momentum_program.config import AppConfig
from momentum_program.pipeline.runner import MomentumPipeline


def main() -> None:
    cfg = AppConfig()
    pipeline = MomentumPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
