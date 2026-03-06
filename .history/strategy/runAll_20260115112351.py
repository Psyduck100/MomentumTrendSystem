from __future__ import annotations

import os
import sys

if __name__ == "__main__" and __package__ is None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import strategy.CRYP as CRYP
import strategy.USEQ as USEQ
import strategy.runPMTL as runPMTL


def main() -> None:
    print("=" * 72)
    print("RUN ALL STRATEGIES")
    print("=" * 72)
    print()

    print("CRYP")
    CRYP.main()
    print()

    print("PMTL")
    runPMTL.main()
    print()

    print("USEQ")
    USEQ.main()


if __name__ == "__main__":
    main()
