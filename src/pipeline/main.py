from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:

    argv = list(argv) if argv is not None else sys.argv[1:]
    print("ai-patterns-mining: entry point called with args:", argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
