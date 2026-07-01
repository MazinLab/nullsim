"""Allow ``python -m nullsim`` to drive the same CLI as the console script."""

from __future__ import annotations

import sys

from nullsim.cli import main

if __name__ == "__main__":
    sys.exit(main())
