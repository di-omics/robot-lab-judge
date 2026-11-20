"""CLI of the linear probe on frozen features.

  python -m labjudge.frozen extract --split train|dev   cache vision features under data/features/
  python -m labjudge.frozen leakcheck                  arm pixels, last against first top view
  python -m labjudge.frozen smoke                      20 train scenes: repeatability, difference, time
  python -m labjudge.frozen fit                        cross-validate on train, refit, predict dev once

Only the train and dev splits are allowed. The sealed test split is refused.
"""

import argparse
import sys

from . import ALLOWED, check_path, check_split


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="labjudge.frozen")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="encode train or dev and cache the features")
    e.add_argument("--split", required=True, help="train or dev")
    e.add_argument("--data", default=None, help="optional, must be data/simset/<split>")
    sub.add_parser("leakcheck", help="compare arm pixels of the last and first top view")
    s = sub.add_parser("smoke", help="smoke test of the encoder on train scenes")
    s.add_argument("--n", type=int, default=20)
    f = sub.add_parser("fit", help="cross-validation on train, refit, predict dev once, write verdicts and report")
    f.add_argument("--control-only", action="store_true", help="only the shuffled-label control, no dev")
    args = p.parse_args(argv)

    if args.cmd == "extract":
        split = check_split(args.split)
        if args.data is not None:
            data = check_path(args.data)
            if data.as_posix().rstrip("/") != f"data/simset/{split}":
                raise SystemExit(f"--data must be data/simset/{split}")
        from .extract import extract

        extract(split)
    elif args.cmd == "leakcheck":
        from .leak import run

        run()
    elif args.cmd == "smoke":
        from .extract import smoke

        smoke(args.n)
    elif args.cmd == "fit":
        from .fit import run

        run(control_only=args.control_only)
    assert set(ALLOWED) == {"train", "dev"}


if __name__ == "__main__":
    sys.exit(main())
