#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Score XG (eXtreme Gammon) against the PASKOGAMMON benchmark via Batch-Analyze results.

Paskogammon twin of :mod:`benchmark_pr_xg`. The logic is identical -- read XG's #1
ranked decision (best move / recommended cube action) for each position out of the
``.xg`` files XG's *Batch Analyze* produced, and score *that* decision against the
benchmark's saved rollout reference, exactly as :func:`benchmark_pasko.benchmark_pr`
scores a live bot. XG's own equities are ignored; only its chosen decision is used.

Two Paskogammon-specific differences from the money scorer:

  * **Native ``.xg`` files.** The ``.txt`` transcripts can't round-trip through XG for
    the non-standard Paskogammon start, so scoring reads the native ``.xg`` archives
    written by ``export_pasko_benchmark_xg.py`` (default dir ``xg_native/``), not a
    ``.txt``-derived folder.
  * **Pasko dataset + stage1.** Matching uses the pasko stage1 capture
    (``data/pasko_money_benchmark/build/stage1``) and the reference comes from the
    pasko ``benchmark.json``. Both are redirected here; every parse/match/score helper
    is reused unchanged from :mod:`benchmark_pr_xg`.

**Multi-level XG testing.** XG Batch Analyze writes its analysis INTO each ``.xg``
(one analysis per record -- re-analyzing overwrites it) and skips already-analyzed
positions, so each XG eval level needs its **own folder copy** of the native games.
Stamp them with ``stamp_xg_levels.py``, Batch-Analyze each copy at the matching XG
level ("Save Games after analyze" ON), then point ``--xg-dir`` at each. The analyzed
``.xg`` folder IS the durable per-level cache -- re-harvesting a level is free.

Usage::

    # after XG Batch-Analyzes each per-level copy:
    python scripts/benchmark_pr_xg_pasko.py --xg-dir data/pasko_money_benchmark/xg_native_xg3ply
    python scripts/benchmark_pr_xg_pasko.py --xg-dir data/pasko_money_benchmark/xg_native_xgrollerplus
"""

import argparse
import logging
import os
import sys
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

# benchmark_pasko sets up the pasko on-disk paths; benchmark_pr_xg supplies the
# parse/match/score machinery. Both defer their bgsage imports, so importing them
# here does not load the engine (this script is pure parse + match + score).
import benchmark_pasko as bp        # noqa: E402
import benchmark_pr_xg as xgm       # noqa: E402

log = logging.getLogger("benchmark_pr_xg_pasko")

# Redirect the reused money scorer at the pasko benchmark: the seed-lookup reads the
# pasko stage1 capture, and the dataset default becomes the pasko benchmark.json.
# (``xgm.benchmark_pr`` / ``xgm._seed_lookup`` reference these module globals at call
# time, so reassigning them before the call is sufficient and reuses all the logic.)
xgm._STAGE1_DIR = bp._STAGE1_DIR
xgm.DEFAULT_DATASET = bp.DEFAULT_DATASET

#: Default source of native .xg games: the pristine (unanalyzed) master. In practice
#: pass ``--xg-dir`` a per-level analyzed copy (xg_native_<tag>/).
DEFAULT_XG_DIR = bp._DATA_DIR / "xg_native"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score XG against the Paskogammon benchmark via analyzed .xg files")
    parser.add_argument("--xg-dir", type=Path, default=DEFAULT_XG_DIR,
                        help="Folder of analyzed seed_<N>.xg files for ONE XG eval level "
                             f"(default: {DEFAULT_XG_DIR.name}/ -- the unanalyzed master)")
    parser.add_argument("--dataset", type=Path, default=bp.DEFAULT_DATASET,
                        help="Pasko benchmark dataset JSON")
    parser.add_argument("--max-seed", type=int, default=None,
                        help="Only score the first N games (.xg files with seed <= MAX_SEED)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    result = xgm.benchmark_pr(xg_dir=args.xg_dir, dataset_path=args.dataset,
                              max_seed=args.max_seed)
    print(f"\nScored: XG (eXtreme Gammon) over {result['n_games']} games  [{args.xg_dir.name}]")
    xgm.bm._print_report(result)
    if result.get("unmatched"):
        print(f"Outside benchmark: {result['unmatched']} XG decisions are positions the "
              f"dataset does not capture (forced moves, trivial spreads/cubes) -- not scored")


if __name__ == "__main__":
    main()
