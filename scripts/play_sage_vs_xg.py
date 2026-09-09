# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Retired replay-based XG benchmark entry point.

This implementation could not certify complete XG decisions or matching game
rules. Its results must not be used for engine-strength comparisons. A live
match requires an external, validated XG controller; the library stays
self-contained and does not import a parent application's harness.
"""

RETIRED = (
    "The replay-based Sage/XG benchmark is retired: it allowed nonstandard "
    "openings, incomplete XG decisions, and skipped failed games. Use the "
    "checkpointed xg_h2h/lockstep_mp.py controller with explicit XG profile "
    "and a fresh output. Historical results cannot be resumed or pooled as validated data."
)


def play_single_game(*args, **kwargs):
    raise RuntimeError(RETIRED)


if __name__ == "__main__":
    raise SystemExit(RETIRED)
