# SPDX-License-Identifier: MPL-2.0
"""Export the published engine with phase timers for the second truncated study."""
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parents[1] / "truncated_search/profile"
sys.path.insert(0, str(PREVIOUS))
import instrument

REVISION = "1ecca3902fc7d04e35c43b9fcd3ec45920d0fcb5"


def main():
    instrument.REVISION = REVISION
    instrument.main()
    path = Path(sys.argv[1]).resolve() / "cpp/src/rollout.cpp"
    source = path.read_text()
    # Separate evaluation of the 21 rolls from cache lookup, the mean,
    # and cube conversions. Legal move generation already has its own timer.
    markers = [
        ("vr_luck, truncation, other, count", "vr_luck, truncation, other, fan, count"),
        ("                    for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {\n"
         "                        roll_best_probs[i] = best_move_probs_for_candidates(",
         "                    timer.next(Phase::fan);\n"
         "                    for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {\n"
         "                        roll_best_probs[i] = best_move_probs_for_candidates("),
        ("                    if (cache) {\n                        cache->publish(",
         "                    timer.next(Phase::vr_mean);\n"
         "                    if (cache) {\n                        cache->publish("),
    ]
    for old, new in markers:
        if old.startswith(" "):
            old, new = "\n" + old, "\n" + new
        if source.count(old) != 1:
            raise ValueError(f"unexpected source anchor {old!r}")
        source = source.replace(old, new)
    path.write_text(source)


if __name__ == "__main__":
    main()
