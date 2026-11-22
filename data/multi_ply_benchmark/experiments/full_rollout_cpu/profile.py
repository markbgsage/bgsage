# SPDX-License-Identifier: MPL-2.0
"""Add worker phase timers to an isolated export of the full-rollout engine."""
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PHASES = ("prefill0", "prefill1", "cube", "movegen", "vr_mean", "select",
          "vr_luck", "truncation", "other", "fan", "cube_race", "select_race")


def instrument(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT / "build"):
        raise ValueError("Instrument a source export under build/")
    original = (HERE.parent / "truncated_search/profile/phase_timer.inc").read_text()
    timer = original.replace("truncation, other, count", "truncation, other, fan, cube_race, select_race, count")
    source = path.read_text()

    def insert(anchor, value, count=1):
        nonlocal source
        if source.count(anchor) != count:
            raise ValueError(f"Expected {count} occurrences of {anchor!r}")
        source = source.replace(anchor, value + "\n" + anchor)

    insert("namespace bgbot {", timer)
    insert("    TrialMoveScratch<std::array<std::vector<Board>, 21>> scratch;\n    auto& move_candidates", "    PhaseTimer timer(Phase::other);")
    insert("        const Move1Cache::Entry* move1_entry = nullptr;", "        timer.next(Phase::other);")
    for marker, phase in (("Phase 1: Cube check", "is_race(board) ? Phase::cube_race : Phase::cube"),
                          ("Phase 2: Generate moves", "Phase::movegen"), ("Phase 3: VR mean", "Phase::vr_mean"),
                          ("Only the played roll needs legal boards", "Phase::movegen"),
                          ("Per-branch cubeful VR mean", "Phase::vr_mean"),
                          ("Phase 4: Pick best move", "is_race(board) ? Phase::select_race : Phase::select"),
                          ("Phase 4b: VR luck", "Phase::vr_luck"), ("Phase 5: Terminal check", "Phase::other")):
        insert("\n        // " + marker, f"\n        timer.next({phase});")
    insert("                    evaluate_vr_rolls(board, move_candidates", "                    timer.next(Phase::fan);")
    insert("                    if (cache) {\n                        cache->publish", "                    timer.next(Phase::vr_mean);")
    insert("    // `board` is from the next mover's perspective", "    timer.next(Phase::truncation);")
    insert("    const Board& move0_chosen = move0_cache.chosen[first_roll_idx];", "    PhaseTimer timer(Phase::prefill1);")
    insert("    auto compute_roll = [&](int roll_idx) {\n        if (shared)", "    // time individual first rolls below")
    source = source.replace("    auto compute_roll = [&](int roll_idx) {\n        if (shared)",
                            "    auto compute_roll = [&](int roll_idx) {\n        PhaseTimer timer(Phase::prefill0);\n        if (shared)")
    mark = "while ((r = next_roll.fetch_add(1, std::memory_order_relaxed)) < 21) {"
    if source.count(mark) != 2:
        raise ValueError("Unexpected prefill dispatch sites")
    source = source.replace(mark, mark + "\n                PhaseTimer first_roll(Phase::prefill0);")
    insert("                move0_cache.state[r].store(2, std::memory_order_release);", "                first_roll.finish();", 2)
    path.write_text(source)


if __name__ == "__main__":
    instrument(Path(sys.argv[1]) / "cpp/src/rollout.cpp")
