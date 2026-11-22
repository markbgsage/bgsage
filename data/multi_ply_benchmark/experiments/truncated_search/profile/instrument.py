# SPDX-License-Identifier: MPL-2.0
"""Export the study control and instrument its rollout phases in a build directory."""
from io import BytesIO
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
REVISION = "60f93556137a7a5c4be671502f4b3f521998ff94"


def export_control(dest):
    if not dest.is_relative_to(ROOT / "build") or dest.exists():
        raise ValueError("Use a new subdirectory of build/")
    archive = subprocess.check_output(
        ["git", "archive", "--format=zip", REVISION, "cpp", "CMakeLists.txt"], cwd=ROOT)
    with ZipFile(BytesIO(archive)) as z:
        z.extractall(dest)
    (dest / "models").mkdir()
    for f in (ROOT / "models").glob("sl_s*.weights.best"):
        # CMake checks for the filenames during configure. The public analyzer
        # loads the actual data from the engine repository at runtime.
        (dest / "models" / f.name).touch()


def main():
    dest = Path(sys.argv[1]).resolve()
    export_control(dest)
    path = dest / "cpp/src/rollout.cpp"
    s = path.read_text()

    def before(anchor, new, count=1):
        nonlocal s
        if s.count(anchor) != count:
            raise RuntimeError(f"Expected {count} copies of {anchor!r}, found {s.count(anchor)}")
        s = s.replace(anchor, new + "\n" + anchor)

    before("namespace bgbot {", (HERE / "phase_timer.inc").read_text())
    before("    thread_local std::array<std::vector<Board>, ALL_ROLLS.size()> move_candidates;",
           "    PhaseTimer timer(Phase::other);")
    before("        const Move1Cache::Entry* move1_entry = nullptr;", "        timer.next(Phase::other);")
    for marker, phase in (("Phase 1: Cube check", "cube"), ("Phase 2: Generate moves", "movegen"),
                          ("Phase 3: VR mean", "vr_mean"), ("Phase 4: Pick best move", "select"),
                          ("Phase 4b: VR luck", "vr_luck"), ("Phase 5: Terminal check", "other")):
        before("        // " + marker, f"        timer.next(Phase::{phase});")
    before("    // `board` is from the next mover's perspective", "    timer.next(Phase::truncation);")
    before("    const Board& move0_chosen = move0_cache.chosen[first_roll_idx];",
           "    PhaseTimer timer(Phase::prefill1);")
    before("        if (shared) MultiPlyStrategy::set_shared_cache(shared);\n\n        thread_local std::vector<Board> candidates;",
           "        PhaseTimer timer(Phase::prefill0);")
    mark = "while ((r = next_roll.fetch_add(1, std::memory_order_relaxed)) < 21) {"
    if s.count(mark) != 2:
        raise RuntimeError("Unexpected prefill dispatch")
    s = s.replace(mark, mark + "\n                PhaseTimer first_roll(Phase::prefill0);")
    before("                move0_cache.state[r].store(2, std::memory_order_release);",
           "                first_roll.finish();", count=2)
    path.write_text(s)


if __name__ == "__main__":
    main()
