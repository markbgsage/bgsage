# SPDX-License-Identifier: MPL-2.0
"""Prepare the experimental VR table in a private export of the control."""
from pathlib import Path
import sys

from instrument import export_control, HERE


def change(s, old, new, count=1):
    if s.count(old) != count:
        raise ValueError(f"Unexpected source for {old!r}")
    return s.replace(old, new)


def main():
    dest = Path(sys.argv[1]).resolve()
    export_control(dest)
    header = dest / "cpp/include/bgbot/rollout.h"
    h = header.read_text()
    mark = "    struct Move1Cache {"
    h = change(h, mark, (HERE / "vr_memo.inc").read_text() + "\n" + mark +
               "\n        std::unique_ptr<VRTable> vr;")
    header.write_text(h)
    cpp = dest / "cpp/src/rollout.cpp"
    s = cpp.read_text()
    s = change(s, "namespace bgbot {", """
#ifdef _WIN32
#define VR_EXPORT __declspec(dllexport)
#else
#define VR_EXPORT __attribute__((visibility("default")))
#endif
static std::atomic<long long> vr_attempts{}, vr_hits{};
extern "C" VR_EXPORT void take_vr_hits(long long* out) {
    out[0] = vr_attempts.exchange(0); out[1] = vr_hits.exchange(0);
}
namespace bgbot {""")
    s = change(s, "    Move1Cache move1_cache;", """    Move1Cache move1_cache;
    if (config_.truncation_depth > 2 && config_.n_trials >= 72 && config_.enable_vr)
        move1_cache.vr = std::make_unique<VRTable>();""", 3)
    begin = """                for (size_t i = 0; i < ALL_ROLLS.size(); ++i) {
                    int idx = -1;
                    roll_best_probs[i]"""
    pre = """                auto* vr_table = move1_cache ? move1_cache->vr.get() : nullptr;
                const int slot = vr_table ? VRTable::slot_for(board) : 0;
                const auto* saved = vr_table ? vr_table->lookup(slot, board, vr_sel_ptr, vr_n_sel, cube_x) : nullptr;
                if (vr_table) vr_attempts.fetch_add(1, std::memory_order_relaxed);
                if (saved) {
                    vr_hits.fetch_add(1, std::memory_order_relaxed);
                    roll_best_probs = saved->probs;
                    best_candidate_idx = saved->picks;
                } else {
"""
    s = change(s, begin, pre + begin)
    end = """                // Cubeless VR mean (always computed)"""
    s = change(s, end, """                    if (vr_table) vr_table->put(slot, board, vr_sel_ptr, vr_n_sel, cube_x,
                                                roll_best_probs, best_candidate_idx);
                }
""" + end)
    cpp.write_text(s)


if __name__ == "__main__":
    main()
