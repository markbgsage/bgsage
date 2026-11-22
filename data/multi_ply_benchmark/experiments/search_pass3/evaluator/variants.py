# SPDX-License-Identifier: MPL-2.0
"""Recreate isolated encoder sources from the qualified control."""
from pathlib import Path
import shutil
import sys
from build_linux import HERE, OLD, WORK


def prepare(name):
    source = WORK / ("source_" + name)
    for path in ("cpp", "python"):
        shutil.copytree(OLD / "guarded" / path, source / path, dirs_exist_ok=True)
    shutil.copyfile(OLD / "guarded/CMakeLists.txt", source / "CMakeLists.txt")
    file = source / "cpp/src/encoding.cpp"
    code = file.read_text()
    if "reach" in name:
        old = "const unsigned blots = masks.enemy_blots;\n    if (blots == 0) return counts;"
        assert code.count(old) == 1
        code = code.replace(old, """unsigned blots = masks.enemy_blots;
    if (!OnBar && blots) {
        if (!masks.occupied) return counts;
        // Blots at or beyond the rearmost checker cannot be hit.
        blots &= (1u << (highest_set_point(masks.occupied) - 1)) - 1;
    }
    if (blots == 0) return counts;""")
    if "bar" in name:
        code = code.replace("struct ShotFeatures {", (HERE / "bar.inc").read_text() + "\nstruct ShotFeatures {")
        code = code.replace("counts.pip_loss = compute_piploss(board);", "counts.pip_loss = entry_pip_loss(board, masks);")
    if "simd" in name:
        marker = "    for (int high = 2; high <= 6; ++high) {"
        start = code.index(marker, code.index("static ShotFeatures shot_features"))
        end = code.index("    if constexpr (OnBar) counts.pip_loss", start)
        code = (code[:start] + "#if defined(__AVX2__)\n    if constexpr (!OnBar) {\n" +
                (HERE / "simd.inc").read_text() + "\n    } else\n#endif\n    {\n" +
                code[start:end] + "    }\n" + code[end:])
    if "lanes" in name:
        code = code.replace("// The bar changes the GNUbg pip-loss calculation.", (HERE / "lanes.inc").read_text().rstrip() +
                            "\n\n// The bar changes the GNUbg pip-loss calculation.")
        code = code.replace("    if (blots == 0) return counts;\n",
            "    if (blots == 0) return counts;\n#if defined(__AVX2__)\n"
            "    if constexpr (!OnBar) return off_bar_shots(masks, blots);\n#endif\n")
    if "bar" in name:
        code = code.replace("// highest hittable target per roll; with checkers on the bar we call the\n"
                            "// original entry-aware calculation. Both cases share the same shot features.",
                            "// highest hittable target per roll. Checkers on the bar restrict legal entries\n"
                            "// and which direct hits by on-board checkers contribute to pip loss.")
    file.write_text(code)
    return source


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(prepare(name))
