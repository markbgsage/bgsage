# SPDX-License-Identifier: MPL-2.0
"""Save minimal replay patches for the standalone dice-fan and trial-tree screens."""
import argparse
from difflib import unified_diff
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def files(root):
    return {p.relative_to(root): p for sub in ("cpp", "python/bgsage")
            for p in (root / sub).rglob("*") if p.suffix in {".cpp",".h",".inc",".py",".cu",".txt"} and p.is_file()}


def patch(a, b):
    left, right = files(a), files(b)
    result = []
    for path in sorted(left.keys() | right.keys()):
        before = left[path].read_bytes().replace(b"\r\n", b"\n") if path in left else b""
        after = right[path].read_bytes().replace(b"\r\n", b"\n") if path in right else b""
        if before != after:
            result.extend(unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
                fromfile="a/" + path.as_posix() if path in left else "/dev/null",
                tofile="b/" + path.as_posix() if path in right else "/dev/null"))
    return "".join(result).encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--builds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = {"checkpoint": "1ecca3902fc7d04e35c43b9fcd3ec45920d0fcb5", "variants": {}}
    with ZipFile(args.output, "w", ZIP_DEFLATED) as z:
        for variant in ("first","per-roll","dedup","trial-tree"):
            parent = "checkpoint" if variant == "first" else "first"
            content = patch(args.baseline if parent == "checkpoint" else args.builds / parent, args.builds / variant)
            name = parent + "-to-" + variant + ".patch"
            z.writestr(name, content)
            manifest["variants"][variant] = {"base": parent,"patch": name,"sha256": sha256(content).hexdigest()}
        z.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
