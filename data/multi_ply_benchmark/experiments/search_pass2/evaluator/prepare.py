# SPDX-License-Identifier: MPL-2.0
"""Apply the archived ablations to independent build-only source snapshots."""
from argparse import ArgumentParser
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryFile
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
WORK = ROOT / "build/det_eval2"
REV = "33c4aa1f6c28d12b846e66e745eef6c2692c5052"
SOURCES = {"route": ("encoding.cpp", "neural_net.cpp"), "mask": ("encoding.cpp",),
           "delta": ("encoding.cpp", "neural_net.cpp", "cube_eval.cpp"),
           "combined": ("encoding.cpp", "neural_net.cpp"), "all": ("encoding.cpp", "neural_net.cpp", "cube_eval.cpp")}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="+", choices=list(SOURCES))
    args = parser.parse_args()
    anchor = WORK / "anchor"
    anchor.mkdir(parents=True, exist_ok=True)
    with TemporaryFile() as data:
        subprocess.run(["git", "archive", REV, "cpp"], cwd=ROOT, check=True, stdout=data)
        data.seek(0)
        with tarfile.open(fileobj=data) as tar:
            tar.extractall(anchor, filter="data")
    for arm in args.arms:
        dest = WORK / arm
        dest.mkdir(exist_ok=True)
        for f in SOURCES[arm]:
            shutil.copy2(anchor / "cpp/src" / f, dest / f)
        shutil.copytree(anchor / "cpp/include", dest / "include", dirs_exist_ok=True)
        names = {"combined": ("route", "mask"), "all": ("route", "mask", "delta_after_combined")}.get(arm, (arm,))
        for name in names:
            subprocess.run(["git", "apply", "--no-index", "--directory", dest.relative_to(ROOT).as_posix(),
                            str(HERE / (name+".patch"))], cwd=ROOT, check=True)
