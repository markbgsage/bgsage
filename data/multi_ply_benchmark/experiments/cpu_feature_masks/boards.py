"""Generate the common raw CPU-encoder fixture from the saved benchmark roots."""
import argparse
import json
from pathlib import Path
import random
import struct
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/"scripts"))
from benchmark_multi_ply import load_engine


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--extension", type=Path,
                        help="Explicit baseline CPU extension (default: local Windows qualification build)")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    extension = args.extension or next((ROOT/"build/multi_ply_release").glob("bgbot_cpp*.pyd"))
    load_engine(extension)
    import bgbot_cpp as cpp
    real=[]
    for d in json.loads((ROOT/"data/multi_ply_benchmark/checker_play.json").read_bytes())["positions"]:
        p=d["inputs"]
        for j,b in enumerate(cpp.possible_moves(p["board"],p["die1"],p["die2"])):
            real.append(b)
            real.extend(cpp.possible_moves(cpp.flip_board(b),j%6+1,(j//6)%6+1))
    for d in json.loads((ROOT/"data/multi_ply_benchmark/cube_action.json").read_bytes())["positions"]:
        real.extend((d["inputs"]["board"],d["post_move_inputs"]["board"]))
    cases=list(real)
    for hi in range(1,25):
        for lo in range(1,hi):
            for source in range(hi+1,26):
                b=[0]*26
                b[lo]=b[hi]=-1
                b[source]=1
                for before in (b,cpp.flip_board(b)):
                    cases.append(before)
                b2=b.copy()
                if lo+1<hi: b2[lo+1]=-2
                b2[source]=2
                cases.extend((b2,cpp.flip_board(b2)))
    rng=random.Random(4201973)
    for _ in range(6000):
        b=[0]*26
        pts=list(range(1,25))
        rng.shuffle(pts)
        split=rng.randrange(1,24)
        pa=[25,*pts[:split]]
        po=[0,*pts[split:]]
        for _ in range(rng.randrange(1,16)): b[rng.choice(pa)]+=1
        for _ in range(rng.randrange(1,16)):
            i=rng.choice(po)
            b[i]+=1 if i==0 else -1
        cases.extend((b,cpp.flip_board(b)))
    for name,items in (("real",real),("all",cases)):
        with (args.output/f"{name}.bin").open("wb") as w:
            for b in items: w.write(struct.pack("<26i",*b))
        print(name,len(items))


if __name__=="__main__": main()
