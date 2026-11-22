"""Check the linked extensions' native training encoders on real and directed boards."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/"scripts"))
from benchmark_multi_ply import load_engine


def directed():
    boards={}
    def board(name,mover,opponent):
        b=[0]*26
        for i,n in mover.items(): b[i]=n
        for i,n in opponent.items():
            assert not b[i]
            b[i]=n if i==0 else -n
        assert sum(mover.values())<=15 and sum(opponent.values())<=15
        boards[name]=b
        boards[name+"_flipped"]=[b[25],*[-i for i in b[24:0:-1]],b[0]]
    board("two_on_bar",{25:2,19:8,4:5},{0:1,24:2,23:2,22:1,18:1,1:8})
    board("bar_trapped_by_six_home_points",{25:3,8:12},{19:2,20:2,21:2,22:2,23:2,24:2,5:3})
    board("all_on_bar",{25:15},{0:15})
    board("high_stack",{24:15},{1:15})
    board("backgame_anchors",{24:2,23:2,19:3,17:1,8:1,4:6},{1:2,2:2,5:2,7:1,9:1,16:7})
    board("bearing_off",{6:2,4:1,3:1},{19:3,20:1,21:2,24:2})
    board("one_borne_off_side",{1:1},{19:15})
    board("terminal",{}, {0:2,6:1,11:2,19:3,20:2,22:2,24:3})
    board("long_doubles_sparse_block",{25:1,24:2,18:1,16:1,10:3},
          {1:1,2:1,3:1,5:2,8:1,11:1,15:2,19:2,23:2})
    board("shared_single_source",{12:1,6:1},{7:1,8:1,5:1})
    board("shared_stacked_source",{12:2,6:1},{7:1,8:1,5:1})
    return boards


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("extension",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    load_engine(args.extension)
    import bgbot_cpp as cpp
    samples=np.fromfile(ROOT/"build/cpu_feature_masks/all.bin",dtype="<i4").reshape(-1,26)
    extra=directed()
    boards=np.concatenate((samples,np.asarray(list(extra.values()),dtype=np.int32)))
    out={"rows":len(boards),"explicit_cases":list(extra),"fields":{}}
    for size in (196,244):
        digest=hashlib.sha256()
        for start in range(0,len(boards),512):
            features=cpp.encode_boards_batch(boards[start:start+512],size)
            assert features.shape==(len(boards[start:start+512]),size) and features.dtype==np.float32
            digest.update(features.tobytes())
        out["fields"][str(size)]=digest.hexdigest()
    args.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out["fields"]),flush=True)


if __name__=="__main__": main()
