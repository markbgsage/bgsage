"""Compare every feature byte and measure CPU encoder ablations on the same boards."""
import argparse
import hashlib
import json
from pathlib import Path
from random import Random
import statistics
import subprocess
from tempfile import TemporaryDirectory
from variants import BASE_SHA256

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
WORK=ROOT/"build/cpu_feature_masks"


def sha(path):
    with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms",nargs="+",default=["baseline","shots","pips","both"])
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    shared,real=WORK/"all.bin",WORK/"real.bin"
    out={"all_boards":shared.stat().st_size//104,"all_sha256":sha(shared),
         "real_boards":real.stat().st_size//104,"real_sha256":sha(real),
         "source_encoder_sha256":BASE_SHA256,
         "single_logical_cpu":14,"feature_outputs":{},"timings_ns_per_board":[]}
    with TemporaryDirectory(dir=WORK) as name:
        ref=None
        for arm in args.arms:
            path=Path(name)/f"{arm}.features"
            subprocess.run([str(WORK/arm/"probe.exe"),str(shared),"dump",str(path)],check=True)
            digest=sha(path)
            out["feature_outputs"][arm]={"sha256":digest,"bytes":path.stat().st_size}
            if ref is None:
                ref=path
                ref_hash=digest
            elif digest!=ref_hash:
                with path.open("rb") as got,ref.open("rb") as expected:
                    for i in range(out["all_boards"]):
                        a=got.read(1760)
                        b=expected.read(1760)
                        if a!=b:
                            slots=[j for j in range(440) if a[4*j:4*j+4]!=b[4*j:4*j+4]]
                            out["feature_outputs"][arm].update(first_different_board=i,slots=slots)
                            break
    if any(item["sha256"] != ref_hash for item in out["feature_outputs"].values()):
        args.output.write_text(json.dumps(out, indent=2) + "\n")
        raise ValueError("Feature parity failed. The report identifies the first differing board and slots.")
    rng=Random(20261121)
    for rep in range(args.repeats):
        order=list(args.arms)
        rng.shuffle(order)
        for arm in order:
            text=subprocess.check_output([str(WORK/arm/"probe.exe"),str(real),"profile"],text=True)
            ns={line.split()[0]:float(line.split()[1]) for line in text.splitlines()}
            out["timings_ns_per_board"].append({"pass":rep+1,"arm":arm,**ns})
        print("pass",rep+1,flush=True)
    out["median_extended_ns"]={arm:statistics.median(x["extended"] for x in out["timings_ns_per_board"]
                                                    if x["arm"]==arm) for arm in args.arms}
    args.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out["feature_outputs"]),json.dumps(out["median_extended_ns"]),flush=True)


if __name__=="__main__": main()
