"""Re-score the saved screening experiment as more frozen rollout units arrive."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / "data/multi_ply_benchmark"
PROBS = ("win", "gammon_win", "backgammon_win", "gammon_loss", "backgammon_loss")


def load(name):
    with gzip.open(HERE / (name + ".jsonl.gz"), "rt") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def fields(m):
    probs = m["probs"]
    if isinstance(probs, list):
        probs = dict(zip(PROBS, probs))
    return {**probs, "equity": m["equity"], "cubeless_equity": m["cubeless_equity"]}


def compare(exact, screened, truth, counters):
    e, s, t = map(fields, (exact, screened, truth))
    for field, ref in t.items():
        delta = abs(s[field]-ref) - abs(e[field]-ref)
        tol = .0025 if field in PROBS else .005
        c = counters.setdefault(field, Counter())
        c["compared"] += 1
        c["error_change_sum"] += delta
        c["materially_worse"] += delta > tol
        c["materially_better"] += delta < -tol


def main():
    exact, screened = load("exact"), load("shortlist")
    assert set(exact) == set(screened) and len(exact) == 100
    pos = {p["id"]: p for p in json.loads((DATA/"checker_play.json").read_bytes())["positions"]}
    source_rows = {}
    for mode, path in [("money", "money_benchmark"), ("match", "match_benchmark/5pt")]:
        with gzip.open(ROOT/"data"/path/"benchmark.json.gz", "rt") as f:
            for d in json.load(f)["decisions"]:
                if d["kind"]=="checker" and d["tier"]=="rollout":
                    source_rows[(mode, d["key"])] = {tuple(m["board"]): m for m in d["moves"] if m["eval_level"]=="Rollout"}
    units = {}
    position_targets = {}
    position_file = DATA / "rollout_positions/results.jsonl"
    if position_file.exists():
        for line in position_file.read_bytes().splitlines():
            row = json.loads(line)
            if row["api"] == "checker_play":
                for move in row["analytics"]["moves"]:
                    if move["eval_level"] == "Rollout":
                        position_targets[row["position_id"], tuple(move["board"])] = move
    manifest = DATA/"rollout_3p_vr1296/manifest.json"
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for line in (manifest.parent / "results.jsonl").read_bytes().splitlines():
        row = json.loads(line)
        if row["api"] == "checker_play":
            for move in row["analytics"]["moves"]:
                units[row["position_id"], tuple(move["board"])] = move
    for path in (ROOT/"logs/multi_ply_rollout_truth/units").glob("move-*.json"):
        row = json.loads(path.read_bytes())
        assert row["manifest_sha256"] == manifest_sha
        units.setdefault((row["position_id"], tuple(row["analytics"]["board"])), row["analytics"])
    stats = Counter()
    refs = {}
    examples = []
    for pid, row in screened.items():
        e = exact[pid]["analytics"]["moves"]
        s = row["analytics"]["moves"]
        exact_by = {tuple(m["board"]):m for m in e}
        assert set(exact_by)=={tuple(m["board"]) for m in s}
        stats["moves"] += len(s)
        stats["best_changed"] += e[0]["board"]!=s[0]["board"]
        top2 = {tuple(m["board"]) for m in e[:2]}
        stats["top_two_set_changed"] += top2!={tuple(m["board"]) for m in s[:2]}
        stats["promotions"] += row["promotions"]
        source = source_rows.get((row["mode"],pos[pid]["source"]["key"]),{})
        for m in s:
            board=tuple(m["board"])
            b=exact_by[board]
            shallow = m["eval_level"]=="2-ply"
            stats["shallow_moves"] += shallow
            stats["exact_best_not_deep"] += shallow and tuple(e[0]["board"])==board
            for key, value in fields(m).items():
                stats["changed_by_threshold_"+key] += abs(value-fields(b)[key]) > (.0025 if key in PROBS else .005)
            for name,r in (("position_benchmark", position_targets.get((pid,board))),
                           ("fixed_1296",units.get((pid,board))),
                           ("historical_source_rollout_finalists",source.get(board))):
                if r:
                    compare(b,m,r,refs.setdefault(name,{}).setdefault("shallow" if shallow else "deep",{}))
                    if name == "fixed_1296" and shallow:
                        change = abs(m["equity"] - r["equity"]) - abs(b["equity"] - r["equity"])
                        if change > .005:
                            examples.append(dict(position_id=pid, board=board, full_3ply=b["equity"],
                                                 screened=m["equity"], rollout=r["equity"],
                                                 increase_in_absolute_error=change))
    ts,te=[sum(r["seconds"] for r in a.values()) for a in (screened,exact)]
    result={"positions":len(screened),"exact_seconds":te,"screened_seconds":ts,"speedup":te/ts,
            "median_paired_seconds_ratio":median(screened[k]["seconds"]/exact[k]["seconds"] for k in screened),
            "counts":stats,"rollout_manifest_sha256":manifest_sha,"error_vs_rollout":refs,
            "largest_fixed_equity_regressions":sorted(examples, key=lambda e: -e["increase_in_absolute_error"])[:5]}
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
