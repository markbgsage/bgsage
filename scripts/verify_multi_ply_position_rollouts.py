# SPDX-License-Identifier: MPL-2.0
"""Audit the completed position corpus. Verification does not run rollouts."""
import argparse
from collections import Counter
from dataclasses import asdict
import json

import multi_ply_position_rollouts as positions
import multi_ply_rollout_truth as pinned
from multi_ply_rollout_sources import load_archived


def indexed(rows, key):
    result = {key(row):row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate record")
    return result


def verify(raw=False):
    out = positions.OUT
    manifest = pinned.read(out / "manifest.json")
    for filename, expected in manifest["outputs"].items():
        if pinned.digest((out / filename).read_bytes()) != expected:
            raise ValueError(f"Published output differs from manifest: {filename}")
    case_key = lambda row:(row["api"],row["position_id"])
    corpus = indexed(positions.rows(out/"results.jsonl"), case_key)
    inputs = []
    for name in ("checker_play.json", "cube_action.json"):
        path = pinned.DATA / name
        if pinned.digest(path.read_bytes()) != manifest["fixed_sets"][name]:
            raise ValueError(f"Fixed inputs differ: {name}")
        inputs.append(indexed(pinned.read(path)["positions"],lambda p:p["id"]))
    checker, cubes = inputs
    expected_keys = {("checker_play", k) for k in checker} | {
        (api,k) for api in ("cube_action","post_move_analytics") for k in cubes}
    if set(corpus) != expected_keys or len(corpus) != 300:
        raise ValueError("Expected the 300 distinct selected API/position cases")
    original, source_cubes = load_archived()
    support = indexed(positions.rows(out/"archived_cube_support.jsonl"), lambda r:r["position_id"])
    if len(original) != 37 or support != source_cubes or len(support) != 10:
        raise ValueError("Historical source coverage mismatch")
    saved_cubes = indexed([r for r in positions.rows(pinned.OUT/"results.jsonl") if r["api"]!="checker_play"],case_key)
    for key,row in saved_cubes.items():
        published = corpus[key]
        if {k:published[k] for k in row} != row or published["reference_origin"] != "frozen_public_rollout":
            raise ValueError(f"Reused public call changed: {key}")
    if set(saved_cubes) != {key for key in expected_keys if key[0]!="checker_play"}:
        raise ValueError("Missing reused cube/post records")
    coverage = indexed(positions.rows(out/"position_coverage.jsonl"),case_key)
    if set(coverage) != expected_keys or not all(r["complete"] for r in coverage.values()):
        raise ValueError("Per-position coverage mismatch")
    sidecars = indexed(positions.rows(out/"checker_cubeful_se.jsonl"),lambda r:(r["position_id"],tuple(r["board"])))
    expected_ses = set()
    origin, levels = Counter(), Counter()
    local = {}
    if raw:
        analyzer, _, _, engine = pinned.make_analyzer(1)
        local = {(p["id"],tuple(b)):name for name,api,p,b in pinned.jobs() if api=="checker_play"}
    total_endpoints = 0
    for pid,p in checker.items():
        row = corpus[("checker_play",pid)]
        a, kwargs = row["analytics"], p["inputs"]
        if row["source"] != p["source"] or row["inputs"] != kwargs:
            raise ValueError(f"Changed position or provenance: {pid}")
        boards = [tuple(m["board"]) for m in a["moves"]]
        if len(boards) != len(set(boards)) or a["board"]!=kwargs["board"] or (a["die1"],a["die2"])!=(kwargs["die1"],kwargs["die2"]):
            raise ValueError(f"Move identity or dice mismatch: {pid}")
        if raw:
            legal = {tuple(b) for b in engine.possible_moves(a["board"],a["die1"],a["die2"])}
            if legal != set(boards):
                raise ValueError(f"Legal endpoint mismatch: {pid}")
        total_endpoints += len(boards)
        source = row["reference_origin"]
        origin[source] += 1
        levels.update((source,m["eval_level"]) for m in a["moves"])
        if pid in original:
            if row != original[pid]:
                raise ValueError(f"Historical source changed: {pid}")
            continue
        if source != "frozen_public_rollout" or row["seed"] != pinned.seed_for(p):
            raise ValueError(f"Frozen RNG seed or origin mismatch: {pid}")
        if a["moves"][0]["eval_level"]!="Rollout":
            raise ValueError(f"Public checker leader was not rolled out: {pid}")
        for move in a["moves"]:
            if move["equity_diff"] != move["equity"]-a["moves"][0]["equity"]:
                raise ValueError(f"Incorrect relative equity: {pid}")
            if move["eval_level"] != "Rollout":
                continue
            key = (pid,tuple(move["board"]))
            expected_ses.add(key)
            if not raw:
                continue
            filename = local[key] + ".json"
            path = next((d/filename for d in (pinned.WORK/"units",positions.MISSES) if (d/filename).is_file()),None)
            if path is None:
                raise ValueError(f"Missing native record: {filename}")
            unit = pinned.read(path)
            result = unit["native_checker"]
            probs = result["probs"]
            board = move["board"]
            if analyzer._bearoff_db.is_bearoff(board):
                probs = analyzer._bearoff_db.lookup_probs(board,post_move=True)
            if (unit["seed"],unit["position_id"],unit["analytics"]["board"],result["cubeful_equity"],
                    result["equity"],result["std_error"],result["prob_std_errors"],result["cubeful_se"],
                    asdict(engine.Probabilities.from_list(probs))) != (
                        row["seed"],pid,board,move["equity"],move["cubeless_equity"],move["std_error"],
                        move["prob_std_errors"],sidecars[key]["cubeful_se"],move["probs"]):
                raise ValueError(f"Native means or standard errors changed: {filename}")
    if set(sidecars)!=expected_ses or total_endpoints!=1709 or origin!={"archival_rollout":37,"frozen_public_rollout":63}:
        raise ValueError("Checker move or standard error coverage mismatch")
    return dict(cases=len(corpus),checker_positions_by_origin=dict(origin),checker_endpoints=total_endpoints,
                checker_move_levels={f"{a}/{b}":n for (a,b),n in levels.items()},
                available_native_checker_cubeful_ses=len(sidecars),reused_cube_post_calls=len(saved_cubes),
                historical_cube_support=len(support),raw_native_checked=raw)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-raw",action="store_true",help="Also check frozen legal moves, bearoff table and raw native SEs")
    args=parser.parse_args()
    print(json.dumps(verify(args.local_raw),indent=2))
