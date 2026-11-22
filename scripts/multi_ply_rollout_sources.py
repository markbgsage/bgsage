# SPDX-License-Identifier: MPL-2.0
"""Recover historical rollout means from the original money and 5-point archives."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from build_multi_ply_benchmark import call_inputs, post_move_inputs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/multi_ply_benchmark"
OUT = DATA / "historical_rollouts"
PROBABILITIES = ("win", "gammon_win", "backgammon_win", "gammon_loss", "backgammon_loss")
COMMITS = {"money": ("1796482", "2026-06-24"), "match": ("4e8cd04", "2026-06-27")}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source_probabilities(values):
    if len(values) != len(PROBABILITIES):
        raise ValueError("Expected five archived probabilities")
    return dict(zip(PROBABILITIES, values))


def checker_analytics(source):
    moves = source["moves"]
    if any(a["equity"] < b["equity"] for a, b in zip(moves, moves[1:])):
        raise ValueError(f"Archived moves are not best-first: {source['key']}")
    best = moves[0]["equity"]
    detailed = []
    for move in moves:
        detailed.append({
            "board": move["board"], "equity": move["equity"],
            "cubeless_equity": move["cubeless_equity"], "probs": source_probabilities(move["probs"]),
            "equity_diff": move["equity"] - best, "eval_level": move["eval_level"],
            "player_game_plan": None, "opponent_game_plan": None,
            "std_error": move.get("std_error"), "prob_std_errors": None})
    return {"board": source["board"], "die1": source["dice"][0], "die2": source["dice"][1],
            "moves": detailed, "eval_level": "Rollout"}


def cube_analytics(source):
    doubles, takes = source["should_double_ref"], source["should_take_ref"]
    if not doubles:
        action, branch = "No Double", "equity_nd"
    elif not takes:
        action, branch = "Double/Pass", "equity_dp"
    else:
        action = "Double/Beaver" if source["is_beaver"] else "Double/Take"
        branch = "equity_dt"
    if source["optimal_action"] != action:
        raise ValueError(f"Archived action and decision flags disagree: {source['key']}")
    # The raw public cubeless equity and all three uncertainty estimates were
    # omitted when the final historical cube dataset was assembled.
    return {
        "probs": source_probabilities(source["probs"]), "cubeless_equity": None,
        "equity_nd": source["equity_nd"], "equity_dt": source["equity_dt"],
        "equity_dp": source["equity_dp"], "should_double": doubles, "should_take": takes,
        "optimal_equity": source[branch], "optimal_action": action, "eval_level": source["eval_level"],
        "is_beaver": source["is_beaver"], "cubeless_se": None, "equity_nd_se": None,
        "equity_dt_se": None, "details": None}, branch


def convert_position(api, position, source, metadata):
    mode = position["mode"]
    inputs = call_inputs(source, mode)
    if inputs != position["inputs"] or position["source"]["key"] != source["key"]:
        raise ValueError(f"Archived inputs or key differ from the fixed position: {position['id']}")
    if api == "cube_action" and post_move_inputs(inputs) != position["post_move_inputs"]:
        raise ValueError(f"Inconsistent opposite-player inputs: {position['id']}")
    if source["tier"] != position["source"]["tier"]:
        raise ValueError(f"Reference tier differs for {position['id']}")
    if source["tier"] != "rollout":
        return None
    commit, date = COMMITS[mode]
    provenance = {"path":metadata["path"], "sha256":metadata["sha256"],
                  "source_commit":commit, "first_committed":date,
                  "decision_index":position["source"]["decision_index"],
                  "key":source["key"], "tier":source["tier"],
                  "scenario_seed":source["seed"], "rollout_rng_seed_recorded":False,
                  "paths_per_batch":metadata["meta"]["rollout_n_trials"], "actual_batches":None,
                  "actual_model_hash":None, "actual_extension_hash":None}
    if api == "checker_play":
        analytics = checker_analytics(source)
        provenance.update(
            full_rollout_move_ranks=[i for i, m in enumerate(source["moves"], 1)
                                     if m["eval_level"] == "Rollout"],
            derived_fields=["moves[].equity_diff"], unavailable_fields=["moves[].prob_std_errors"])
    else:
        analytics, branch = cube_analytics(source)
        win, gw, bw, gl, bl = source["probs"]
        provenance.update(
            optimal_equity_source_field=branch,
            cubeless_equity_from_archived_probabilities=2*win - 1 + gw - gl + bw - bl,
            derived_fields=["optimal_equity"],
            unavailable_fields=["cubeless_equity","cubeless_se","equity_nd_se","equity_dt_se"])
    return {"api":api, "position_id":position["id"], "mode":mode,
            "source_game_plan":position["game_plan"], "source":position["source"],
            "inputs":inputs, "seed":None, "reference_origin":"archival_rollout",
            "provenance":provenance, "analytics":analytics}


def inspect_source(mode, fixed, native):
    metadata = fixed["checker_play"]["sources"][mode]
    if metadata != fixed["cube_action"]["sources"][mode]:
        raise ValueError(f"Conflicting source metadata for {mode}")
    source_path = ROOT / metadata["path"]
    if sha(source_path.read_bytes()) != metadata["sha256"]:
        raise ValueError(f"Source checksum mismatch for {source_path}")
    with gzip.open(source_path, "rt", encoding="utf-8") as stream:
        original = json.load(stream)
    if original["meta"] != metadata["meta"]:
        raise ValueError(f"Unexpected source metadata for {mode}")
    pool = defaultdict(Counter)
    for decision in original["decisions"]:
        pool[(decision["kind"], decision["tier"])][decision["game_plan"]] += 1
    report = {"path":metadata["path"], "sha256":metadata["sha256"], "meta":original["meta"],
              "whole_source_position_counts":[{"kind":kind, "tier":tier, "plans":counts}
                  for (kind,tier),counts in sorted(pool.items())], "fixed":{}}
    rows = []
    for api, corpus in fixed.items():
        tiers, move_counts, root_coverage, keys = Counter(), defaultdict(Counter), Counter(), []
        for position in corpus["positions"]:
            if position["mode"] != mode:
                continue
            ref = position["source"]
            decision = original["decisions"][ref["decision_index"]]
            row = convert_position(api, position, decision, metadata)
            tiers[decision["tier"]] += 1
            record = {"id":position["id"], "key":decision["key"],
                      "decision_index":ref["decision_index"], "tier":decision["tier"]}
            if api == "checker_play":
                moves = decision["moves"]
                boards = {tuple(m["board"]) for m in moves}
                if len(boards) != len(moves):
                    raise ValueError(f"Repeated source endpoint for {record['id']}")
                if native:
                    legal = {tuple(b) for b in native.possible_moves(decision["board"], *decision["dice"])}
                    if legal != boards:
                        raise ValueError(f"Legal endpoints differ for {record['id']}")
                levels = Counter(m["eval_level"] for m in moves)
                move_counts[decision["tier"]].update(levels)
                ranks = row["provenance"]["full_rollout_move_ranks"] if row else []
                record.update(legal_moves=len(moves), eval_level_counts=levels, full_rollout_ranks=ranks)
                root_coverage["all" if len(ranks)==len(moves) else "partial" if ranks else "none"] += 1
                root_coverage["best_move_rolled"] += 1 in ranks
            else:
                record.update(optimal_action=decision["optimal_action"],
                              has_double_decision=decision["has_double"],
                              has_take_decision=decision["has_take"])
            keys.append(record)
            if row:
                rows.append(row)
        report["fixed"][api] = {"position_tiers":tiers, "positions":keys}
        if api == "checker_play":
            report["fixed"][api].update(move_eval_level_by_root_tier=move_counts,
                                        full_rollout_coverage=root_coverage)
    return rows, report


def generate(native=None):
    fixed = {api: json.loads((DATA / (api + ".json")).read_bytes())
             for api in ("checker_play", "cube_action")}
    report = {"source_modes":{}, "source_kind":"native Sage adaptive reference",
              "historical_generator":{
                  "source_revisions":COMMITS, "default_model_at_both_revisions":"stage9",
                  "full_rollout":{"truncation_depth":0, "checker_ply":3, "cube_ply":3,
                                  "variance_reduction":True, "paths_per_batch":1296,
                                  "base_seed":42, "target_equity_se":0.00255, "max_batches":16},
                  "unrecorded_per_position":["weights_hash","binary_hash","trial_seed","batch_count"]},
              "checker_recorded":["post-move board","cubeful equity","cubeless equity",
                                  "five cubeless probabilities","per-move eval_level","cubeless equity std_error"],
              "cube_recorded":["five cubeless probabilities","equity_nd","equity_dt","equity_dp",
                               "should_double_ref","should_take_ref","optimal_action","is_beaver","eval_level"],
              "post_move_support":{"board":"flip source board","cube_owner":"swap player and opponent",
                                   "away_scores":"swap","cube_value_crawford_jacoby":"preserve",
                                   "beaver_and_cube_limit":"not accepted by the post-move API",
                                   "probabilities":["1-win","gammon_loss","backgammon_loss","gammon_win","backgammon_win"],
                                   "cubeful_equity_proxy":"negative of the source optimal branch equity",
                                   "separately_recorded_post_move_call":False}}
    rows = []
    for mode in ("money","match"):
        results, report["source_modes"][mode] = inspect_source(mode, fixed, native)
        rows.extend(results)
    rows.sort(key=lambda row: (row["api"], row["position_id"]))
    return rows, report


def load_archived():
    """Return two stable position-ID maps with canonical public-shaped historical rows."""
    packed = (OUT / "rows.jsonl.gz").read_bytes()
    audit = json.loads((OUT / "audit.json").read_bytes())
    if sha(packed) != audit["rows_archive_sha256"]:
        raise ValueError("Historical row archive checksum mismatch")
    payload = gzip.decompress(packed)
    if sha(payload) != audit["rows_jsonl_sha256"]:
        raise ValueError("Historical row content checksum mismatch")
    by_api = {"checker_play":{}, "cube_action":{}}
    for line in payload.splitlines():
        row = json.loads(line)
        output = by_api[row["api"]]
        pid = row["position_id"]
        if pid in output or row["reference_origin"] != "archival_rollout":
            raise ValueError(f"Unexpected archival record: {pid}")
        output[pid] = row
    return by_api["checker_play"], by_api["cube_action"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extension", type=Path, help="Also verify every stored checker endpoint against legal moves")
    p.add_argument("--verify", action="store_true", help="Check both existing files against fresh extraction")
    args = p.parse_args()
    native = None
    if args.extension:
        from benchmark_multi_ply import load_engine
        load_engine(args.extension)
        import bgbot_cpp as native
    rows, report = generate(native)
    payload = b"".join((json.dumps(row, separators=(",",":"), allow_nan=False)+"\n").encode()
                       for row in rows)
    if args.verify:
        packed = (OUT / "rows.jsonl.gz").read_bytes()
        if gzip.decompress(packed) != payload:
            raise ValueError("Packed rows differ from the original rollout sources")
    else:
        packed = gzip.compress(payload, compresslevel=9, mtime=0)
    report["rows_archive_sha256"] = sha(packed)
    report["rows_jsonl_sha256"] = sha(payload)
    audit = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    for name, contents in (("rows.jsonl.gz", packed), ("audit.json", audit)):
        path = OUT / name
        if args.verify:
            if not path.is_file() or path.read_bytes() != contents:
                raise ValueError(f"Source-derived artifact differs: {path}")
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
    checkers, cubes = load_archived()
    print(f"Historical full-rollout rows: {len(checkers)} checker, {len(cubes)} cube")
    if native:
        print("All 100 archived checker lists exactly match native legal endpoints")


if __name__ == "__main__":
    main()
