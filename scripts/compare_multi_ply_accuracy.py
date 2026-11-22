# SPDX-License-Identifier: MPL-2.0
"""Compare two fixed 3-ply runs to genuine rollout values on the timing corpus.

Match checker candidates by their 26-point endpoint, independent of rank.
Errors are in absolute probability (0..1) or normalized equity units.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean
import time

from benchmark_multi_ply import DATA, file_identity, inside_repo, write_json


def records(path):
    return (json.loads(line) for line in path.read_bytes().splitlines())


def projected(api, analytics):
    if api == "checker_play":
        for move in analytics["moves"]:
            yield tuple(move["board"]), move
    else:
        yield None, analytics


def metrics(item):
    for field, value in item["probs"].items():
        yield "p." + field, value
    for field in ("cubeless_equity", "equity", "equity_diff", "equity_nd",
                  "equity_dt", "equity_dp", "optimal_equity", "cubeful_equity"):
        if field in item:
            yield field, item[field]


def predictions(path):
    rows = {}
    for row in records(path):
        if row["level"] != "3ply" or row["repeat"] != 1:
            continue
        key = row["api"], row["position_id"]
        if key in rows:
            raise ValueError(f"Duplicate prediction {key}")
        rows[key] = row
    if len(rows) != 300:
        raise ValueError(f"Expected 300 3-ply calls in {path}, found {len(rows)}")
    return rows


def summary(points, threshold):
    changes = [b - a for a, b in points]
    return {"n": len(points), "baseline_mae": mean(a for a, b in points),
            "candidate_mae": mean(b for a, b in points),
            "mean_change_in_abs_error": mean(changes), "worst_change_in_abs_error": max(changes),
            "materially_farther": sum(d > threshold for d in changes),
            "materially_closer": sum(d < -threshold for d in changes)}


def score(args, base, test):
    truth_path = inside_repo(args.truth)
    truth = {}
    completed = {}
    origins = {}
    reference_moves = Counter()
    for row in records(truth_path):
        key = row["api"], row["position_id"]
        if key in completed or key not in base or row["source"] != base[key]["source"]:
            raise ValueError(f"Invalid truth row {key}")
        rollout_inputs = dict(row["inputs"])
        if row["api"] == "cube_action":
            rollout_inputs.setdefault("incl_2ply_details", False)
        if rollout_inputs != base[key]["inputs"]:
            raise ValueError(f"Prediction and rollout input contexts differ: {key}")
        completed[key] = row["analytics"]
        origins[key] = row.get("reference_origin", "frozen_full_candidate_rollout")
        for board, item in projected(key[0], row["analytics"]):
            if key[0] == "checker_play":
                reference_moves[item["eval_level"]] += 1
                if item["eval_level"] != "Rollout":
                    continue
            if (key, board) in truth:
                raise ValueError(f"Duplicate reference endpoint in {key}")
            truth[key, board] = dict(metrics(item))
    if args.require_complete and completed.keys() != base.keys():
        raise ValueError(f"Only {len(completed)}/300 decisions have complete rollout references")
    if args.units:
        from hashlib import sha256
        fingerprint = sha256((truth_path.parent / "manifest.json").read_bytes()).hexdigest()
        for path in inside_repo(args.units).glob("move-*.json"):
            row = json.loads(path.read_bytes())
            if row["manifest_sha256"] != fingerprint:
                raise ValueError(f"Unmatched reference run: {path}")
            key = "checker_play", row["position_id"]
            if key in completed:
                continue
            move = row["analytics"]
            fields = dict(metrics(move))
            del fields["equity_diff"]  # depends on the actual best of all legal moves
            truth[key, tuple(move["board"])] = fields
    groups, coverage, regressions = defaultdict(list), Counter(), []
    origin_errors = defaultdict(list)
    complete_case_errors = defaultdict(lambda: defaultdict(list))
    shifts = defaultdict(list)
    exact, decisions_changed = Counter(), Counter()
    decisions = defaultdict(list)
    matched = set()
    for key, row in base.items():
        api, pid = key
        other = test[key]
        if row["inputs"] != other["inputs"] or row["source"] != other["source"]:
            raise ValueError(f"Different position contexts: {key}")
        left, right = row["analytics"], other["analytics"]
        exact[api] += left == right
        if api == "checker_play":
            decisions_changed[api] += left["moves"][0]["board"] != right["moves"][0]["board"]
        if api == "cube_action":
            decisions_changed[api] += left["optimal_action"] != right["optimal_action"]
        if key in completed and api != "post_move_analytics":
            ref = completed[key]
            for arm, prediction in (("baseline", left), ("candidate", right)):
                if api == "checker_play":
                    board = tuple(prediction["moves"][0]["board"])
                    if ref["moves"][0]["eval_level"] != "Rollout":
                        raise ValueError(f"Best reference move has no rollout: {pid}")
                    # A normal public checker call can leave another legal move
                    # at its filter depth. Its true rollout loss is then unknown.
                    target = truth.get((key, board))
                    loss = max(0.0, -target["equity_diff"]) if target else None
                    decisions[arm, api].append((board == tuple(ref["moves"][0]["board"]), loss))
                else:
                    decisions[arm, api].append((prediction["optimal_action"] == ref["optimal_action"], None))
        original, candidate = dict(projected(api, left)), dict(projected(api, right))
        if original.keys() != candidate.keys():
            raise ValueError(f"Different legal candidate sets: {key}")
        if key in completed and original.keys() != dict(projected(api, completed[key])).keys():
            raise ValueError(f"Incomplete rollout candidate set for {key}")
        for board, item in original.items():
            old, new = dict(metrics(item)), dict(metrics(candidate[board]))
            for field, value in old.items():
                delta = abs(new[field] - value)
                if not math.isfinite(delta):
                    raise ValueError(f"Non-finite prediction {key}/{field}")
                shifts[api, field].append(delta)
            actual = truth.get((key, board))
            if actual is None:
                continue
            matched.add((key, board))
            coverage[api] += 1
            for field, target in actual.items():
                a, b = abs(old[field] - target), abs(new[field] - target)
                if not all(map(math.isfinite, (a, b))):
                    raise ValueError(f"Non-finite error {key}/{field}")
                groups[api, field].append((a, b))
                origin_errors[origins.get(key, "frozen_full_candidate_rollout"), api, field].append((a, b))
                if key in completed:
                    complete_case_errors[api, field][pid].append((a, b))
                limit = 0.0025 if field.startswith("p.") else 0.005
                if b - a > limit:
                    regressions.append({"api":api, "position_id":pid, "board":board, "field":field,
                                        "baseline":old[field], "candidate":new[field], "truth":target})
    if matched != truth.keys():
        raise ValueError("Some rollout targets could not be matched to a prediction")
    decision_accuracy = []
    for (arm, api), outcomes in sorted(decisions.items()):
        row = {"arm":arm, "api":api, "n":len(outcomes), "same_best_action":sum(same for same, _ in outcomes)}
        if api == "checker_play":
            losses = [loss for _, loss in outcomes if loss is not None]
            row.update(moves_with_rollout_for_loss=len(losses),
                       moves_without_rollout_for_loss=len(outcomes) - len(losses),
                       mean_equity_lost=mean(losses) if losses else None,
                       max_equity_lost=max(losses, default=None),
                       moves_within_0_005_of_rollout_best=sum(loss <= 0.005 for loss in losses))
        decision_accuracy.append(row)
    error_metrics = []
    for (api, field), points in sorted(groups.items()):
        limit = 0.0025 if field.startswith("p.") else 0.005
        details = summary(points, limit)
        # An endpoint-weighted MAE gives more weight to rolls with many legal
        # moves. Provide the equal-position average separately for full cases.
        positions = complete_case_errors[api, field]
        details["complete_positions"] = len(positions)
        for column, arm in enumerate(("baseline", "candidate")):
            position_maes = [mean(pair[column] for pair in errors) for errors in positions.values()]
            details[arm + "_position_weighted_mae"] = mean(position_maes) if position_maes else None
        error_metrics.append({"api":api, "field":field, **details})
    report = {"baseline_input":file_identity(args.baseline), "candidate_input":file_identity(args.candidate),
              "reference_positions_complete": len(completed), "reference_values_by_endpoint": coverage,
              "reference_positions_by_origin": Counter(origins.values()),
              "checker_move_evaluation_levels": reference_moves,
              "rollout_decision_accuracy":decision_accuracy,
              "exact_api_outputs_out_of_100":exact, "best_action_changes":decisions_changed,
              "material_probability":0.0025, "material_equity":0.005,
              # For any rollout value t, ||new-t| - |old-t|| <= |new-old|.
              "prediction_shifts":[{"api":api,"field":f,"n":len(d),"max":max(d),
                                     "over_material":sum(x > (0.0025 if f.startswith("p.") else 0.005) for x in d)}
                                    for (api,f),d in sorted(shifts.items())],
              "error_metrics":error_metrics,
              "error_metrics_by_reference_origin":[
                  {"reference_origin": origin, "api": api, "field": field,
                   **summary(points, 0.0025 if field.startswith("p.") else 0.005)}
                  for (origin, api, field), points in sorted(origin_errors.items())],
              "material_regressions":regressions}
    out = inside_repo(args.output)
    out.parent.mkdir(exist_ok=True, parents=True)
    write_json(out, report)
    print(f"Exact outputs: {dict(exact)}; scored reference endpoints: {dict(coverage)}; "
          f"materially worse values: {len(regressions)}", flush=True)
    return len(completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DATA / "baseline_stage11/results.jsonl")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--truth", type=Path, default=DATA / "rollout_positions/results.jsonl")
    parser.add_argument("--units", type=Path, help="Include completed checker moves of unfinished positions")
    status = parser.add_mutually_exclusive_group()
    status.add_argument("--require-complete", action="store_true", help="Require references for all 300 calls")
    status.add_argument("--follow", action="store_true",
                        help="Refresh when rollout coverage increases and finish at all 300 calls")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, test = (predictions(inside_repo(p)) for p in (args.baseline, args.candidate))
    if base.keys() != test.keys():
        raise ValueError("Prediction position IDs differ")
    if not args.follow:
        score(args, base, test)
        return
    coverage_path = inside_repo(args.truth).parent / "coverage.json"
    last_coverage = None
    while True:
        coverage = coverage_path.read_bytes()
        if coverage != last_coverage:
            if score(args, base, test) == len(base):
                return
            last_coverage = coverage
        time.sleep(60)


if __name__ == "__main__":
    main()
