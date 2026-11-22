# SPDX-License-Identifier: MPL-2.0
"""Compare paired public analytics at deterministic or truncated levels against frozen rollouts."""
import argparse
from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
from statistics import mean

from benchmark_multi_ply import APIS, DATA, TRUNCATED_LEVELS, file_identity, inside_repo, timing_stats, write_json
from compare_multi_ply_accuracy import metrics, projected, records, summary


def level_name(name):
    for ply in range(1, 5):
        if name in (f"{ply}P", f"{ply}ply"):
            return f"{ply}P"
    for label, preset in TRUNCATED_LEVELS.items():
        if name in (label, preset):
            return label
    raise ValueError(f"Unknown evaluation level {name!r}")


def insert(rows, row):
    key = row["api"], row["position_id"]
    if key in rows:
        raise ValueError(f"Duplicate call {key}")
    rows[key] = row


def inputs(row):
    result = dict(row["inputs"])
    if row["api"] == "cube_action":
        result.setdefault("incl_2ply_details", False)
    return result


def load_runs(paths, wanted):
    groups = defaultdict(dict)
    for path in paths:
        for row in records(path):
            level = level_name(row["level"])
            if level not in wanted:
                continue
            seconds = row["elapsed_seconds"]
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f"Invalid timing {seconds} at {row['position_id']}")
            if level in TRUNCATED_LEVELS:
                key = "truncated-benchmark-v1:" + row["source"]["key"]
                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
                if type(row.get("seed")) is not int or row["seed"] != seed:
                    raise ValueError(f"Invalid seed at {row['position_id']} in {path}")
            insert(groups[level, row["repeat"]], row)
    if {level for level, _ in groups} != set(wanted):
        raise ValueError(f"Missing level in {paths} (expected {wanted})")
    return groups


def endpoints(api, value):
    pairs = list(projected(api, value))
    result = dict(pairs)
    if not pairs or len(result) != len(pairs):
        raise ValueError("Empty or duplicate checker endpoints")
    return result


def threshold(field):
    return 0.0025 if field.startswith(("p.", "probs.", "prob_std_errors[")) else 0.005


def compare_fields(left, right, path, shifts, changes, location):
    """Inspect public values recursively. Checker moves have already been paired by endpoint."""
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left.keys() | right.keys():
            field = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                changes[field] += 1
            else:
                compare_fields(left[key], right[key], field, shifts, changes, location)
    elif isinstance(left, list) and isinstance(right, list) and path != "board":
        if len(left) != len(right):
            changes[path + ".length"] += 1
        for i, (a, b) in enumerate(zip(left, right)):
            compare_fields(a, b, f"{path}[{i}]", shifts, changes, location)
    elif type(left) in (int, float) and type(right) in (int, float):
        delta = abs(left - right)
        if not math.isfinite(delta):
            raise ValueError(f"Non-finite result at {location}/{path}")
        shifts[path].append((delta, location))
        if delta:
            changes[path] += 1
    elif left != right or type(left) is not type(right):
        changes[path] += 1


def outcome(api, predicted, truth, ref_moves):
    if api == "cube_action":
        # Measure the doubler's loss allowing the opponent the best rollout response.
        chosen = (min(truth["equity_dt"], truth["equity_dp"]) if predicted["should_double"]
                  else truth["equity_nd"])
        return predicted["optimal_action"] == truth["optimal_action"], max(0, truth["optimal_equity"] - chosen)
    chosen = tuple(predicted["moves"][0]["board"])
    best = truth["moves"][0]
    if best["eval_level"] != "Rollout":
        raise ValueError("Best reference move has no rollout")
    ref = ref_moves[chosen]
    loss = max(0, -ref["equity_diff"]) if ref["eval_level"] == "Rollout" else None
    return chosen == tuple(best["board"]), loss


def decision_stats(outcomes):
    losses = [loss for _, loss in outcomes if loss is not None]
    return {"decisions": len(outcomes), "agree_with_rollout": sum(same for same, _ in outcomes),
            "scorable_regrets": len(losses), "unscorable_regrets": len(outcomes) - len(losses),
            "mean_regret_equity": mean(losses) if losses else None, "max_regret_equity": max(losses, default=None),
            "within_0_005_equity_of_best": sum(loss <= 0.005 for loss in losses)}


def score_api(api, pairs):
    shifts, changes = defaultdict(list), Counter()
    errors, position_errors = defaultdict(list), defaultdict(lambda: defaultdict(list))
    observed_values = defaultdict(list)
    outcomes, times = defaultdict(list), defaultdict(list)
    source_levels = Counter()
    materially_changed, changed_outputs = [], []
    exact = same_order = same_best = 0
    for left, right, reference in pairs:
        pid, repeat = left["position_id"], left["repeat"]
        if any(row[field] != reference[field] for row in (left, right)
               for field in ("source", "mode", "source_game_plan")) or not (
                   inputs(left) == inputs(right) == inputs(reference)):
            raise ValueError(f"Different input contexts for {api}/{pid}")
        a, b, truth = (row["analytics"] for row in (left, right, reference))
        exact += a == b
        if a != b:
            changed_outputs.append({"position_id": pid, "repeat": repeat})
        base_moves, new_moves, ref_moves = (endpoints(api, v) for v in (a, b, truth))
        if not (base_moves.keys() == new_moves.keys() == ref_moves.keys()):
            raise ValueError(f"Different legal endpoints for {pid}")
        if api == "checker_play":
            same_order += list(base_moves) == list(new_moves)
            same_best += next(iter(base_moves)) == next(iter(new_moves))
            compare_fields({k: v for k, v in a.items() if k != "moves"},
                           {k: v for k, v in b.items() if k != "moves"}, "", shifts, changes,
                           (pid, repeat, None))
        elif api == "cube_action":
            same_best += a["optimal_action"] == b["optimal_action"]
        for arm, row in (("baseline", left), ("candidate", right)):
            times[arm].append(row["elapsed_seconds"])
            if api != "post_move_analytics":
                outcomes[arm].append(outcome(api, row["analytics"], truth, ref_moves))
        for board, old_item in base_moves.items():
            cand_item, ref_item = new_moves[board], ref_moves[board]
            location = (pid, repeat, board)
            compare_fields(old_item, cand_item, "", shifts, changes, location)
            source_levels[ref_item["eval_level"]] += 1
            if api == "checker_play" and ref_item["eval_level"] != "Rollout":
                continue
            if ref_item["eval_level"] != "Rollout":
                raise ValueError(f"Missing rollout for {api}/{pid}")
            old, new = dict(metrics(old_item)), dict(metrics(cand_item))
            for field, target in metrics(ref_item):
                observed_values[field].append((old[field], new[field], target))
                before, after = abs(old[field] - target), abs(new[field] - target)
                if not math.isfinite(before) or not math.isfinite(after):
                    raise ValueError(f"Non-finite error at {pid}/{field}")
                errors[field].append((before, after))
                position_errors[field][pid].append((before, after))
                difference = after - before
                if abs(difference) > threshold(field):
                    materially_changed.append({"position_id": pid, "repeat": repeat, "checker_endpoint": board,
                                               "field": field, "baseline": old[field], "candidate": new[field],
                                               "truth": target, "change_in_absolute_error": difference})

    timing = {arm: timing_stats(values) for arm, values in times.items()}
    baseline_total, candidate_total = (timing[k]["total_seconds"] for k in ("baseline", "candidate"))
    timing.update(speedup=baseline_total / candidate_total,
                  seconds_saved=baseline_total - candidate_total,
                  candidate_faster_calls=sum(b < a for a, b in zip(times["baseline"], times["candidate"])))
    error_report = {}
    for field, values in sorted(errors.items()):
        item = summary(values, threshold(field))
        item["best_change_in_abs_error"] = min(b - a for a, b in values)
        for i, arm in enumerate(("baseline", "candidate", "rollout")):
            item[arm + "_mean_value"] = mean(v[i] for v in observed_values[field])
        for i, arm in enumerate(("baseline", "candidate")):
            item[arm + "_max_abs_error"] = max(v[i] for v in values)
            item[arm + "_position_weighted_mae"] = mean(
                mean(v[i] for v in pos) for pos in position_errors[field].values())
        error_report[field] = item
    shift_report = {}
    for field, values in sorted(shifts.items()):
        largest, location = max(values, key=lambda pair: pair[0])
        shift_report[field] = {"n": len(values), "max_abs_shift": largest,
                               "worst_location": location if largest else None,
                               "over_material": sum(x > threshold(field) for x, _ in values)}
    report = {"timing": timing, "exact_analytics": exact, "changed_outputs": changed_outputs,
              "public_field_changes": dict(sorted(changes.items())),
              "numeric_shifts": shift_report, "rollout_error_by_field": error_report,
              "reference_endpoint_levels": dict(source_levels),
              "material_changes_in_rollout_error": materially_changed}
    if outcomes:
        report["same_best_action"] = same_best
        report["decision_accuracy"] = {arm: decision_stats(v) for arm, v in outcomes.items()}
    if api == "checker_play":
        report["same_full_move_order"] = same_order
    return report


def compare(baseline, candidate, truth_path, levels):
    truth = {}
    for row in records(truth_path):
        insert(truth, row)
    if Counter(api for api, _ in truth) != {api: 100 for api in APIS}:
        raise ValueError("Expected 100 distinct rollout decisions per API")
    old, new = (load_runs(paths, levels) for paths in (baseline, candidate))
    if old.keys() != new.keys():
        raise ValueError("Different levels or repeats")
    by_level = {}
    for level in levels:
        pairs = defaultdict(list)
        for group in sorted(k for k in old if k[0] == level):
            if old[group].keys() != truth.keys() or new[group].keys() != truth.keys():
                raise ValueError(f"Incomplete or different positions for {group}")
            if list(old[group]) != list(new[group]):
                raise ValueError(f"Different timing call order for {group}")
            for key, left in old[group].items():
                pairs[key[0]].append((left, new[group][key], truth[key]))
        by_api = {api: score_api(api, pairs[api]) for api in APIS}
        totals = {arm: sum(result["timing"][arm]["total_seconds"] for result in by_api.values())
                  for arm in ("baseline", "candidate")}
        by_level[level] = {"by_api": by_api, "total_call_seconds": totals,
                           "speedup": totals["baseline"] / totals["candidate"]}
    return {"inputs": {"baseline": list(map(file_identity, baseline)), "candidate": list(map(file_identity, candidate)),
                       "truth": file_identity(truth_path)}, "material_probability": 0.0025, "material_equity": 0.005,
            "reference_positions_by_origin": Counter(row.get("reference_origin", "unspecified") for row in truth.values()),
            "levels": by_level}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--truth", type=Path, default=DATA / "rollout_positions/results.jsonl")
    parser.add_argument("--levels", nargs="+", type=level_name, default=list(TRUNCATED_LEVELS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(list(map(inside_repo, args.baseline)), list(map(inside_repo, args.candidate)),
                     inside_repo(args.truth), args.levels)
    out = inside_repo(args.output)
    out.parent.mkdir(exist_ok=True, parents=True)
    write_json(out, report)
    for level, result in report["levels"].items():
        for api, stats in result["by_api"].items():
            times = stats["timing"]
            print(f"{level} {api}: {times['baseline']['total_seconds']:.3f}s -> "
                  f"{times['candidate']['total_seconds']:.3f}s ({times['speedup']:.2f}x), "
                  f"{stats['exact_analytics']}/{times['baseline']['count']} identical outputs")


if __name__ == "__main__":
    main()
