"""
Orchestrate parallel rollouts on AWS ECS Fargate.

Maintains a continuous pool of max_concurrent tasks. As each task completes,
its result is saved and a new task is immediately launched to replace it.
This keeps ECS utilization at 100% at all times. Fully resumable — re-run
to continue from where we left off.

Usage:
  python python/cloud_rollout.py                    # Run all pending rollouts
  python python/cloud_rollout.py --batch-size 30    # Custom batch size
  python python/cloud_rollout.py --dry-run          # Show what would be launched
"""

import os
import sys
import json
import time
import uuid
import argparse
import datetime

# Setup import paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
build_dir = os.path.join(project_dir, 'build')

if sys.platform == 'win32':
    cuda_x64 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_x64):
        os.add_dll_directory(cuda_x64)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)

sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

import boto3
from botocore.exceptions import ClientError

BENCHMARK_DIR = os.path.join(project_dir, 'data', 'benchmark_pr')
LOGS_DIR = os.path.join(project_dir, 'logs')

# AWS config
AWS_REGION = 'us-east-1'
S3_BUCKET = 'bgbot-rollout-data'
S3_PREFIX = 'benchmark-pr'
ECS_CLUSTER = 'bgbot-rollout'
ECS_TASK_DEF = 'bgbot-rollout-worker'

# Rollout config
N_TRIALS = 1296
DECISION_PLY = 1


def log(log_file, msg):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(line + '\n')


def load_positions_needing_rollout():
    """Load decisions and find positions that need rollouts."""
    decisions_file = os.path.join(BENCHMARK_DIR, 'decisions.json')
    rollout_file = os.path.join(BENCHMARK_DIR, 'rollouts.jsonl')

    if not os.path.exists(decisions_file):
        print(f"ERROR: {decisions_file} not found. Run generate_benchmark_pr.py first.")
        sys.exit(1)

    with open(decisions_file, 'r') as f:
        saved = json.load(f)
    decisions = saved['decisions']

    # Collect unique positions
    positions = {}
    for dec in decisions:
        for cand in dec['candidates']:
            key = tuple(cand)
            if key not in positions:
                positions[key] = cand

    # Load completed rollouts
    completed = set()
    if os.path.exists(rollout_file):
        with open(rollout_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                completed.add(tuple(rec['board']))

    remaining = {k: v for k, v in positions.items() if k not in completed}
    return remaining, len(completed), len(positions)


def is_race_position(board):
    """Check if a position is a pure race (no contact between pieces)."""
    if board[25] > 0:  # p1 on bar
        return False
    if board[0] > 0:  # p2 on bar
        return False
    p1_back = 0
    p2_back = 25
    for i in range(1, 25):
        if board[i] > 0:
            p1_back = i
        if board[i] < 0 and i < p2_back:
            p2_back = i
    return p1_back < p2_back


def classify_positions(positions):
    """Split positions into race and contact lists."""
    race = []
    contact = []
    for pos in positions.values():
        if is_race_position(pos):
            race.append(pos)
        else:
            contact.append(pos)
    return contact, race


def make_batches(contact_positions, race_positions, batch_size):
    """Create batch lists (without uploading). Returns list of (positions, is_race)."""
    contact_batch_size = batch_size
    race_batch_size = batch_size * 8

    batches = []
    for i in range(0, len(contact_positions), contact_batch_size):
        batches.append((contact_positions[i:i + contact_batch_size], False))
    for i in range(0, len(race_positions), race_batch_size):
        batches.append((race_positions[i:i + race_batch_size], True))
    return batches


def get_network_config():
    """Get VPC/subnet/security group for Fargate tasks."""
    ec2 = boto3.client('ec2', region_name=AWS_REGION)

    vpcs = ec2.describe_vpcs(Filters=[{'Name': 'isDefault', 'Values': ['true']}])
    if not vpcs['Vpcs']:
        raise RuntimeError("No default VPC found.")
    vpc_id = vpcs['Vpcs'][0]['VpcId']

    subnets = ec2.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])
    subnet_ids = [s['SubnetId'] for s in subnets['Subnets']]

    sgs = ec2.describe_security_groups(
        Filters=[
            {'Name': 'vpc-id', 'Values': [vpc_id]},
            {'Name': 'group-name', 'Values': ['default']}
        ])
    sg_id = sgs['SecurityGroups'][0]['GroupId']

    return {
        'awsvpcConfiguration': {
            'subnets': subnet_ids[:3],
            'securityGroups': [sg_id],
            'assignPublicIp': 'ENABLED'
        }
    }


def count_running_tasks(ecs):
    """Count currently running tasks in the cluster."""
    try:
        running = []
        paginator = ecs.get_paginator('list_tasks')
        for page in paginator.paginate(cluster=ECS_CLUSTER, desiredStatus='RUNNING'):
            running.extend(page.get('taskArns', []))
        return len(running)
    except Exception:
        return 0


def load_existing_rollout_keys():
    """Load set of board tuples already in rollouts.jsonl."""
    rollout_file = os.path.join(BENCHMARK_DIR, 'rollouts.jsonl')
    existing = set()
    if os.path.exists(rollout_file):
        with open(rollout_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                existing.add(tuple(rec['board']))
    return existing


def save_results(results_list, existing_keys, log_file):
    """Append new rollout results to rollouts.jsonl. Returns count of new positions."""
    rollout_file = os.path.join(BENCHMARK_DIR, 'rollouts.jsonl')
    n_new = 0
    with open(rollout_file, 'a') as f:
        for batch_data in results_list:
            for r in batch_data['results']:
                key = tuple(r['board'])
                if key not in existing_keys:
                    rec = {'board': r['board'], 'equity': r['equity'], 'se': r['se']}
                    f.write(json.dumps(rec) + '\n')
                    existing_keys.add(key)
                    n_new += 1
    return n_new


def upload_and_launch(s3, ecs, batch_idx, positions, network_config, run_id, log_file):
    """Upload one batch to S3 and launch its ECS task. Returns (batch_idx, batch_key, result_key, n_pos, task_arn) or None on failure."""
    batch_key = f"{S3_PREFIX}/{run_id}/batches/batch_{batch_idx:05d}.json"
    result_key = f"{S3_PREFIX}/{run_id}/results/batch_{batch_idx:05d}.json"
    n_pos = len(positions)

    batch_data = json.dumps({'positions': positions, 'batch_id': batch_idx})
    s3.put_object(Bucket=S3_BUCKET, Key=batch_key, Body=batch_data)

    response = ecs.run_task(
        cluster=ECS_CLUSTER,
        taskDefinition=ECS_TASK_DEF,
        launchType='FARGATE',
        networkConfiguration=network_config,
        overrides={
            'containerOverrides': [{
                'name': 'bgbot-rollout-worker',
                'environment': [
                    {'name': 'S3_BUCKET', 'value': S3_BUCKET},
                    {'name': 'BATCH_KEY', 'value': batch_key},
                    {'name': 'RESULT_KEY', 'value': result_key},
                    {'name': 'N_TRIALS', 'value': str(N_TRIALS)},
                    {'name': 'DECISION_PLY', 'value': str(DECISION_PLY)},
                ],
            }],
        },
        count=1,
    )

    if response['tasks']:
        task_arn = response['tasks'][0]['taskArn']
        return (batch_idx, batch_key, result_key, n_pos, task_arn)
    else:
        failures = response.get('failures', [])
        reason = failures[0].get('reason', '') if failures else 'unknown'
        # Clean up uploaded batch since task didn't launch
        try:
            s3.delete_object(Bucket=S3_BUCKET, Key=batch_key)
        except Exception:
            pass
        return None


def salvage_orphaned_results(s3, existing_keys, log_file):
    """Check S3 for any result files from previous runs and save them."""
    paginator = s3.get_paginator('list_objects_v2')
    results = []
    # Search entire prefix tree for result files (any run ID)
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f'{S3_PREFIX}/'):
        for obj in page.get('Contents', []):
            if '/results/' in obj['Key']:
                try:
                    resp = s3.get_object(Bucket=S3_BUCKET, Key=obj['Key'])
                    data = json.loads(resp['Body'].read())
                    results.append(data)
                except Exception:
                    continue
    if results:
        return save_results(results, existing_keys, log_file)
    return 0


def cleanup_all_s3(s3, log_file):
    """Remove all files under the S3 prefix (all run IDs)."""
    paginator = s3.get_paginator('list_objects_v2')
    objects = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f'{S3_PREFIX}/'):
        for obj in page.get('Contents', []):
            objects.append({'Key': obj['Key']})
    if objects:
        for i in range(0, len(objects), 1000):
            s3.delete_objects(Bucket=S3_BUCKET, Delete={'Objects': objects[i:i+1000]})
        log(log_file, f"Cleaned up {len(objects)} leftover S3 objects")


def main():
    parser = argparse.ArgumentParser(description='Cloud-parallel rollouts on AWS ECS')
    parser.add_argument('--batch-size', type=int, default=30,
                        help='Contact positions per batch (race=8x this). Default: 30')
    parser.add_argument('--max-concurrent', type=int, default=100,
                        help='Max concurrent ECS tasks (default: 100, limit=400 vCPUs/4=100)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be launched without running')
    parser.add_argument('--poll-interval', type=int, default=15,
                        help='Seconds between result polls (default: 15)')
    args = parser.parse_args()

    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, 'cloud_rollout.log')

    log(log_file, "=== Cloud Rollout Orchestrator (continuous pool) ===")

    # Load positions
    remaining, n_done, n_total = load_positions_needing_rollout()
    log(log_file, f"Positions: {n_done}/{n_total} done, {len(remaining)} remaining")

    if not remaining:
        log(log_file, "All positions already rolled out!")
        return

    # Classify and create batches
    contact_positions, race_positions = classify_positions(remaining)
    all_batches = make_batches(contact_positions, race_positions, args.batch_size)
    n_batches = len(all_batches)

    n_contact = len(contact_positions)
    n_race = len(race_positions)
    contact_batch_size = args.batch_size
    race_batch_size = args.batch_size * 8

    # Estimate cost and time
    est_contact_time = 25
    est_race_time = 3
    est_batch_time = max(est_contact_time * contact_batch_size,
                         est_race_time * race_batch_size)
    est_total_pos_seconds = n_contact * est_contact_time + n_race * est_race_time
    # Continuous pool: wall clock ≈ total_batch_time / max_concurrent
    est_wall_clock = (n_batches * est_batch_time) / args.max_concurrent + 60
    est_vcpu_hours = 4 * est_total_pos_seconds / 3600
    est_cost = est_vcpu_hours * 0.04048 + n_batches * 8 * (est_batch_time / 3600) * 0.004445

    n_contact_batches = sum(1 for _, is_race in all_batches if not is_race)
    n_race_batches = sum(1 for _, is_race in all_batches if is_race)

    log(log_file, f"Plan: {n_batches} batches ({n_contact_batches} contact of {contact_batch_size}, "
        f"{n_race_batches} race of {race_batch_size}), max {args.max_concurrent} concurrent")
    log(log_file, f"  Positions: {n_contact} contact, {n_race} race")
    log(log_file, f"  Estimated wall-clock: ~{datetime.timedelta(seconds=int(est_wall_clock))}")
    log(log_file, f"  Estimated cost: ~${est_cost:.2f}")

    if args.dry_run:
        log(log_file, "DRY RUN - not launching tasks")
        return

    # AWS clients
    s3 = boto3.client('s3', region_name=AWS_REGION)
    ecs = boto3.client('ecs', region_name=AWS_REGION)

    existing_keys = load_existing_rollout_keys()

    # Salvage any completed results from orphaned tasks of previous runs
    salvaged = salvage_orphaned_results(s3, existing_keys, log_file)
    if salvaged > 0:
        log(log_file, f"Salvaged {salvaged} positions from previous run's orphaned results")
        # Reload remaining positions
        remaining, n_done, n_total = load_positions_needing_rollout()
        contact_positions, race_positions = classify_positions(remaining)
        all_batches = make_batches(contact_positions, race_positions, args.batch_size)
        n_batches = len(all_batches)
        log(log_file, f"After salvage: {n_done}/{n_total} done, {len(remaining)} remaining, "
            f"{n_batches} batches")

    # Clean up old S3 objects. Orphaned ECS tasks will fail gracefully.
    cleanup_all_s3(s3, log_file)

    running = count_running_tasks(ecs)
    if running > 0:
        log(log_file, f"Note: {running} orphaned tasks still running from previous run. "
            f"They'll count against our pool limit until they finish.")

    # Unique run ID prevents S3 key collisions between runs
    run_id = uuid.uuid4().hex[:8]
    log(log_file, f"Run ID: {run_id}")

    # Get network config
    log(log_file, "Getting network configuration...")
    network_config = get_network_config()
    start_time = time.perf_counter()
    next_batch_idx = 0      # Index into all_batches for next batch to launch
    total_completed = 0
    total_failed = 0
    total_positions_saved = 0

    # Active tasks: batch_idx -> (batch_idx, batch_key, result_key, n_pos, task_arn)
    active = {}

    log(log_file, f"Starting continuous pool (target: {args.max_concurrent} concurrent tasks)")

    # Main loop: keep pool full, collect results
    consecutive_launch_errors = 0
    stale_rounds = 0
    max_stale = 60  # 15 min at 15s interval

    while active or next_batch_idx < n_batches:
        # --- Fill pool: launch tasks up to max_concurrent ---
        # Check actual running tasks (includes orphans from previous runs)
        actual_running = count_running_tasks(ecs)
        can_launch = max(0, args.max_concurrent - actual_running)
        launched_this_round = 0
        while next_batch_idx < n_batches and launched_this_round < can_launch:
            positions, is_race = all_batches[next_batch_idx]
            try:
                result = upload_and_launch(
                    s3, ecs, next_batch_idx, positions, network_config, run_id, log_file)
                if result:
                    batch_idx, batch_key, result_key, n_pos, task_arn = result
                    active[batch_idx] = result
                    consecutive_launch_errors = 0
                    next_batch_idx += 1
                    launched_this_round += 1
                else:
                    # vCPU limit or similar — stop filling, let some finish first
                    break
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code in ('Throttling', 'ThrottlingException',
                                  'RequestLimitExceeded', 'ServiceUnavailable'):
                    consecutive_launch_errors += 1
                    if consecutive_launch_errors >= 10:
                        log(log_file, f"  Too many launch throttle errors, waiting...")
                    break  # Stop filling, wait for poll cycle
                else:
                    raise  # Auth errors etc. — abort
            except Exception:
                raise  # Abort on unexpected errors

        # --- Poll for completed results ---
        time.sleep(args.poll_interval)

        done_this_round = []
        for batch_idx, (_, batch_key, result_key, n_pos, task_arn) in active.items():
            try:
                resp = s3.get_object(Bucket=S3_BUCKET, Key=result_key)
                data = json.loads(resp['Body'].read())
                done_this_round.append((batch_idx, data, batch_key, result_key))
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    continue
                continue
            except Exception:
                continue

        # Save results and clean up completed tasks
        if done_this_round:
            stale_rounds = 0
            results_to_save = [data for _, data, _, _ in done_this_round]
            n_saved = save_results(results_to_save, existing_keys, log_file)
            total_positions_saved += n_saved
            total_completed += len(done_this_round)

            # Clean up S3 objects for completed batches
            objects_to_delete = []
            for batch_idx, _, batch_key, result_key in done_this_round:
                del active[batch_idx]
                objects_to_delete.append({'Key': batch_key})
                objects_to_delete.append({'Key': result_key})
            if objects_to_delete:
                try:
                    s3.delete_objects(Bucket=S3_BUCKET,
                                     Delete={'Objects': objects_to_delete})
                except Exception:
                    pass  # Non-critical cleanup
        else:
            stale_rounds += 1

        # --- Progress logging ---
        if done_this_round or stale_rounds % 8 == 0:
            elapsed = time.perf_counter() - start_time
            remaining_batches = n_batches - next_batch_idx + len(active)
            rate = total_completed / elapsed if elapsed > 0 else 0
            eta_s = remaining_batches / rate if rate > 0 else 0
            eta = datetime.timedelta(seconds=int(eta_s))
            log(log_file, f"  Progress: {total_completed}/{n_batches} batches done, "
                f"{total_positions_saved} positions saved, "
                f"{len(active)} active, {n_batches - next_batch_idx} queued, "
                f"ETA {eta}, {elapsed:.0f}s elapsed")

        # Stale timeout: if no results for a long time, something is wrong
        if stale_rounds >= max_stale and active:
            log(log_file, f"  Stale timeout: no results for {max_stale * args.poll_interval}s "
                f"with {len(active)} active tasks. Abandoning them.")
            # Clean up abandoned batch S3 objects
            objects_to_delete = []
            for batch_idx, (_, batch_key, result_key, _, _) in active.items():
                objects_to_delete.append({'Key': batch_key})
                objects_to_delete.append({'Key': result_key})
                total_failed += 1
            if objects_to_delete:
                try:
                    s3.delete_objects(Bucket=S3_BUCKET,
                                     Delete={'Objects': objects_to_delete})
                except Exception:
                    pass
            active.clear()
            stale_rounds = 0
            # Note: abandoned batches won't be retried this run.
            # Re-running the script will pick them up since their positions
            # won't be in rollouts.jsonl.

    elapsed = time.perf_counter() - start_time
    status = 'complete' if total_completed == n_batches else 'stopped'
    log(log_file, f"=== Cloud rollout {status} ===")
    log(log_file, f"  {total_completed}/{n_batches} batches, "
        f"{total_positions_saved} positions saved, "
        f"{total_failed} failed, "
        f"{datetime.timedelta(seconds=int(elapsed))} elapsed")
    log(log_file, f"  Total rollouts in file: {len(existing_keys)}")


if __name__ == '__main__':
    main()
