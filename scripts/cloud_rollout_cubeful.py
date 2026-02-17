"""
Orchestrate parallel rollouts on AWS ECS Fargate for the Cubeful Benchmark PR.

Reads decisions from data/benchmark_cubeful_pr/decisions.json, identifies all
unique positions needing rollouts (both checker-play candidates and cube
pre-roll positions), and dispatches them to ECS workers.

For cube positions, the flipped board is rolled out (post-move from opponent's
perspective). The scoring script will invert the rollout probs to get pre-roll
cubeless probs and apply Janowski conversion.

Both checker-play and cube positions use the same rollout config:
  1-ply decision, 1296 trials, VR=0, truncation=0, late=0@3

Usage:
  python python/cloud_rollout_cubeful.py                    # Run all pending
  python python/cloud_rollout_cubeful.py --dry-run          # Estimate only
  python python/cloud_rollout_cubeful.py --batch-size 15    # Custom batch size
"""

import os
import sys
import json
import time
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

import bgbot_cpp

BENCHMARK_DIR = os.path.join(project_dir, 'data', 'benchmark_cubeful_pr')
LOGS_DIR = os.path.join(project_dir, 'logs')

# AWS config
AWS_REGION = 'us-east-1'
S3_BUCKET = 'bgbot-rollout-data'
S3_PREFIX = 'benchmark-cubeful-pr'
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
    """Load decisions and find all unique positions that need rollouts."""
    decisions_file = os.path.join(BENCHMARK_DIR, 'decisions.json')
    rollout_file = os.path.join(BENCHMARK_DIR, 'rollouts.jsonl')

    if not os.path.exists(decisions_file):
        print(f"ERROR: {decisions_file} not found. Run generate_cubeful_benchmark_pr.py first.")
        sys.exit(1)

    with open(decisions_file, 'r') as f:
        saved = json.load(f)

    checker_decisions = saved['checker_decisions']
    cube_decisions = saved['cube_decisions']

    # Collect unique positions from checker-play decisions (post-move candidates)
    positions = {}
    for dec in checker_decisions:
        for cand in dec['candidates']:
            key = tuple(cand)
            if key not in positions:
                positions[key] = cand

    # Collect unique positions from cube decisions (flipped pre-roll boards)
    bgbot_cpp.init_escape_tables()
    for dec in cube_decisions:
        board = dec['board']
        flipped = bgbot_cpp.flip_board(board)
        key = tuple(flipped)
        if key not in positions:
            positions[key] = list(flipped)

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
    """Check if a position is a pure race (no contact)."""
    if board[25] > 0:
        return False
    if board[0] > 0:
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
    """Create batch lists. Returns list of (positions, is_race)."""
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
    try:
        running = []
        paginator = ecs.get_paginator('list_tasks')
        for page in paginator.paginate(cluster=ECS_CLUSTER, desiredStatus='RUNNING'):
            running.extend(page.get('taskArns', []))
        return len(running)
    except Exception:
        return 0


def load_existing_rollout_keys():
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


def save_batch_results(batch_results, existing_keys, log_file):
    rollout_file = os.path.join(BENCHMARK_DIR, 'rollouts.jsonl')
    n_new = 0
    with open(rollout_file, 'a') as f:
        for batch_data in batch_results:
            for r in batch_data['results']:
                key = tuple(r['board'])
                if key not in existing_keys:
                    rec = {
                        'board': r['board'],
                        'equity': r['equity'],
                        'se': r['se'],
                        'probs': r.get('probs'),
                    }
                    f.write(json.dumps(rec) + '\n')
                    existing_keys.add(key)
                    n_new += 1
    return n_new


def run_wave(s3, ecs, wave_batches, wave_offset, network_config,
             max_concurrent, poll_interval, existing_keys, log_file):
    """Upload, launch, poll, and save one wave of batches."""
    n_wave = len(wave_batches)

    # Upload batches
    batch_keys = []
    for i, (positions, is_race) in enumerate(wave_batches):
        batch_id = wave_offset + i
        batch_key = f"{S3_PREFIX}/batches/batch_{batch_id:05d}.json"
        result_key = f"{S3_PREFIX}/results/batch_{batch_id:05d}.json"
        batch_data = json.dumps({'positions': positions, 'batch_id': batch_id})
        s3.put_object(Bucket=S3_BUCKET, Key=batch_key, Body=batch_data)
        batch_keys.append((batch_id, batch_key, result_key, len(positions)))

    log(log_file, f"  Uploaded {n_wave} batches")

    # Launch tasks
    launched = {}
    failed = []
    idx = 0
    consecutive_errors = 0

    while idx < len(batch_keys):
        running = count_running_tasks(ecs)
        can_launch = max(0, max_concurrent - running)
        launched_this_round = 0
        hit_limit = False

        while idx < len(batch_keys) and launched_this_round < can_launch and not hit_limit:
            batch_id, batch_key, result_key, n_pos = batch_keys[idx]
            try:
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
                    launched[batch_id] = response['tasks'][0]['taskArn']
                    launched_this_round += 1
                    idx += 1
                    consecutive_errors = 0
                else:
                    failures = response.get('failures', [])
                    reason = failures[0].get('reason', '') if failures else ''
                    if 'vCPU' in reason or 'limit' in reason.lower():
                        hit_limit = True
                    else:
                        log(log_file, f"    FAILED batch {batch_id}: {failures}")
                        failed.append(batch_id)
                        idx += 1
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code in ('Throttling', 'ThrottlingException',
                                  'RequestLimitExceeded', 'ServiceUnavailable'):
                    hit_limit = True
                    consecutive_errors += 1
                else:
                    raise
            except Exception:
                raise

        if consecutive_errors >= 10:
            log(log_file, f"    Too many consecutive errors, aborting wave")
            break

        if idx < len(batch_keys):
            time.sleep(30 if hit_limit else 5)

    log(log_file, f"  Launched {len(launched)}/{n_wave} tasks ({len(failed)} failed)")

    # Poll for results
    pending = {bk[0]: bk for bk in batch_keys if bk[0] in launched}
    completed_data = []
    stale_rounds = 0
    max_stale = 120
    poll_start = time.perf_counter()

    while pending:
        time.sleep(poll_interval)
        done_this_round = []

        for batch_id, (_, _, result_key, n_pos) in pending.items():
            try:
                resp = s3.get_object(Bucket=S3_BUCKET, Key=result_key)
                data = json.loads(resp['Body'].read())
                completed_data.append(data)
                done_this_round.append(batch_id)
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    continue
                continue
            except Exception:
                continue

        for batch_id in done_this_round:
            del pending[batch_id]

        if done_this_round:
            stale_rounds = 0
            new_results = completed_data[-len(done_this_round):]
            n_saved = save_batch_results(new_results, existing_keys, log_file)
        else:
            stale_rounds += 1

        if done_this_round or stale_rounds % 8 == 0:
            elapsed = time.perf_counter() - poll_start
            n_pos_done = sum(d['n_positions'] for d in completed_data)
            log(log_file, f"    Wave results: {len(completed_data)}/{len(launched)} "
                f"({n_pos_done} positions, {elapsed:.0f}s, {len(pending)} pending)")

        if stale_rounds >= max_stale:
            log(log_file, f"    Timeout waiting for {len(pending)} batches")
            break

    # Cleanup S3
    objects = []
    for _, batch_key, result_key, _ in batch_keys:
        objects.append({'Key': batch_key})
        objects.append({'Key': result_key})
    for i in range(0, len(objects), 1000):
        s3.delete_objects(Bucket=S3_BUCKET, Delete={'Objects': objects[i:i+1000]})

    n_total_saved = sum(d['n_positions'] for d in completed_data)
    return len(completed_data), len(failed), n_total_saved


def cleanup_all_s3(s3, log_file):
    paginator = s3.get_paginator('list_objects_v2')
    objects = []
    for prefix in [f'{S3_PREFIX}/batches/', f'{S3_PREFIX}/results/']:
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get('Contents', []):
                objects.append({'Key': obj['Key']})
    if objects:
        for i in range(0, len(objects), 1000):
            s3.delete_objects(Bucket=S3_BUCKET, Delete={'Objects': objects[i:i+1000]})
        log(log_file, f"Cleaned up {len(objects)} leftover S3 objects")


def main():
    parser = argparse.ArgumentParser(description='Cloud rollouts for Cubeful Benchmark PR')
    parser.add_argument('--batch-size', type=int, default=15,
                        help='Contact positions per batch (race=8x). Default: 15')
    parser.add_argument('--max-concurrent', type=int, default=100,
                        help='Max concurrent ECS tasks (default: 100)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be launched without running')
    parser.add_argument('--poll-interval', type=int, default=15,
                        help='Seconds between S3 result polls (default: 15)')
    args = parser.parse_args()

    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, 'cloud_rollout_cubeful.log')

    log(log_file, "=== Cloud Rollout Orchestrator (Cubeful Benchmark) ===")

    remaining, n_done, n_total = load_positions_needing_rollout()
    log(log_file, f"Positions: {n_done}/{n_total} done, {len(remaining)} remaining")

    if not remaining:
        log(log_file, "All positions already rolled out!")
        return

    contact_positions, race_positions = classify_positions(remaining)
    all_batches = make_batches(contact_positions, race_positions, args.batch_size)
    n_batches = len(all_batches)

    n_contact = len(contact_positions)
    n_race = len(race_positions)
    contact_batch_size = args.batch_size
    race_batch_size = args.batch_size * 8

    # Estimates
    est_contact_time = 25
    est_race_time = 3
    est_batch_time = max(est_contact_time * contact_batch_size,
                         est_race_time * race_batch_size)
    est_total_pos_seconds = n_contact * est_contact_time + n_race * est_race_time
    n_waves = (n_batches + args.max_concurrent - 1) // args.max_concurrent
    est_wall_clock = n_waves * est_batch_time + 60
    est_vcpu_hours = 4 * est_total_pos_seconds / 3600
    est_cost = est_vcpu_hours * 0.04048 + n_batches * 8 * (est_batch_time / 3600) * 0.004445

    n_contact_batches = sum(1 for _, is_race in all_batches if not is_race)
    n_race_batches = sum(1 for _, is_race in all_batches if is_race)

    log(log_file, f"Plan: {n_batches} batches ({n_contact_batches} contact of {contact_batch_size}, "
        f"{n_race_batches} race of {race_batch_size}), max {args.max_concurrent} concurrent")
    log(log_file, f"  Positions: {n_contact} contact, {n_race} race")
    log(log_file, f"  Estimated {n_waves} waves, ~{est_batch_time}s/batch")
    log(log_file, f"  Estimated wall-clock: ~{datetime.timedelta(seconds=int(est_wall_clock))}")
    log(log_file, f"  Estimated cost: ~${est_cost:.2f}")

    if args.dry_run:
        log(log_file, "DRY RUN - not launching tasks")
        return

    s3 = boto3.client('s3', region_name=AWS_REGION)
    ecs = boto3.client('ecs', region_name=AWS_REGION)

    cleanup_all_s3(s3, log_file)

    log(log_file, "Getting network configuration...")
    network_config = get_network_config()

    wave_size = args.max_concurrent
    total_completed = 0
    total_failed = 0
    total_positions_saved = 0
    existing_keys = load_existing_rollout_keys()
    start_time = time.perf_counter()

    for wave_idx in range(0, n_batches, wave_size):
        wave_batches = all_batches[wave_idx:wave_idx + wave_size]
        wave_num = wave_idx // wave_size + 1
        total_waves = (n_batches + wave_size - 1) // wave_size

        n_wave_pos = sum(len(positions) for positions, _ in wave_batches)
        log(log_file, f"Wave {wave_num}/{total_waves}: {len(wave_batches)} batches, "
            f"{n_wave_pos} positions")

        try:
            completed, failed, pos_saved = run_wave(
                s3, ecs, wave_batches, wave_idx, network_config,
                args.max_concurrent, args.poll_interval, existing_keys, log_file)
        except Exception as e:
            log(log_file, f"  Wave {wave_num} aborted: {type(e).__name__}: {e}")
            log(log_file, f"  Re-run to resume from where we left off.")
            break

        total_completed += completed
        total_failed += failed
        total_positions_saved += pos_saved

        elapsed = time.perf_counter() - start_time
        log(log_file, f"  Wave {wave_num} done. Cumulative: {total_completed} batches, "
            f"{total_positions_saved} positions, {elapsed:.0f}s elapsed")

    elapsed = time.perf_counter() - start_time
    log(log_file, f"=== Cloud rollout {'complete' if total_completed == n_batches else 'stopped'} ===")
    log(log_file, f"  {total_completed}/{n_batches} batches, "
        f"{total_positions_saved} positions saved, "
        f"{total_failed} failed, "
        f"{datetime.timedelta(seconds=int(elapsed))} elapsed")
    log(log_file, f"  Total rollouts in file: {len(existing_keys)}")


if __name__ == '__main__':
    main()
