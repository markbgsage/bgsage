"""
ECS Fargate worker: download a batch of positions from S3, run rollouts,
upload results back to S3.

Environment variables:
  S3_BUCKET       - S3 bucket name
  BATCH_KEY       - S3 key for input positions JSON
  RESULT_KEY      - S3 key for output results JSON
  N_TRIALS        - Rollout trials per position (default: 1296)
  DECISION_PLY    - Ply depth for move decisions in rollout (default: 1)
  ROLLOUT_THREADS - Threads for rollout (default: 0 = auto)
"""

import os
import sys
import json
import time
import datetime

import boto3
import bgbot_cpp

MODELS_DIR = '/app/models'
NH_PR, NH_RC, NH_AT, NH_PM, NH_AN = 200, 400, 400, 400, 400


def log(msg):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


def main():
    s3_bucket = os.environ['S3_BUCKET']
    batch_key = os.environ['BATCH_KEY']
    result_key = os.environ['RESULT_KEY']
    n_trials = int(os.environ.get('N_TRIALS', '1296'))
    decision_ply = int(os.environ.get('DECISION_PLY', '1'))
    n_threads = int(os.environ.get('ROLLOUT_THREADS', '0'))

    log(f"Worker starting: bucket={s3_bucket} batch={batch_key} "
        f"trials={n_trials} ply={decision_ply}")

    s3 = boto3.client('s3')

    # Download batch
    log("Downloading batch from S3...")
    resp = s3.get_object(Bucket=s3_bucket, Key=batch_key)
    batch = json.loads(resp['Body'].read())
    positions = batch['positions']
    log(f"  Got {len(positions)} positions")

    # Init
    bgbot_cpp.init_escape_tables()

    weights = {
        'purerace': os.path.join(MODELS_DIR, 'sl_s5_purerace.weights.best'),
        'racing': os.path.join(MODELS_DIR, 'sl_s5_racing.weights.best'),
        'attacking': os.path.join(MODELS_DIR, 'sl_s5_attacking.weights.best'),
        'priming': os.path.join(MODELS_DIR, 'sl_s5_priming.weights.best'),
        'anchoring': os.path.join(MODELS_DIR, 'sl_s5_anchoring.weights.best'),
    }

    wt = (weights['purerace'], weights['racing'], weights['attacking'],
          weights['priming'], weights['anchoring'])

    log("Creating rollout strategy...")
    rollout = bgbot_cpp.create_rollout_5nn(
        *wt, NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
        n_trials=n_trials, truncation_depth=0,
        decision_ply=decision_ply, vr_ply=0,
        n_threads=n_threads,
        late_ply=0, late_threshold=3)

    # Run rollouts
    results = []
    t_start = time.perf_counter()

    for i, board in enumerate(positions):
        t0 = time.perf_counter()
        result = rollout.rollout_position(board)
        elapsed = time.perf_counter() - t0

        results.append({
            'board': board,
            'equity': result.equity,
            'se': result.std_error,
            'probs': list(result.mean_probs),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(positions):
            total_elapsed = time.perf_counter() - t_start
            rate = (i + 1) / total_elapsed
            log(f"  {i+1}/{len(positions)} ({rate:.2f} pos/s, "
                f"last={elapsed:.1f}s)")

    total_elapsed = time.perf_counter() - t_start
    log(f"Rollouts complete: {len(positions)} positions in {total_elapsed:.1f}s "
        f"({len(positions)/total_elapsed:.2f} pos/s)")

    # Upload results
    log("Uploading results to S3...")
    result_data = json.dumps({
        'batch_key': batch_key,
        'n_positions': len(positions),
        'total_seconds': total_elapsed,
        'results': results,
    })
    s3.put_object(Bucket=s3_bucket, Key=result_key, Body=result_data)
    log(f"  Uploaded to s3://{s3_bucket}/{result_key}")
    log("Worker done.")


if __name__ == '__main__':
    main()
