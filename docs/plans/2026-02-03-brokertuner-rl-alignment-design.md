# BrokerTuner RL Alignment Design (CDBTuner Parity)

**Date:** 2026-02-03

## Goal
Align BrokerTuner’s reward shaping, replay buffer behavior, policy defaults, latency probing, logging, and docs with CDBTuner while keeping SB3 DDPG as the training engine and preserving absolute action mode/knob ranges.

## Architecture Overview
- **Training Engine:** Stable-Baselines3 DDPG remains the core trainer.
- **PER Integration:** Introduce a lightweight `PrioritizedDDPG` subclass that overrides `train()` only to compute per-sample TD-errors and update priorities in a custom replay buffer. This preserves SB3’s training loop and avoids reimplementing CDBTuner’s loop.
- **Environment:** `MosquittoBrokerEnv` will compute reward from raw throughput/latency metrics (msg/s, ms) with CDBTuner formula, track default/last/best metrics, and expose reward components in `info`.
- **Latency Probing:** `WorkloadManager` will run a background MQTT latency probe using `paho-mqtt`, providing rolling p50/p95 stats to the environment.

## Reward & Metrics Flow
1. **Baseline:** At `reset()`, sample baseline metrics from current throughput/latency and store `default_external_metrics`.
2. **Step Metrics:** Each step gathers raw throughput (from $SYS) and raw latency (probe p50/p95). If `reward_use_best_as_last` is true, use best metrics as `last_external_metrics`; otherwise use previous step.
3. **Reward:** Apply CDBTuner formula for tps and latency deltas with guardrails (eps, invalid metric penalty, optional early terminate). Combine with 0.4/0.6 weights; apply positive reward multiplier.
4. **Info/Logging:** Record `reward_tps`, `reward_latency`, `reward_raw`, `reward_final`, `default_tps/lat`, `best_tps/lat`, `latency_p50/p95`.

## Prioritized Replay Buffer
- New module: `model/prioritized_replay_buffer.py` using SumTree (alpha/beta/beta_increment) compatible with SB3 replay buffer API.
- `PrioritizedDDPG.train()` computes TD-error per sample and updates priorities by index.

## Policy & Hyperparameters
- Keep `CustomDDPGPolicy` in `model/ddpg.py` and use it in DDPG creation.
- Defaults set to CDBTuner values: `tau=1e-5`, `actor_lr=1e-5`, `critic_lr=1e-5`, `gamma=0.9`, `batch_size=16`, `replay_buffer_size=100000`.

## CLI & Logging
- Add CLI flags for PER (`--replay-buffer-size`, `--replay-alpha`, `--replay-beta`, `--replay-beta-increment`).
- Extend `ActionThroughputLoggerWrapper` CSV to include latency and reward components plus baseline/best metrics.
- Log PER class name and priority stats (min/max/mean) when available.

## Error Handling
- Invalid metrics (NaN/Inf/<=0) yield `reward_invalid_penalty` and optional early terminate after configurable patience.
- Latency probe failures must not crash training; fall back to last known or defaults and emit a warning.

## Verification
- Dry-run: `python -m tuner.train --total-timesteps 10 --enable-workload`.
- Check CSV includes latency/reward columns.
- Confirm reward and PER update logs behave as expected.
