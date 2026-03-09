# Findings & Decisions

## Requirements
- Audit the real BrokerTuner implementation before rewriting paper pseudocode.
- Final answer must have sections A/B/C/D exactly as requested.
- Must explicitly state mapping `APN-DDPG = EnhancedDDPG`.
- Final algorithm bodies must use only `APN-DDPG`; `EnhancedDDPG` may appear only outside algorithms.
- Four modules must be explicit: environment/broker actuation, APN-DDPG network structure, experience sampling/replay, training update.
- Attention, PER, and N-step must be explicit in the final pseudocode, even if source enables them via config switches.
- Need code evidence with file path plus class/function names.
- Need to distinguish always-on mechanisms from supported-but-configurable mechanisms.

## Research Findings
- `using-superpowers` and `planning-with-files` skills apply here; planning files were required before deeper exploration.
- The task requires reading at least the user-specified source files plus tests/docs as secondary evidence.
- `tuner/utils.py` shows the project-level mapping clearly: training always instantiates `EnhancedDDPG(policy="MlpPolicy", ...)`, with `FeatureWiseAttentionExtractor` injected only when `use_attention=True`, and `PrioritizedNStepReplayBuffer` injected only when `use_per` or `use_nstep` is enabled.
- `environment/config.py` and `environment/utils.py` confirm a 10-dimensional state and an 11-dimensional action space.
- `environment/knobs.py` confirms the 11 action dimensions map to Mosquitto knobs: `max_inflight_messages`, `max_inflight_bytes`, `max_queued_messages`, `max_queued_bytes`, `queue_qos0_messages`, `memory_limit`, `persistence`, `autosave_interval`, `set_tcp_nodelay`, `max_packet_size`, `message_size_limit`.
- `environment/knobs.py::apply_knobs` writes an independent `broker_tuner.conf`, stops existing Mosquitto processes, starts `mosquitto -c <config> -d`, and updates `MOSQUITTO_PID`.
- `environment/broker.py::step` performs the full broker actuation chain: `decode_action` -> `apply_knobs` -> wait for broker ready -> restart workload if available -> wait workload warmup -> wait for `$SYS` publish -> sample metrics -> build next state -> compute reward/constraint -> return `(next_state, reward, terminated, truncated, info)`.
- `environment/broker.py::_compute_reward` combines throughput improvement and latency reduction; when `constraint_mode="lagrangian_hinge"` it also computes `latency_limit_ms`, `latency_violation_ms`, `constraint_lambda`, `constraint_penalty`, and `unsafe`.
- `environment/broker.py::_make_failure_transition` converts execution failures into learnable failure transitions, with `done = terminated or truncated` semantics preserved in `info`.
- `model/attention_extractor.py` implements feature-wise gating on vector observations, not temporal attention. The gated state is forwarded to actor/critic through SB3 `features_extractor_class`.
- `model/prioritized_nstep_replay_buffer.py` confirms the replay path is collector/environment-driven, not actor-direct. N-step aggregation happens inside `add()` before storage when enabled; `sample()` returns `observations`, `actions`, `next_observations`, `dones`, `rewards`, `indices`, `weights`, `n_steps`, and `constraint_ratios`.
- `model/enhanced_ddpg.py::train` applies `gradient_steps * utd_ratio`, uses IS weights and optional constraint weights in critic loss, supports `mse`/`huber`, supports grad clipping and `target_q_clip`, delays actor/target updates with `policy_delay`, and writes TD-error-based priorities back to replay.
- Tests confirm the intended data structures: `tests/test_attention_extractor.py`, `tests/test_replay_buffer_per_nstep.py`, `tests/test_env_reward.py`, and `tests/test_evaluate_metrics.py`.
- The local Python 3 environment has `stable_baselines3==2.4.1` installed under `/home/qincai/.local/lib/python3.8/site-packages/stable_baselines3`.
- Local SB3 source confirms the collector write path: `OffPolicyAlgorithm.collect_rollouts() -> _store_transition() -> replay_buffer.add(...)`.
- Local SB3 source also confirms that the collector sends `action` to the environment but stores `buffer_action` in replay; for continuous actions this `buffer_action` is the action rescaled to `[-1, 1]`, while the environment receives the unscaled action in its declared action space `[0, 1]^{11}`.
- Local SB3 source confirms `DDPG` is TD3-derived and, unless overridden, forces `policy_delay=1`, `target_noise_clip=0.0`, `target_policy_noise=0.1`, and `n_critics=1`. Therefore the current BrokerTuner path does not instantiate twin critics by default, and target smoothing noise is effectively disabled by zero clip.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Treat source code as authoritative over docs | Explicit user requirement |
| Use tests/docs only to confirm behavior and naming context | User allows them only as secondary references |
| Use official SB3 source/docs only for non-vendored collector and DDPG internals | Those details are outside the repository but still part of the real execution path |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Initial `python` interpreter lookup failed to import `stable_baselines3` | Switched to `python3`, confirmed local installation, and used local package source for verification |

## Resources
- Requested source files under `tuner/`, `environment/`, and `model/`
- Supporting docs under `docs/`
- Supporting tests under `tests/`
- Local SB3 off-policy source: `/home/qincai/.local/lib/python3.8/site-packages/stable_baselines3/common/off_policy_algorithm.py`
- Local SB3 DDPG source: `/home/qincai/.local/lib/python3.8/site-packages/stable_baselines3/ddpg/ddpg.py`
- Local SB3 TD3 policy source: `/home/qincai/.local/lib/python3.8/site-packages/stable_baselines3/td3/policies.py`

## Visual/Browser Findings
- No browser or image artifacts used.
