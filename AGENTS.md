# Repository Guidelines

## Project Structure & Module Organization
- `environment/`: Gym environment, broker sampling, and configuration knobs.
- `model/`: DDPG policy and network definitions.
- `tuner/`: Training and evaluation entry points (`train.py`, `evaluate.py`).
- `server/`: HTTP control server and startup script.
- `script/`: Helper scripts (training runner, workload checks).
- `docs/` and `misc/`: Additional guides, troubleshooting, and experiments.
- `checkpoints/`: Generated model artifacts and logs (not source).

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt` plus `pip install stable-baselines3[extra] gym paho-mqtt`.
- Set runtime env:
  - `export PYTHONPATH=$(pwd):$PYTHONPATH`
  - `export MOSQUITTO_PID=$(pgrep mosquitto)`
- Training (quick path): `./start_training.sh 100000 ./checkpoints 10000`.
- Training (full control): `./script/run_train.sh --enable-workload --total-timesteps 100000 --save-dir ./checkpoints --save-freq 10000`.
- Evaluate: `python -m tuner.evaluate --model-path ./checkpoints/ddpg_mosquitto_final.zip --n-episodes 10`.
- API server: `bash server/start_server.sh` (listens on `0.0.0.0:8080`).
- Workload smoke test: `python3 script/test_workload.py --duration 10`.

## Coding Style & Naming Conventions
- Python 3.8+, 4-space indentation, PEP 8 conventions.
- Use `snake_case` for functions/variables, `CapWords` for classes, and keep config structures in `environment/config.py`.
- Shell scripts are `bash`; keep flags explicit and document new ones in `START_TRAINING.md`.

## Testing Guidelines
- No formal unit-test framework configured.
- Validation is manual via workload runs and short training/evaluation cycles.
- When changing broker interaction, verify Mosquitto is running and `$SYS/#` metrics are available.

## Commit & Pull Request Guidelines
- Git history uses short, concise summaries (e.g., "modify"); no enforced convention.
- Keep commits focused and describe the subsystem if helpful (e.g., "tuner: add reward term").
- PRs should include a brief rationale, commands run, and any Mosquitto config or permission changes.

## Environment & Configuration Tips
- Optional env vars: `MOSQUITTO_TUNER_CONFIG` for config path, `EMQTT_BENCH_PATH` for workload tool.
- Writes to `/etc/mosquitto/conf.d` and service restarts may require `sudo`.
