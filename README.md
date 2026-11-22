# VLA instruction experiments

This repository contains local copies of the [VLA evaluation harness](https://github.com/allenai/vla-evaluation-harness)
and [Embodied Red-Teaming](https://github.com/Improbable-AI/embodied-red-teaming),
plus a LIBERO-Goal pilot and model server entry points.

The upstream leaderboard, published raw result archives, and bundled upstream
GitHub automation have been removed. The current checkout uses ordinary Git files.

## Setup

Run these commands from the repository root. Use Python 3.11 and install `uv`.
Charliecloud 0.45.1 or newer must be on `PATH`, along with `nvidia-container-cli`
for GPU rendering. The host must allow unprivileged user namespaces
(`unshare -Ur true`). Charliecloud can be installed with `pixi global install charliecloud`.

```bash
uv venv --python 3.11 venv
cp .env.example .env
source ./run-vla.sh
uv pip install --link-mode=copy -r requirements.txt
```

Enter the API key and endpoint in `.env` for actual ERT generation. The file is
ignored by Git. The null instruction generator works without API credentials.

`run-vla.sh` sets all environment paths, the shared Python environment,
workspace download caches, local `/tmp` container exports, and
`CUDA_VISIBLE_DEVICES=2,3`. Adjust that GPU selection for your machine.
Source it in every terminal used for the experiment.

## A. Model server

In the first terminal, start either server on port 8000.

```bash
source ./run-vla.sh
python "$TESTING_DIR/null_policy.py"
```

For SmolVLA, use this command instead of the null server command.

```bash
python "$TESTING_DIR/smolvla_server.py"
```

SmolVLA defaults to `lerobot/smolvla_libero` revision `a24f1f38`, one action
per inference, and logical `cuda:0`. The model downloads into the workspace
Hugging Face cache. These defaults follow the harness's SmolVLA configuration.
The paper's Table III uses a different snapshot and executes eight actions
per inference, so the default pilot settings require adjustment for that reproduction.

## B. Instruction generation

In a second terminal, generate two copies of the original task instruction.

```bash
source ./run-vla.sh
python "$TESTING_DIR/ert_instructions.py" --null --count 2
```

Omit `--null` to call ERT. The generator saves `instructions.json` and `eval.yaml`
under `$ERT_OUTPUT_DIR` (`results/ert`). Each new generation replaces these files.

## C. Evaluation

Keep the model server running and check readiness before launching the simulator.

```bash
curl -fsS http://127.0.0.1:8000/health
vla-eval run --runtime charliecloud --render gpu --dev --yes \
  --config "$ERT_OUTPUT_DIR/eval.yaml"
```

The first run downloads the simulator image. Each generated instruction runs
from the same first initial state of the first LIBERO-Goal task, with a
300-step limit. Results go under `results/ert/evaluation`. Episode metrics
include designated-task success and `max_eef_displacement_m`, measured from
the end-effector position after settling.

For the original task instruction without generating an instruction set, run

```bash
vla-eval run --runtime charliecloud --render gpu --dev --yes \
  --config "$VLA_EVAL_DIR/configs/benchmarks/libero/goal_instruction_pilot.yaml"
```

The reference paper is `2610.41609v1.pdf`. This pilot measures displacement
and designated-task success. Reproducing its full non-start and any-task
completion metrics also requires contact detection and all-task goal checks.
