#!/usr/bin/env bash


script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1

if [[ -f "$script_dir/.env" ]]; then
    source "$script_dir/.env" || { return 1 2>/dev/null || exit 1; }
fi

export VLA_EVAL_DIR="$script_dir/vla-evaluation-harness"
export ERT_DIR="$script_dir/embodied-red-teaming"
export TESTING_DIR="$script_dir/testing"
export ERT_OUTPUT_DIR="$script_dir/results/ert"
export VIRTUAL_ENV="$script_dir/venv"
export UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV"
export PATH="$VIRTUAL_ENV/bin:$script_dir/.pixi/bin:$PATH"
export PYTHONPATH="$ERT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CH_IMAGE_STORAGE="$script_dir/.charliecloud"
export UV_CACHE_DIR="$script_dir/.cache/uv"
export HF_HOME="$script_dir/.cache/huggingface"
export VLA_EVAL_HOME="/tmp/vla-eval-$UID"
if ! mkdir -m 700 -- "$VLA_EVAL_HOME" 2>/dev/null &&
    [[ ! -d $VLA_EVAL_HOME || ! -O $VLA_EVAL_HOME || -L $VLA_EVAL_HOME ]]; then
    printf 'Cannot use local cache directory: %s\n' "$VLA_EVAL_HOME" >&2
    return 1 2>/dev/null || exit 1
fi
chmod 700 -- "$VLA_EVAL_HOME" || { return 1 2>/dev/null || exit 1; }
export TMPDIR="$VLA_EVAL_HOME/tmp"
mkdir -p -- "$TMPDIR" || { return 1 2>/dev/null || exit 1; }
export CUDA_VISIBLE_DEVICES=2,3
