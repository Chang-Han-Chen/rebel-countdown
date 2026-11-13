#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ ${1:-} == --dry-run ]]; then
    dry_run=true
    shift
fi
if [[ $# -lt 2 || $2 != /* ]]; then
    echo "Usage: $0 [--dry-run] {ordinary|centered|pg|centered-pg|sampler-reference|centered-sampler-reference} /absolute/output [key=value ...]" >&2
    exit 2
fi

variant=$1
output=$2
shift 2
case $variant in
    ordinary) method=(rl.rebel=true rl.sc=false) ;;
    centered) method=(rl.rebel=true rl.sc=true) ;;
    sampler-reference) method=(rl.rebel=true rl.sc=false rl.rebel_reference=sampler) ;;
    centered-sampler-reference) method=(rl.rebel=true rl.sc=true rl.rebel_reference=sampler) ;;
    pg) method=(rl.sc=false opt.lr=3e-6) ;;
    centered-pg) method=(rl.sc=true opt.lr=3e-6) ;;
    *) echo "Unknown method: $variant" >&2; exit 2 ;;
esac

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
weights=${WEIGHTS_DIR:-"$repo/.weights"}
if [[ $weights != /* ]]; then
    echo "WEIGHTS_DIR must be an absolute path." >&2
    exit 2
fi

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

command=(
    "$repo/.venv/bin/python" -u train_rl.py "model.source=$weights/Qwen3-0.6B"
    env.id=countdown env.prompt_format=chat
    rl.num_prompts=64 rl.group_size=8 rl.seq_len=512 eval.num_prompts=64 eval.every_steps=2
    stop.steps=100 stop.collapse=0 run.seed=1
    opt.optimizer=adamw opt.lr=1.2e-7 opt.eps=1e-8 opt.weight_decay=0.01
    sampler.staleness=0 sampler.vocab_logprobs=128 sampler.weight_noise_scale=0.02
    dtypes.sampler=float32 dtypes.eval=float32 rl.is.level=null
    rl.rebel_inner_steps=4 rl.rebel_beta=4.0 rl.rebel_dtype=float32 rl.rebel_microbatch_prompts=16
    rl.pg_dtype=float32 rl.pg_head_chunk=32 rl.pg_microbatch_prompts=16
    log.wandb_mode=disabled "log.dir=$output" "log.run_name=$variant"
    "${method[@]}" "$@"
)

if "$dry_run"; then
    printf 'cd %q\n' "$repo/.upstream"
    printf 'JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=highest'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

if [[ ! -x "$repo/.venv/bin/python" || ! -s "$weights/Qwen3-0.6B/.pinned-revision" ]]; then
    echo "Run ./setup.sh with the same WEIGHTS_DIR first." >&2
    exit 1
fi
if [[ -s "$output/postax-rl/$variant/logs.jsonl" ]]; then
    echo "This run already has metrics. Use a fresh output directory." >&2
    exit 1
fi

cd -- "$repo/.upstream"
exec "${command[@]}"
