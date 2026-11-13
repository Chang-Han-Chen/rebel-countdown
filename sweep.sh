#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ ${1:-} == --dry-run ]]; then
    dry_run=true
    shift
fi
if [[ $# -lt 1 || $1 != /* ]]; then
    echo "Usage: $0 [--dry-run] /absolute/output [pg|rebel|all|sampler-reference] [key=value ...]" >&2
    exit 2
fi

output=${1%/}
shift
case ${1:-all} in
    pg) methods=(pg centered-pg); shift ;;
    rebel) methods=(ordinary centered); shift ;;
    sampler-reference) methods=(sampler-reference centered-sampler-reference); shift ;;
    all) methods=(pg centered-pg ordinary centered); if [[ $# -gt 0 ]]; then shift; fi ;;
    *=*) methods=(pg centered-pg ordinary centered) ;;
    *) echo "Select pg, rebel, all, or sampler-reference." >&2; exit 2 ;;
esac
for override in "$@"; do
    case $override in
        sampler.weight_noise_scale=*|rl.sc=*|rl.rebel=*|rl.rebel_reference=*|log.dir=*|log.run_name=*)
            echo "The sweep sets ${override%%=*} for each condition." >&2
            exit 2
            ;;
        *=*) ;;
        *) echo "Expected a key=value override: $override" >&2; exit 2 ;;
    esac
done

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run=("$repo/run.sh")
if "$dry_run"; then
    run+=(--dry-run)
fi
for method in "${methods[@]}"; do
    for noise in 0 0.02 0.05; do
        echo "$method at sigma=$noise"
        "${run[@]}" "$method" "$output/$method/sigma-$noise" \
            "sampler.weight_noise_scale=$noise" "$@"
    done
done
