#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
upstream_url=https://github.com/martin-marek/score-centering.git
upstream_commit=fdca22b19793babdf08e33e5ba51537e82cbed2f
model_revision=c1899de289a04d12100db370d81485cdf75e47ca
source_dir="$repo_dir/.upstream"
patch="$repo_dir/patches/score_centering_rebel.patch"
loss="$repo_dir/patches/score_centering_rebel_loss.py"
venv="$repo_dir/.venv"
weights_dir=${WEIGHTS_DIR:-"$repo_dir/.weights"}

if [[ $# -gt 1 || ( $# -eq 1 && $1 != --source-only ) ]]; then
    echo "Usage: $0 [--source-only]" >&2
    exit 2
fi

if [[ -d "$source_dir" ]]; then
    if [[ $(git -C "$source_dir" rev-parse HEAD) != "$upstream_commit" ]] ||
        ! git -C "$source_dir" apply --reverse --check "$patch" ||
        ! cmp -s "$loss" "$source_dir/score_centering_rebel_loss.py"; then
        echo "Existing .upstream differs from the pinned source. Move it aside and rerun setup." >&2
        exit 1
    fi
else
    staging=$(mktemp -d "$repo_dir/.upstream.tmp.XXXXXXXX")
    trap 'rm -rf -- "$staging"' EXIT
    git -C "$staging" init --quiet
    git -C "$staging" fetch --quiet --depth=1 "$upstream_url" "$upstream_commit"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$staging" checkout --quiet --detach FETCH_HEAD
    git -C "$staging" apply "$patch"
    cp -- "$loss" "$staging/score_centering_rebel_loss.py"
    mv -- "$staging" "$source_dir"
fi
echo "Prepared Score Centering at $upstream_commit."

if [[ ${1:-} == --source-only ]]; then
    exit 0
fi
if [[ $(uname -s) != Linux ]]; then
    echo "CUDA setup requires Linux. Use --source-only to prepare code on this machine." >&2
    exit 1
fi
if [[ $weights_dir != /* ]]; then
    echo "WEIGHTS_DIR must be an absolute path." >&2
    exit 2
fi

"${PYTHON:-python3.13}" -m venv "$venv"
"$venv/bin/python" -m pip install -r "$source_dir/requirements.txt" 'jax[cuda12]==0.11.1'
"$venv/bin/python" - "$weights_dir/Qwen3-0.6B" "$model_revision" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

model_dir = Path(sys.argv[1])
revision = sys.argv[2]
snapshot_download(repo_id="Qwen/Qwen3-0.6B", revision=revision, local_dir=model_dir)
(model_dir / ".pinned-revision").write_text(revision + "\n")
print(f"Prepared Qwen/Qwen3-0.6B at {model_dir}.")
PY
