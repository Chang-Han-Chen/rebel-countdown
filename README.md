# REBEL on Countdown

Does ordinary REBEL tolerate train–inference mismatch better than policy gradient, and does score centering help REBEL when the mismatch is strong enough to hurt it?

## What to run

1. **Measure sensitivity to mismatch.** Run ordinary PG, score-centered PG, ordinary REBEL, and score-centered REBEL at `σ = 0, 0.02, 0.05` for 27 outer updates with seed 1. Compare each noisy curve against that method's zero-noise control. Ordinary REBEL rose from 10.2% to 40.4% by update 11 in the `σ = 0.02` pilot. The longer run tests whether it keeps learning.
2. **Test whether centering helps REBEL.** If ordinary REBEL learns in the control and loses held-out accuracy under noise, check whether score-centered REBEL retains accuracy at that same noise. Repeat the informative conditions across matched seeds before concluding that either method is more robust.
3. **Test the denominator separately.** Compare `ordinary` with `sampler-reference` at the same noise and seed. Both sample from the perturbed policy $q_t$. Ordinary REBEL fits $\log(p_\theta/p_t)$ using the frozen clean trainer $p_t$. The sampler-reference run fits $\log(p_\theta/q_t)$ using the sampler's logged token probabilities. Score-centered REBEL keeps the trainer reference and centers the token scores using the sampler's top-128 probabilities and a trainer approximation for the tail.

All runs use synchronous FP32 sampling, training, and clean evaluation on Qwen3-0.6B. Each outer update uses 64 prompts × 8 completions, a 512-token sequence cap, and 16 prompts per trainer microbatch. REBEL fits the same rollouts for four AdamW steps at β = 4 and LR `1.2e-7`. PG takes one AdamW step at LR `3e-6`. Both use epsilon `1e-8` and weight decay `0.01`.

## Launch

On Linux with Git and Python 3.13, run `./setup.sh` where GitHub, PyPI, and Hugging Face are reachable. It prepares the pinned [Score Centering trainer](https://github.com/martin-marek/score-centering/tree/fdca22b19793babdf08e33e5ba51537e82cbed2f), JAX 0.11.1 for CUDA 12, and the pinned base model. Set `PYTHON` if Python 3.13 has a different executable name. Every run starts from the base model. The reference ordinary run took about 40 minutes for 12 updates and peaked at 86,087 MiB on one H200. The full batch still needs validation on smaller GPUs.

From your GPU allocation, run the 12 conditions in order. The script runs both PG methods at all three noise levels before the two REBEL methods.

```bash
./sweep.sh /scratch/rebel/seed-1
```

Use `./sweep.sh --dry-run /scratch/rebel/seed-1` to preview the commands. Add `pg` or `rebel` after the output path to run only that phase. For another matched seed, use a fresh output directory and append `all run.seed=2`. The output directories are organized by method and noise.

To schedule a condition individually, call `run.sh` with a fresh absolute output directory.

```bash
./run.sh ordinary /scratch/rebel/ordinary-s02
./run.sh centered /scratch/rebel/centered-s05 sampler.weight_noise_scale=0.05
./run.sh sampler-reference /scratch/rebel/sampler-reference-s02
```

For a quick GPU smoke run, append `stop.steps=1 rl.num_prompts=4 eval.num_prompts=4 rl.rebel_microbatch_prompts=4` to an ordinary or centered launch. To inspect the command, put `--dry-run` before the method.

Share `postax-rl/<method>/logs.jsonl` and `config.json` from each output. Compare `eval_reward` as clean held-out accuracy. The trainer evaluates every two updates and at the last logged step, before applying that step's optimizer updates. REBEL also logs the inner residual MSE.

The upstream MIT license is in `LICENSE.upstream`.
