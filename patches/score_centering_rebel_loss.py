"""REBEL regression losses for the score-centering Countdown trainer."""

import jax
import jax.numpy as jnp

from losses import take_vocab


def rollout_scores(hidden, head, batch, *, old_hidden=None, old_head=None, chunk=32):
    """Sum sampled token scores, optionally centered under a frozen sampler.

    The centered score uses the logged sampler mass on its top-k tokens. It
    allocates the remaining mass over the tail in proportion to the old trainer
    probabilities. Both distributions stay fixed across regression updates.
    """
    if (old_hidden is None) != (old_head is None):
        raise ValueError("Centering requires both old_hidden and old_head.")

    targets = jnp.roll(batch["tokens"], -1, axis=1)
    mask = jnp.roll(batch["mask"], -1, axis=1).at[:, -1].set(False)
    seq_len = targets.shape[1]
    chunk = next(n for n in range(min(chunk, seq_len), 0, -1) if seq_len % n == 0)

    def chunk_scores(start):
        def select(values):
            return jax.lax.dynamic_slice_in_dim(values, start, chunk, axis=1)

        logits = head(select(hidden)).astype(jnp.float32)
        ids = select(targets)[..., None]
        if old_hidden is None:
            scores = take_vocab(jax.nn.log_softmax(logits), ids)[..., 0]
        else:
            # Centering cancels the current log-softmax normalizer.
            head_ids = select(batch["center_ids"])
            sampler_head = jnp.exp(select(batch["center_logprobs"]).astype(jnp.float32))
            head_mass = sampler_head.sum(axis=-1, keepdims=True)
            # A small roundoff overshoot in the logged mass should leave no tail.
            normalizer = jnp.maximum(head_mass, 1.0)
            if head_ids.shape[-1] == logits.shape[-1]:
                normalizer = jnp.maximum(head_mass, jnp.finfo(jnp.float32).tiny)
            mean = (sampler_head / normalizer * take_vocab(logits, head_ids)).sum(axis=-1)

            if head_ids.shape[-1] < logits.shape[-1]:
                old_logits = jax.lax.stop_gradient(old_head(select(old_hidden)).astype(jnp.float32))
                batch_ids = jnp.arange(head_ids.shape[0])[:, None, None]
                positions = jnp.arange(head_ids.shape[1])[None, :, None]
                # Excluding the head before softmax avoids subtracting nearly equal masses.
                tail_logits = old_logits.at[batch_ids, positions, head_ids].set(
                    -jnp.inf, out_sharding=jax.typeof(old_logits).sharding.spec
                )
                tail_probs = jax.nn.softmax(tail_logits, axis=-1)
                tail_mass = jnp.maximum(1.0 - head_mass[..., 0], 0.0)
                mean += tail_mass * (tail_probs * logits).sum(axis=-1)

            scores = take_vocab(logits, ids)[..., 0] - mean

        return jnp.where(select(mask), scores, 0.0).sum(axis=1)

    return jax.lax.map(jax.checkpoint(chunk_scores), jnp.arange(0, seq_len, chunk)).sum(axis=0)


def rebel_loss(scores, old_scores, rewards, group_size, beta):
    """Compute the within-prompt pairwise REBEL loss via centered residuals."""
    residual = scores - jax.lax.stop_gradient(old_scores) - beta * rewards
    residual = residual.reshape(-1, group_size)
    residual = residual - residual.mean(axis=1, keepdims=True)
    return 0.5 * jnp.square(residual).mean()
