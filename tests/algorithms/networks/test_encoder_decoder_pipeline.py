"""End-to-end tokenizer -> encoder -> decoder wiring, without going through an algorithm class.

Mirrors what ``BaseDiffusionAgent._encode()`` + ``self.decoder(...)`` run: the algorithm tokenizes,
normalizes, and hands the encoder a ``{"proprio", "task"[, "goal_task"]}`` tree.
"""

import hydra_zen
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from policy.algorithms.networks.encoder import ConditioningEncoder
from policy.algorithms.tokenizers import ObjectTokenizer, StateTokenizer
from policy.utils import pop_leaf_key

PROPRIO_DIM = 18
POSE_KEYS = ("obj_0_pose", "obj_1_pose", "tcp_pose")


def _compose_decoder(config_name, **overrides):
    with initialize_config_module(config_module="policy.configs", version_base="1.2"):
        cfg = compose(config_name=f"algorithm/decoder/{config_name}")
    net_cfg = cfg.algorithm.decoder
    OmegaConf.set_struct(net_cfg, False)
    for key, value in overrides.items():
        net_cfg[key] = value
    return net_cfg


def _obs_tree(batch_size, obs_horizon, num_objects=2, time_axis=True):
    shape = (batch_size, obs_horizon) if time_axis else (batch_size,)
    tree = {"proprio": torch.randn(*shape, PROPRIO_DIM), "tcp_pose": torch.randn(*shape, 7)}
    for i in range(num_objects):
        tree[f"obj_{i}_pose"] = torch.randn(*shape, 7)
        role = [0.0, 0.0, 1.0] if i > 1 else [float(i == 0), float(i == 1), 0.0]
        tree[f"obj_{i}_role"] = torch.tensor(role).expand(*shape, 3).clone()
    return tree


def _tokenize(tokenizer, obs, goal=None, relative_goal=True):
    """The algorithm-side half of the pipeline, inlined so this test stays network-scoped."""
    proprio, obs_task = pop_leaf_key(obs, "proprio", PROPRIO_DIM)
    if goal is None:
        return {"proprio": proprio, "task": tokenizer.tokenize(obs_task, None)}

    _, goal_task = pop_leaf_key(goal, "proprio", PROPRIO_DIM)
    if not relative_goal:
        return {
            "proprio": proprio,
            "task": tokenizer.tokenize(obs_task, None),
            "goal_task": tokenizer.tokenize(None, goal_task),
        }

    goal_task = {k: v.unsqueeze(1) for k, v in goal_task.items()}
    return {"proprio": proprio, "task": tokenizer.tokenize(obs_task, goal_task)}


def _assert_gradients_flow(*modules):
    for module in modules:
        for name, p in module.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name} got no gradient"
                assert torch.isfinite(p.grad).all(), f"{name} got a non-finite gradient"


def test_film_pipeline_instantiates_and_runs():
    """Plain, unconditioned FiLM pipeline: no goal, default (flatten, no-embedder) encoding."""
    batch_size, horizon, act_dim = 128, 16, 8

    obs = _obs_tree(batch_size, horizon)
    task_dim = {k: v.shape[-1] for k, v in obs.items() if k != "proprio"}
    tokenizer = StateTokenizer(task_dim=task_dim, relative_goal=False)

    encoder = ConditioningEncoder(
        proprio_dim=PROPRIO_DIM,
        token_dim=tokenizer.output_dim,
        tokens_per_step=tokenizer.tokens_per_step,
        goal_conditioned=False,
    )
    decoder_cfg = _compose_decoder("film_decoder1d", act_dim=act_dim, obs_horizon=horizon)
    decoder = hydra_zen.instantiate(decoder_cfg, cond_dims=encoder.cond_dims)

    sample = torch.randn(batch_size, horizon, act_dim)
    timestep = torch.randint(0, 100, (batch_size,))

    output = decoder(sample, timestep, external_cond=encoder(_tokenize(tokenizer, obs)))
    assert output.shape == sample.shape

    output.sum().backward()
    _assert_gradients_flow(encoder, decoder)


def test_film_pipeline_goal_conditioned_absolute_mode_instantiates_and_runs():
    """goal_conditioned=True, relative_goal=False: a separate 'goal' entry alongside 'obs'."""
    batch_size, horizon, act_dim = 8, 4, 4

    obs = _obs_tree(batch_size, horizon)
    goal = _obs_tree(batch_size, horizon, time_axis=False)
    task_dim = {k: v.shape[-1] for k, v in obs.items() if k != "proprio"}
    tokenizer = StateTokenizer(task_dim=task_dim, relative_goal=False)

    encoder = ConditioningEncoder(
        proprio_dim=PROPRIO_DIM,
        token_dim=tokenizer.output_dim,
        tokens_per_step=tokenizer.tokens_per_step,
        goal_conditioned=True,
    )
    decoder_cfg = _compose_decoder("film_decoder1d", act_dim=act_dim, obs_horizon=horizon)
    decoder = hydra_zen.instantiate(decoder_cfg, cond_dims=encoder.cond_dims)

    sample = torch.randn(batch_size, horizon, act_dim)
    timestep = torch.randint(0, 100, (batch_size,))
    tokens = _tokenize(tokenizer, obs, goal, relative_goal=False)
    assert "goal_task" in tokens

    output = decoder(sample, timestep, external_cond=encoder(tokens))
    assert output.shape == sample.shape

    output.sum().backward()
    _assert_gradients_flow(encoder, decoder)


def test_cross_attention_pipeline_instantiates_and_runs():
    """Cross-attention with a real ObjectTokenizer + SelfAttention embedder: the per-object token
    sequence is attended over instead of flattened into FiLM."""
    batch_size, pred_horizon, obs_horizon, act_dim = 8, 16, 2, 4
    context_dim = 8

    tokenizer = ObjectTokenizer(relative_goal=True)
    encoder = ConditioningEncoder(
        proprio_dim=PROPRIO_DIM,
        token_dim=tokenizer.output_dim,
        tokens_per_step=None,
        goal_conditioned=True,
        relative_goal=True,
        decoder_type="cross_attention",
        embedder={
            "_target_": "policy.algorithms.networks.encoder.embedders.self_attention.SelfAttention",
            "output_dim": context_dim,
            "obs_horizon": obs_horizon,
            "num_heads": 2,
        },
    )
    decoder_cfg = _compose_decoder(
        "cross_attention_decoder1d", act_dim=act_dim, obs_horizon=obs_horizon
    )
    decoder = hydra_zen.instantiate(decoder_cfg, cond_dims=encoder.cond_dims)

    sample = torch.randn(batch_size, pred_horizon, act_dim)
    timestep = torch.randint(0, 100, (batch_size,))

    obs = _obs_tree(batch_size, obs_horizon)
    goal = _obs_tree(batch_size, obs_horizon, time_axis=False)
    output = decoder(sample, timestep, external_cond=encoder(_tokenize(tokenizer, obs, goal)))
    assert output.shape == sample.shape

    output.sum().backward()
    _assert_gradients_flow(encoder, decoder)

    # The same instantiated pipeline handles a different number of objects (2 clutter added).
    obs_c = _obs_tree(batch_size, obs_horizon, num_objects=4)
    goal_c = _obs_tree(batch_size, obs_horizon, num_objects=4, time_axis=False)
    encoded = encoder(_tokenize(tokenizer, obs_c, goal_c))
    assert encoded["context"].shape == (batch_size, obs_horizon * 4, context_dim)
    assert decoder(sample, timestep, external_cond=encoded).shape == sample.shape


def test_film_pipeline_with_object_tokenizer_and_attention_pooling():
    """ObjectTokenizer + AttentionPooling(mode="objects"): K object tokens compress to 1 vector per
    timestep, which is what allows a variable object count under FiLM."""
    batch_size, pred_horizon, obs_horizon, act_dim = 8, 16, 2, 4
    embed_dim = 16

    tokenizer = ObjectTokenizer(relative_goal=True)
    encoder = ConditioningEncoder(
        proprio_dim=PROPRIO_DIM,
        token_dim=tokenizer.output_dim,
        tokens_per_step=None,
        goal_conditioned=True,
        relative_goal=True,
        embedder={
            "_target_": "policy.algorithms.networks.encoder.embedders.mlp.MLP",
            "output_dim": embed_dim,
            "hidden_dims": [32],
        },
        pooling={
            "_target_": "policy.algorithms.networks.encoder.pooling.AttentionPooling",
            "mode": "objects",
            "num_heads": 2,
        },
    )
    decoder_cfg = _compose_decoder("film_decoder1d", act_dim=act_dim, obs_horizon=obs_horizon)
    decoder = hydra_zen.instantiate(decoder_cfg, cond_dims=encoder.cond_dims)

    sample = torch.randn(batch_size, pred_horizon, act_dim)
    timestep = torch.randint(0, 100, (batch_size,))

    obs = _obs_tree(batch_size, obs_horizon)
    goal = _obs_tree(batch_size, obs_horizon, time_axis=False)
    output = decoder(sample, timestep, external_cond=encoder(_tokenize(tokenizer, obs, goal)))
    assert output.shape == sample.shape

    # FiLM conditioning shape is unchanged by extra objects, because of the object pooling.
    obs_c = _obs_tree(batch_size, obs_horizon, num_objects=5)
    goal_c = _obs_tree(batch_size, obs_horizon, num_objects=5, time_axis=False)
    encoded = encoder(_tokenize(tokenizer, obs_c, goal_c))
    assert encoded["obs"]["task"].shape == (batch_size, obs_horizon, embed_dim)
    assert decoder(sample, timestep, external_cond=encoded).shape == sample.shape


def test_film_pipeline_rejects_variable_objects_without_pooling():
    """FiLM without object pooling cannot process a dynamic object-token count."""
    tokenizer = ObjectTokenizer(relative_goal=True)
    with pytest.raises(ValueError, match="pooling across objects"):
        ConditioningEncoder(
            proprio_dim=PROPRIO_DIM,
            token_dim=tokenizer.output_dim,
            tokens_per_step=None,
            goal_conditioned=True,
            relative_goal=True,
        )
