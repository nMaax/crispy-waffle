"""End-to-end tokenizer -> encoder -> decoder wiring, without going through an algorithm class.

Mirrors what ``BaseDiffusionAgent._encode()`` + ``self.decoder(...)`` run: the algorithm tokenizes,
normalizes, and hands the encoder an ``{"obs"[, "goal"]}`` tree of ``{"proprio", "task"}``.
"""

import hydra_zen
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from policy.algorithms.networks.encoder import ConditioningEncoder
from policy.algorithms.tokenizers import GraphTokenizer, ObjectTokenizer, StateTokenizer
from policy.utils import pop_leaf_key

PROPRIO_DIM = 18
ROLE_DIM = 4
ROLES = ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0])
CLUTTER_ROLE = [0.0, 0.0, 0.0, 1.0]


def _task_dim(num_objects):
    """Canonical task spec for a pool of ``num_objects`` scene objects plus the TCP at slot 0."""
    slots = num_objects + 1
    return {"obj_pose": (slots, 7), "obj_role": (slots, 4), "obj_valid": (slots,)}


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
    slots = num_objects + 1  # slot 0 is the TCP
    roles = [ROLES[i] if i < len(ROLES) else CLUTTER_ROLE for i in range(slots)]
    return {
        "proprio": torch.randn(*shape, PROPRIO_DIM),
        "obj_pose": torch.randn(*shape, slots, 7),
        "obj_role": torch.tensor(roles).expand(*shape, slots, 4).clone(),
        "obj_valid": torch.ones(*shape, slots),
    }


def _tokenize(tokenizer, obs, goal=None, relative_goal=True):
    """The algorithm-side half of the pipeline, inlined so this test stays network-scoped."""
    proprio, obs_task = pop_leaf_key(obs, "proprio", PROPRIO_DIM)
    if goal is None:
        return {"obs": {"proprio": proprio, "task": tokenizer.tokenize(obs_task, None)}}

    goal_proprio, goal_task = pop_leaf_key(goal, "proprio", PROPRIO_DIM)
    if not relative_goal:
        return {
            "obs": {"proprio": proprio, "task": tokenizer.tokenize(obs_task, None)},
            "goal": {
                "proprio": goal_proprio,
                "task": tokenizer.tokenize(None, goal_task),
            },
        }

    goal_task = {k: v.unsqueeze(1) for k, v in goal_task.items()}  # add the goal's time axis
    return {"obs": {"proprio": proprio, "task": tokenizer.tokenize(obs_task, goal_task)}}


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
    tokenizer = StateTokenizer(task_dim=_task_dim(2), relative_goal=False)

    encoder = ConditioningEncoder(
        proprio_dim=PROPRIO_DIM,
        token_dim=tokenizer.output_dim,
        tokens_per_step=tokenizer.tokens_per_step,
        relative_goal=False,
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
    tokenizer = StateTokenizer(task_dim=_task_dim(2), relative_goal=False)

    encoder = ConditioningEncoder(
        proprio_dim=PROPRIO_DIM,
        token_dim=tokenizer.output_dim,
        tokens_per_step=tokenizer.tokens_per_step,
        relative_goal=False,
        goal_conditioned=True,
    )
    decoder_cfg = _compose_decoder("film_decoder1d", act_dim=act_dim, obs_horizon=horizon)
    decoder = hydra_zen.instantiate(decoder_cfg, cond_dims=encoder.cond_dims)

    sample = torch.randn(batch_size, horizon, act_dim)
    timestep = torch.randint(0, 100, (batch_size,))
    tokens = _tokenize(tokenizer, obs, goal, relative_goal=False)
    assert "goal" in tokens

    output = decoder(sample, timestep, external_cond=encoder(tokens))
    assert output.shape == sample.shape

    output.sum().backward()
    _assert_gradients_flow(encoder, decoder)


def test_cross_attention_pipeline_instantiates_and_runs():
    """Cross-attention with a real ObjectTokenizer + SelfAttention embedder: the per-object token
    sequence is attended over instead of flattened into FiLM."""
    batch_size, pred_horizon, obs_horizon, act_dim = 8, 16, 2, 4
    context_dim = 8

    tokenizer = ObjectTokenizer(_task_dim(2), relative_goal=True)
    encoder = ConditioningEncoder(
        proprio_dim=PROPRIO_DIM,
        token_dim=tokenizer.output_dim,
        tokens_per_step=tokenizer.tokens_per_step,
        goal_conditioned=True,
        relative_goal=True,
        decoder_type="cross_attention",
        embedder={
            "_target_": "policy.algorithms.networks.encoder.embedders.self_attention.SelfAttention",
            "output_dim": context_dim,
            "obs_horizon": obs_horizon,
            "num_heads": 2,
        },
        role_dim=ROLE_DIM,
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

    # Cross-attention fixes only the context width, never the sequence length, so the same
    # instantiated pipeline absorbs a larger pool (2 clutter objects added). This is what lets a
    # variable object count survive without padding the decoder's conditioning.
    tokenizer_c = ObjectTokenizer(_task_dim(4), relative_goal=True)
    assert tokenizer_c.output_dim == tokenizer.output_dim
    obs_c = _obs_tree(batch_size, obs_horizon, num_objects=4)
    goal_c = _obs_tree(batch_size, obs_horizon, num_objects=4, time_axis=False)
    encoded = encoder(_tokenize(tokenizer_c, obs_c, goal_c))
    assert encoded["context"].shape == (batch_size, obs_horizon * 5, context_dim)
    assert decoder(sample, timestep, external_cond=encoded).shape == sample.shape


def test_film_pipeline_with_object_tokenizer_and_attention_pooling():
    """ObjectTokenizer + AttentionPooling(mode="objects"): K object tokens compress to 1 vector per
    timestep, which is what allows a variable object count under FiLM."""
    batch_size, pred_horizon, obs_horizon, act_dim = 8, 16, 2, 4
    embed_dim = 16

    tokenizer = ObjectTokenizer(_task_dim(2), relative_goal=True)
    encoder = ConditioningEncoder(
        proprio_dim=PROPRIO_DIM,
        token_dim=tokenizer.output_dim,
        tokens_per_step=tokenizer.tokens_per_step,
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
        role_dim=ROLE_DIM,
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
    tokenizer_c = ObjectTokenizer(_task_dim(5), relative_goal=True)
    obs_c = _obs_tree(batch_size, obs_horizon, num_objects=5)
    goal_c = _obs_tree(batch_size, obs_horizon, num_objects=5, time_axis=False)
    encoded = encoder(_tokenize(tokenizer_c, obs_c, goal_c))
    assert encoded["obs"]["task"].shape == (batch_size, obs_horizon, embed_dim)
    assert decoder(sample, timestep, external_cond=encoded).shape == sample.shape


def test_film_pipeline_rejects_variable_objects_without_pooling():
    """FiLM without object pooling cannot process a dynamic object-token count."""
    tokenizer = ObjectTokenizer(_task_dim(2), relative_goal=True)
    with pytest.raises(ValueError, match="pooling across objects"):
        ConditioningEncoder(
            proprio_dim=PROPRIO_DIM,
            token_dim=tokenizer.output_dim,
            tokens_per_step=None,
            goal_conditioned=True,
            relative_goal=True,
            role_dim=ROLE_DIM,
        )


def test_graph_pipeline_masks_absent_objects_out_of_the_decoder():
    """GraphTokenizer + GraphTransformer: absent clutter is carried as a validity mask all the way
    into the decoder's cross-attention, instead of being attended to at a parked pose."""
    batch_size, pred_horizon, obs_horizon, act_dim = 8, 16, 2, 4
    goal_horizon, num_slots, context_dim = 1, 5, 16

    tokenizer = GraphTokenizer(_task_dim(num_slots - 1), relative_goal=True)
    encoder = ConditioningEncoder(
        proprio_dim=PROPRIO_DIM,
        token_dim=tokenizer.output_dim,
        tokens_per_step=tokenizer.tokens_per_step,
        goal_conditioned=True,
        relative_goal=True,
        decoder_type="cross_attention",
        embedder={
            "_target_": (
                "policy.algorithms.networks.encoder.embedders.graph_transformer.GraphTransformer"
            ),
            "output_dim": context_dim,
            "obs_horizon": obs_horizon,
            "goal_horizon": goal_horizon,
            "num_heads": 2,
        },
        role_dim=ROLE_DIM,
    )
    decoder_cfg = _compose_decoder(
        "cross_attention_decoder1d", act_dim=act_dim, obs_horizon=obs_horizon
    )
    decoder = hydra_zen.instantiate(decoder_cfg, cond_dims=encoder.cond_dims)

    obs = _obs_tree(batch_size, obs_horizon, num_objects=num_slots - 1)
    goal = _obs_tree(batch_size, obs_horizon, num_objects=num_slots - 1, time_axis=False)
    obs["obj_valid"][0, :, 3:] = 0.0  # two objects absent, in one sample only
    goal["obj_valid"][0, 3:] = 0.0

    payload = encoder(_tokenize(tokenizer, obs, goal))

    steps = obs_horizon + goal_horizon
    assert payload["context"].shape == (batch_size, steps * num_slots, context_dim)
    # True marks an ignored token: two absent slots across every timestep, and only in sample 0.
    assert payload["context_mask"][0].sum() == 2 * steps
    assert not payload["context_mask"][1:].any()

    sample = torch.randn(batch_size, pred_horizon, act_dim)
    timestep = torch.randint(0, 100, (batch_size,))
    output = decoder(sample, timestep, external_cond=payload)

    assert output.shape == sample.shape
    assert torch.isfinite(output).all()

    output.sum().backward()
    _assert_gradients_flow(encoder, decoder)
