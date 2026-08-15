import hydra_zen
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf


def _compose_encoder(config_name, **overrides):
    with initialize_config_module(config_module="policy.configs", version_base="1.2"):
        cfg = compose(config_name=f"algorithm/encoder/{config_name}")
    encoder_cfg = cfg.algorithm.encoder
    OmegaConf.set_struct(encoder_cfg, False)
    for key, value in overrides.items():
        encoder_cfg[key] = value
    return encoder_cfg


def _compose_decoder(config_name, **overrides):
    with initialize_config_module(config_module="policy.configs", version_base="1.2"):
        cfg = compose(config_name=f"algorithm/decoder/{config_name}")
    net_cfg = cfg.algorithm.decoder
    OmegaConf.set_struct(net_cfg, False)
    for key, value in overrides.items():
        net_cfg[key] = value
    return net_cfg


def test_film_pipeline_instantiates_and_runs():
    """Plain, unconditioned FiLM pipeline: no goal, default (flatten, no-embedder) encoding.

    Exercises the full obs -> ConditioningEncoder -> FiLMDecoder1D chain exactly as
    DiffusionPolicy._encode() + self.decoder(...) run it, without going through the algorithm
    class itself.
    """
    batch_size, horizon, act_dim = 128, 16, 8
    proprio_dim, task_dim = 17, 50

    encoder_cfg = _compose_encoder(
        "film", obs_dim=proprio_dim + task_dim, goal_conditioned=False, proprio_dim=proprio_dim
    )
    encoder = hydra_zen.instantiate(encoder_cfg)

    decoder_cfg = _compose_decoder("film_decoder1d", act_dim=act_dim, obs_horizon=horizon)
    decoder = hydra_zen.instantiate(decoder_cfg, cond_dims=encoder.cond_dims)

    sample = torch.randn(batch_size, horizon, act_dim)
    timestep = torch.randint(0, 100, (batch_size,))
    obs = torch.randn(batch_size, horizon, proprio_dim + task_dim)

    encoded_cond = encoder(obs)
    output = decoder(sample, timestep, external_cond=encoded_cond)
    assert output.shape == sample.shape

    loss = output.sum()
    loss.backward()
    for name, p in list(encoder.named_parameters()) + list(decoder.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} got a non-finite gradient"


def test_film_pipeline_goal_conditioned_absolute_mode_instantiates_and_runs():
    """goal_conditioned=True, relative_goal=False: a separate 'goal' entry alongside 'obs'."""
    batch_size, horizon, act_dim = 8, 4, 4
    proprio_dim, task_dim = 5, 10

    encoder_cfg = _compose_encoder(
        "film", obs_dim=proprio_dim + task_dim, goal_conditioned=True, proprio_dim=proprio_dim
    )
    encoder = hydra_zen.instantiate(encoder_cfg)

    decoder_cfg = _compose_decoder("film_decoder1d", act_dim=act_dim, obs_horizon=horizon)
    decoder = hydra_zen.instantiate(decoder_cfg, cond_dims=encoder.cond_dims)

    sample = torch.randn(batch_size, horizon, act_dim)
    timestep = torch.randint(0, 100, (batch_size,))
    obs = torch.randn(batch_size, horizon, proprio_dim + task_dim)
    goal = torch.randn(batch_size, proprio_dim + task_dim)

    encoded_cond = encoder(obs, goal)
    output = decoder(sample, timestep, external_cond=encoded_cond)
    assert output.shape == sample.shape

    loss = output.sum()
    loss.backward()
    for name, p in list(encoder.named_parameters()) + list(decoder.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} got a non-finite gradient"


def test_cross_attention_pipeline_instantiates_and_runs():
    """Cross-attention pipeline with a real ObjectTokenizer + SelfAttention embedder: the per-
    object token sequence is cross-attended over instead of flattened into FiLM."""
    batch_size, pred_horizon, obs_horizon, act_dim = 8, 16, 2, 4
    proprio_dim, context_dim = 18, 8

    encoder_cfg = _compose_encoder(
        "cross_attention",
        obs_dim={"proprio": proprio_dim, "obj_0_pose": 7, "obj_1_pose": 7, "tcp_pose": 7},
        goal_conditioned=True,
        tokenizer={"_target_": "policy.algorithms.networks.encoder.tokenizers.ObjectTokenizer"},
        embedder={
            "_target_": "policy.algorithms.networks.encoder.embedders.self_attention.SelfAttention",
            "output_dim": context_dim,
            "obs_horizon": obs_horizon,
            "num_heads": 2,
        },
    )
    encoder = hydra_zen.instantiate(encoder_cfg)

    decoder_cfg = _compose_decoder(
        "cross_attention_decoder1d", act_dim=act_dim, obs_horizon=obs_horizon
    )
    decoder = hydra_zen.instantiate(decoder_cfg, cond_dims=encoder.cond_dims)

    sample = torch.randn(batch_size, pred_horizon, act_dim)
    timestep = torch.randint(0, 100, (batch_size,))
    obs = {
        "proprio": torch.randn(batch_size, obs_horizon, proprio_dim),
        "obj_0_pose": torch.randn(batch_size, obs_horizon, 7),
        "obj_0_role": torch.tensor([1.0, 0.0, 0.0]).expand(batch_size, obs_horizon, 3),
        "obj_1_pose": torch.randn(batch_size, obs_horizon, 7),
        "obj_1_role": torch.tensor([0.0, 1.0, 0.0]).expand(batch_size, obs_horizon, 3),
        "tcp_pose": torch.randn(batch_size, obs_horizon, 7),
    }
    goal = {
        "proprio": torch.randn(batch_size, proprio_dim),
        "obj_0_pose": torch.randn(batch_size, 7),
        "obj_0_role": torch.tensor([1.0, 0.0, 0.0]).expand(batch_size, 3),
        "obj_1_pose": torch.randn(batch_size, 7),
        "obj_1_role": torch.tensor([0.0, 1.0, 0.0]).expand(batch_size, 3),
        "tcp_pose": torch.randn(batch_size, 7),
    }

    encoded_cond = encoder(obs, goal)
    output = decoder(sample, timestep, external_cond=encoded_cond)
    assert output.shape == sample.shape

    loss = output.sum()
    loss.backward()
    for name, p in list(encoder.named_parameters()) + list(decoder.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} got a non-finite gradient"

    # Verify that the same instantiated pipeline handles a different number of objects (e.g. adding 2 clutter objects)
    obs_with_clutter = dict(obs)
    obs_with_clutter["obj_2_pose"] = torch.randn(batch_size, obs_horizon, 7)
    obs_with_clutter["obj_2_role"] = torch.tensor([0.0, 0.0, 1.0]).expand(batch_size, obs_horizon, 3)
    obs_with_clutter["obj_3_pose"] = torch.randn(batch_size, obs_horizon, 7)
    obs_with_clutter["obj_3_role"] = torch.tensor([0.0, 0.0, 1.0]).expand(batch_size, obs_horizon, 3)
    goal_with_clutter = dict(goal)
    goal_with_clutter["obj_2_pose"] = torch.randn(batch_size, 7)
    goal_with_clutter["obj_2_role"] = torch.tensor([0.0, 0.0, 1.0]).expand(batch_size, 3)
    goal_with_clutter["obj_3_pose"] = torch.randn(batch_size, 7)
    goal_with_clutter["obj_3_role"] = torch.tensor([0.0, 0.0, 1.0]).expand(batch_size, 3)

    encoded_cond_clutter = encoder(obs_with_clutter, goal_with_clutter)
    # Context sequence length should now be obs_horizon * 4 objects = 8
    assert encoded_cond_clutter["context"].shape == (batch_size, obs_horizon * 4, context_dim)
    output_clutter = decoder(sample, timestep, external_cond=encoded_cond_clutter)
    assert output_clutter.shape == sample.shape


def test_film_pipeline_with_object_tokenizer_and_attention_pooling():
    """FiLM pipeline with ObjectTokenizer + AttentionPooling(mode="objects"):

    K object tokens are compressed into 1 vector per timestep, allowing variable objects.
    """
    batch_size, pred_horizon, obs_horizon, act_dim = 8, 16, 2, 4
    proprio_dim, embed_dim = 18, 16

    encoder_cfg = _compose_encoder(
        "film",
        obs_dim={"proprio": proprio_dim, "obj_0_pose": 7, "obj_1_pose": 7, "tcp_pose": 7},
        goal_conditioned=True,
        relative_goal=True,
        proprio_dim=proprio_dim,
        tokenizer={"_target_": "policy.algorithms.networks.encoder.tokenizers.ObjectTokenizer"},
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
    encoder = hydra_zen.instantiate(encoder_cfg)

    decoder_cfg = _compose_decoder("film_decoder1d", act_dim=act_dim, obs_horizon=obs_horizon)
    decoder = hydra_zen.instantiate(decoder_cfg, cond_dims=encoder.cond_dims)

    sample = torch.randn(batch_size, pred_horizon, act_dim)
    timestep = torch.randint(0, 100, (batch_size,))
    obs = {
        "proprio": torch.randn(batch_size, obs_horizon, proprio_dim),
        "obj_0_pose": torch.randn(batch_size, obs_horizon, 7),
        "obj_0_role": torch.tensor([1.0, 0.0, 0.0]).expand(batch_size, obs_horizon, 3),
        "obj_1_pose": torch.randn(batch_size, obs_horizon, 7),
        "obj_1_role": torch.tensor([0.0, 1.0, 0.0]).expand(batch_size, obs_horizon, 3),
        "tcp_pose": torch.randn(batch_size, obs_horizon, 7),
    }
    goal = {
        "proprio": torch.randn(batch_size, proprio_dim),
        "obj_0_pose": torch.randn(batch_size, 7),
        "obj_0_role": torch.tensor([1.0, 0.0, 0.0]).expand(batch_size, 3),
        "obj_1_pose": torch.randn(batch_size, 7),
        "obj_1_role": torch.tensor([0.0, 1.0, 0.0]).expand(batch_size, 3),
        "tcp_pose": torch.randn(batch_size, 7),
    }

    encoded_cond = encoder(obs, goal)
    output = decoder(sample, timestep, external_cond=encoded_cond)
    assert output.shape == sample.shape

    # Same pipeline with 3 clutter objects added
    obs_with_clutter = dict(obs)
    obs_with_clutter["obj_2_pose"] = torch.randn(batch_size, obs_horizon, 7)
    obs_with_clutter["obj_2_role"] = torch.tensor([0.0, 0.0, 1.0]).expand(batch_size, obs_horizon, 3)
    obs_with_clutter["obj_3_pose"] = torch.randn(batch_size, obs_horizon, 7)
    obs_with_clutter["obj_3_role"] = torch.tensor([0.0, 0.0, 1.0]).expand(batch_size, obs_horizon, 3)
    obs_with_clutter["obj_4_pose"] = torch.randn(batch_size, obs_horizon, 7)
    obs_with_clutter["obj_4_role"] = torch.tensor([0.0, 0.0, 1.0]).expand(batch_size, obs_horizon, 3)
    goal_with_clutter = dict(goal)
    goal_with_clutter["obj_2_pose"] = torch.randn(batch_size, 7)
    goal_with_clutter["obj_2_role"] = torch.tensor([0.0, 0.0, 1.0]).expand(batch_size, 3)
    goal_with_clutter["obj_3_pose"] = torch.randn(batch_size, 7)
    goal_with_clutter["obj_3_role"] = torch.tensor([0.0, 0.0, 1.0]).expand(batch_size, 3)
    goal_with_clutter["obj_4_pose"] = torch.randn(batch_size, 7)
    goal_with_clutter["obj_4_role"] = torch.tensor([0.0, 0.0, 1.0]).expand(batch_size, 3)

    encoded_cond_clutter = encoder(obs_with_clutter, goal_with_clutter)
    # FiLM conditioning shape remains exactly the same because of object pooling
    assert encoded_cond_clutter["obs"]["task"].shape == (batch_size, obs_horizon, embed_dim)
    output_clutter = decoder(sample, timestep, external_cond=encoded_cond_clutter)
    assert output_clutter.shape == sample.shape


def test_film_pipeline_rejects_variable_objects_without_pooling():
    """FiLM without object pooling cannot process dynamic/variable object tokens."""
    import pytest
    encoder_cfg = _compose_encoder(
        "film",
        obs_dim={"proprio": 18, "obj_0_pose": 7, "obj_1_pose": 7, "tcp_pose": 7},
        goal_conditioned=True,
        relative_goal=True,
        proprio_dim=18,
        tokenizer={"_target_": "policy.algorithms.networks.encoder.tokenizers.ObjectTokenizer"},
        pooling=None,
    )
    with pytest.raises(Exception, match="pooling across objects"):
        hydra_zen.instantiate(encoder_cfg)
