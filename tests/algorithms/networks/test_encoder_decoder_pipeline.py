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
        obs_dim={"proprio": proprio_dim, "a_pose": 7, "b_pose": 7, "tcp_pose": 7},
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
        "a_pose": torch.randn(batch_size, obs_horizon, 7),
        "b_pose": torch.randn(batch_size, obs_horizon, 7),
        "tcp_pose": torch.randn(batch_size, obs_horizon, 7),
    }
    goal = {
        "proprio": torch.randn(batch_size, proprio_dim),
        "a_pose": torch.randn(batch_size, 7),
        "b_pose": torch.randn(batch_size, 7),
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
