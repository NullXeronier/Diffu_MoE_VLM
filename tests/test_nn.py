import numpy as np
import pytest

torch = pytest.importorskip("torch")

from diffu_moe_vlm.imu import load_trajectories, make_synthetic_trajectories, save_trajectories, to_torch_batch  # noqa: E402
from diffu_moe_vlm.nn.diffusion import build_diffusion_policy  # noqa: E402
from diffu_moe_vlm.nn.encoders import build_encoder  # noqa: E402
from diffu_moe_vlm.nn.moe import MoELayer, Trunk  # noqa: E402
from diffu_moe_vlm.nn.policy import build_actor_critic  # noqa: E402
from diffu_moe_vlm.nn.time_embedding import TrajectoryEncoder  # noqa: E402


def images(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, 64, 64, 3), dtype=torch.uint8, generator=g)


@pytest.mark.parametrize("name", ["cnn", "vit"])
def test_encoders_output_feature_dim(name):
    enc = build_encoder(name, feature_dim=96)
    assert enc(images()).shape == (4, 96)


def test_pretrained_encoder_from_config():
    transformers = pytest.importorskip("transformers")
    config = transformers.CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                           num_attention_heads=2, image_size=32, patch_size=8)
    from diffu_moe_vlm.nn.encoders import PretrainedVisionEncoder
    enc = PretrainedVisionEncoder(feature_dim=48, config=config)
    assert enc(images(2)).shape == (2, 48)
    assert not any(p.requires_grad for p in enc.vision.parameters())


def test_moe_routes_top_k_and_balances():
    torch.manual_seed(0)
    layer = MoELayer(dim=16, hidden_dim=32, num_experts=4, top_k=2)
    x = torch.randn(256, 16)
    y, aux = layer(x)
    assert y.shape == x.shape
    assert 1.0 <= aux.item() <= 4.0       # 1 = perfectly balanced, E = everything on one expert
    assert abs(layer.last_load.sum().item() - 1.0) < 1e-5
    (y.pow(2).mean() + aux).backward()
    assert layer.gate.weight.grad is not None and layer.gate.weight.grad.abs().sum() > 0


def test_moe_aux_loss_penalizes_collapse():
    layer = MoELayer(dim=8, hidden_dim=16, num_experts=4, top_k=1, noisy_gating=False)
    with torch.no_grad():
        layer.gate.weight.zero_()
        layer.gate.weight[0] = 10.0       # every positive input goes to expert 0
    _, aux = layer(torch.ones(32, 8))
    assert aux.item() > 3.5


def test_dense_trunk_has_no_aux_loss():
    h, aux = Trunk(16, 32, kind="mlp")(torch.randn(3, 16))
    assert h.shape == (3, 16) and aux.item() == 0.0
    with pytest.raises(ValueError):
        Trunk(16, 32, kind="unknown")


def test_actor_critic_shapes_and_trajectory_conditioning():
    model = build_actor_critic({"encoder": {"name": "cnn", "feature_dim": 64}, "hidden_dim": 64,
                                "trajectory": {"num_joints": 3, "dim": 32, "out_dim": 16}}, num_actions=17)
    traj = to_torch_batch(make_synthetic_trajectories(4, 20))
    logits, value, aux = model(images(), traj)
    assert logits.shape == (4, 17) and value.shape == (4,) and aux.ndim == 0
    action, logp, _ = model.act(images(), traj)
    assert action.shape == (4,) and torch.all(logp <= 0)
    with pytest.raises(ValueError):
        model(images())


def test_trajectory_encoder_ignores_masked_padding():
    torch.manual_seed(0)
    enc = TrajectoryEncoder(num_joints=3, dim=32, out_dim=8).eval()
    data = to_torch_batch(make_synthetic_trajectories(2, 30))
    short = enc(data["positions"][:, :20], data["timestamps"][:, :20])
    mask = torch.zeros(2, 30, dtype=torch.bool)
    mask[:, :20] = True
    padded = enc(data["positions"], data["timestamps"], mask)
    assert torch.allclose(short, padded, atol=1e-5)


def test_trajectory_encoder_learns_synthetic_motions(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "traj.npz"
    save_trajectories(path, make_synthetic_trajectories(512, 48, seed=1))
    data = load_trajectories(path)
    labels = torch.as_tensor(data["labels"])
    enc = TrajectoryEncoder(num_joints=3, dim=64, out_dim=4)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    train, test = np.arange(384), np.arange(384, 512)
    for step in range(150):
        idx = np.random.default_rng(step).choice(train, 64)
        loss = torch.nn.functional.cross_entropy(enc(**to_torch_batch(data, idx)), labels[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (enc(**to_torch_batch(data, test)).argmax(-1) == labels[test]).float().mean().item()
    assert acc > 0.9, acc


def test_diffusion_policy_memorizes_observation_conditioned_chunks():
    torch.manual_seed(0)
    policy = build_diffusion_policy({"encoder": {"name": "cnn", "feature_dim": 64}, "horizon": 4,
                                     "num_diffusion_steps": 20, "hidden_dim": 128, "depth": 2}, num_actions=5)
    obs = torch.zeros(2, 64, 64, 3, dtype=torch.uint8)
    obs[1] = 255
    targets = torch.tensor([[0, 1, 2, 3], [4, 4, 0, 2]])
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for _ in range(200):
        loss = policy.loss(obs.repeat(16, 1, 1, 1), targets.repeat(16, 1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    policy.eval()
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        assert torch.equal(policy.sample(obs, generator=g), targets)              # full DDPM
        assert torch.equal(policy.sample(obs, num_steps=5, generator=g), targets)  # strided DDIM


def test_diffusion_epsilon_prediction_runs():
    policy = build_diffusion_policy({"encoder": {"name": "cnn", "feature_dim": 32}, "horizon": 2,
                                     "num_diffusion_steps": 5, "hidden_dim": 32, "depth": 1,
                                     "prediction_type": "epsilon"}, num_actions=3)
    obs = torch.zeros(2, 64, 64, 3, dtype=torch.uint8)
    assert policy.loss(obs, torch.zeros(2, 2, dtype=torch.long)).ndim == 0
    assert policy.sample(obs).shape == (2, 2) and policy.sample(obs, num_steps=2).shape == (2, 2)
    with pytest.raises(ValueError):
        build_diffusion_policy({"prediction_type": "v"}, num_actions=3)
