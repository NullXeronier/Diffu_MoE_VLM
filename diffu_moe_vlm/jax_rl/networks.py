"""Flax actor-critic networks: feed-forward, GRU (PPO-RNN) and an MoE layer"""

import functools
from typing import Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


def _act(name: str):
    return {"tanh": nn.tanh, "relu": nn.relu, "gelu": nn.gelu}[name]


class MoEDense(nn.Module):
    """Top-k gated mixture of MLP experts; returns (output, load-balancing loss)"""
    features: int
    num_experts: int = 4
    top_k: int = 2
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        logits = nn.Dense(self.num_experts, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        probs = jax.nn.softmax(logits, axis=-1)
        top_vals, top_idx = jax.lax.top_k(probs, self.top_k)
        weights = top_vals / top_vals.sum(-1, keepdims=True)
        experts = [nn.Dense(self.features, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
                   for _ in range(self.num_experts)]
        outs = jnp.stack([_act(self.activation)(e(x)) for e in experts], axis=-2)   # (..., E, F)
        chosen = jnp.take_along_axis(outs, top_idx[..., None], axis=-2)              # (..., k, F)
        y = (weights[..., None] * chosen).sum(-2)
        flat_probs = probs.reshape(-1, self.num_experts)
        load = jax.nn.one_hot(top_idx[..., 0].reshape(-1), self.num_experts).mean(0)
        aux = self.num_experts * jnp.sum(load * flat_probs.mean(0))
        return y, aux


class MLPTrunk(nn.Module):
    """`num_layers` hidden layers of width `width`; dense or MoE"""
    width: int = 512
    num_layers: int = 3
    activation: str = "tanh"
    moe: bool = False
    num_experts: int = 4
    top_k: int = 2

    @nn.compact
    def __call__(self, x):
        aux = jnp.zeros(())
        for _ in range(self.num_layers):
            if self.moe:
                x, a = MoEDense(self.width, self.num_experts, self.top_k, self.activation)(x)
                aux = aux + a
            else:
                x = _act(self.activation)(nn.Dense(self.width, kernel_init=orthogonal(np.sqrt(2)),
                                                   bias_init=constant(0.0))(x))
        return x, aux


class ActorCritic(nn.Module):
    """Craftax-baseline style feed-forward actor-critic (separate actor and critic trunks)"""
    action_dim: int
    width: int = 512
    num_layers: int = 3
    activation: str = "tanh"
    moe: bool = False
    num_experts: int = 4
    top_k: int = 2

    @nn.compact
    def __call__(self, x):
        trunk = functools.partial(MLPTrunk, self.width, self.num_layers, self.activation,
                                  self.moe, self.num_experts, self.top_k)
        a, aux_a = trunk()(x)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        c, aux_c = trunk()(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        return distrax.Categorical(logits=logits), jnp.squeeze(value, -1), aux_a + aux_c


class ScannedGRU(nn.Module):
    """GRU scanned over time; the hidden state is reset where `resets` is true"""

    @functools.partial(
        nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        carry = jnp.where(resets[:, None], self.initialize_carry(ins.shape[0], ins.shape[-1]), carry)
        new_carry, y = nn.GRUCell(features=ins.shape[-1])(carry, ins)
        return new_carry, y

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int):
        return jnp.zeros((batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    """PPO-RNN: embedding -> GRU -> actor / critic heads. Inputs are (T, B, ...)"""
    action_dim: int
    width: int = 512
    activation: str = "relu"
    moe: bool = False
    num_experts: int = 4
    top_k: int = 2

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        emb = _act(self.activation)(nn.Dense(self.width, kernel_init=orthogonal(np.sqrt(2)),
                                             bias_init=constant(0.0))(obs))
        hidden, emb = ScannedGRU()(hidden, (emb, dones))
        a, aux_a = MLPTrunk(self.width, 1, self.activation, self.moe, self.num_experts, self.top_k)(emb)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        c, aux_c = MLPTrunk(self.width, 1, self.activation, self.moe, self.num_experts, self.top_k)(emb)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        return hidden, distrax.Categorical(logits=logits), jnp.squeeze(value, -1), aux_a + aux_c


class ICM(nn.Module):
    """Intrinsic Curiosity Module (Pathak et al., 2017): encoder, inverse and forward models"""
    action_dim: int
    feature_dim: int = 256
    width: int = 256

    def setup(self):
        self.encoder = nn.Sequential([nn.Dense(self.width), nn.relu, nn.Dense(self.feature_dim)])
        self.inverse = nn.Sequential([nn.Dense(self.width), nn.relu, nn.Dense(self.action_dim)])
        self.forward_model = nn.Sequential([nn.Dense(self.width), nn.relu, nn.Dense(self.feature_dim)])

    def __call__(self, obs, next_obs, action) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Returns (inverse_logits, predicted next features, next features)"""
        phi, phi_next = self.encoder(obs), self.encoder(next_obs)
        inverse_logits = self.inverse(jnp.concatenate([phi, phi_next], -1))
        pred_next = self.forward_model(
            jnp.concatenate([jax.lax.stop_gradient(phi), jax.nn.one_hot(action, self.action_dim)], -1))
        return inverse_logits, pred_next, phi_next


def num_params(params) -> int:
    return int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)))

