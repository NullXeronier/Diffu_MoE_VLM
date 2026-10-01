"""
Visual encoders for pixel observations.

All encoders take uint8 images of shape (B, H, W, 3) and return (B, feature_dim).
`cnn` and `vit` are small models trained from scratch (CPU friendly);
`pretrained` wraps a HuggingFace CLIP/SigLIP vision tower.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_float(images: torch.Tensor) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> (B, 3, H, W) float in [0, 1]"""
    return images.permute(0, 3, 1, 2).float() / 255.0


class CNNEncoder(nn.Module):
    """Small convolutional encoder (Nature-DQN style) for 64x64 inputs"""

    def __init__(self, image_size: int = 64, feature_dim: int = 256, channels=(32, 64, 64)):
        super().__init__()
        c1, c2, c3 = channels
        self.conv = nn.Sequential(
            nn.Conv2d(3, c1, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, 3, image_size, image_size)).shape[1]
        self.proj = nn.Sequential(nn.Linear(flat, feature_dim), nn.ReLU())
        self.feature_dim = feature_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(_to_float(images)))


class ViTEncoder(nn.Module):
    """Tiny Vision Transformer: patch embedding + transformer encoder + CLS pooling"""

    def __init__(self, image_size: int = 64, patch_size: int = 8, dim: int = 128, depth: int = 2,
                 heads: int = 4, feature_dim: int = 256):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        num_patches = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Linear(dim, feature_dim), nn.ReLU())
        self.feature_dim = feature_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch(_to_float(images)).flatten(2).transpose(1, 2)  # (B, N, dim)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos
        x = self.norm(self.transformer(x))
        return self.proj(x[:, 0])


class PretrainedVisionEncoder(nn.Module):
    """
    HuggingFace vision tower (e.g. "openai/clip-vit-base-patch32",
    "google/siglip-base-patch16-224"). Images are resized to the model input
    size and normalized with the model's statistics. Requires `transformers`.
    """

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", feature_dim: int = 256,
                 freeze: bool = True, config=None):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        if config is None:
            model = AutoModel.from_pretrained(model_name)
        else:  # build from a config without downloading weights (tests, custom sizes)
            model = AutoModel.from_config(config if not isinstance(config, str) else AutoConfig.from_pretrained(config))
        self.vision = getattr(model, "vision_model", model)
        vision_cfg = getattr(self.vision, "config", None) or model.config
        self.input_size = vision_cfg.image_size
        hidden = vision_cfg.hidden_size
        mean, std = self.CLIP_MEAN, self.CLIP_STD
        if "siglip" in type(self.vision).__name__.lower():
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        if freeze:
            for p in self.vision.parameters():
                p.requires_grad_(False)
        self.freeze = freeze
        self.proj = nn.Sequential(nn.Linear(hidden, feature_dim), nn.ReLU())
        self.feature_dim = feature_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = _to_float(images)
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        with torch.set_grad_enabled(not self.freeze and torch.is_grad_enabled()):
            out = self.vision(pixel_values=x)
        pooled = out.pooler_output if getattr(out, "pooler_output", None) is not None \
            else out.last_hidden_state.mean(dim=1)
        return self.proj(pooled)


def build_encoder(name: str = "cnn", image_size: int = 64, feature_dim: int = 256,
                  pretrained_model: Optional[str] = None, freeze: bool = True, **kwargs) -> nn.Module:
    """Encoder factory used by configs: name in {cnn, vit, pretrained}"""
    if name == "cnn":
        return CNNEncoder(image_size=image_size, feature_dim=feature_dim)
    if name == "vit":
        return ViTEncoder(image_size=image_size, feature_dim=feature_dim, **kwargs)
    if name == "pretrained":
        return PretrainedVisionEncoder(pretrained_model or "openai/clip-vit-base-patch32",
                                       feature_dim=feature_dim, freeze=freeze)
    raise ValueError(f"Unknown encoder {name!r} (expected cnn, vit or pretrained)")
