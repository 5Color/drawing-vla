"""정책 모델 모음. 학습(train_policy.py)과 롤아웃(rollout_eval.py)이 함께 쓴다.

공통 인터페이스
  policy.loss(obs, state, action) -> (loss, {이름: 값})     학습용
  policy.predict(obs, state)      -> (xy [B,16,2] ∈ [0,1], pen_prob [B,16])   추론용

  VisuomotorPolicy  (head="mlp")       v1·v2: ConvNet → MLP가 청크를 한 번에 회귀. 여러 스타일이 섞이면 평균을 냄.
  DiffusionPolicy   (head="diffusion") v3   : 같은 ConvNet 조건으로 잡음을 단계적으로 걷어내 청크를 '샘플링'.
                                              여러 갈래 중 한 스타일을 골라 그대로 밀고 나감.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CHUNK_SIZE, OBS_SIZE


def build_encoder(channels=(32, 64, 128, 256), feat_dim: int = 256, obs_size: int = OBS_SIZE) -> nn.Sequential:
    """(B, 2, 128, 128) → (B, feat_dim). Conv → BN → ReLU → MaxPool 4회: 128 → 8."""
    layers, c_in = [], 2
    for c in channels:
        layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        c_in = c
    side = obs_size // 2 ** len(channels)
    return nn.Sequential(*layers, nn.Flatten(), nn.Linear(c_in * side * side, feat_dim), nn.ReLU(inplace=True))


# ---------------------------------------------------------------------------
# v1·v2: MLP 회귀 헤드
# ---------------------------------------------------------------------------
def policy_loss(xy, pen_logit, action, pen_up_weight: float = 1.0):
    l_xy = F.l1_loss(xy, action[..., :2])
    pen = action[..., 2]
    w = torch.where(pen < 0.5, torch.full_like(pen, pen_up_weight), torch.ones_like(pen))
    l_pen = F.binary_cross_entropy_with_logits(pen_logit, pen, weight=w)
    return l_xy + 0.2 * l_pen, l_xy, l_pen


def build_state_net(state_dim: int, state_embed: int) -> nn.Module | None:
    """펜 상태(3) → state_embed차원. 3개 숫자를 256차원 이미지 특징 옆에 그냥 붙이면 모델이 무시하기 쉬워서(v2) 키워서 붙인다."""
    if not (state_dim and state_embed):
        return None
    return nn.Sequential(nn.Linear(state_dim, state_embed), nn.ReLU(inplace=True),
                         nn.Linear(state_embed, state_embed), nn.ReLU(inplace=True))


class VisuomotorPolicy(nn.Module):
    head_type = "mlp"

    def __init__(self, chunk_size: int = CHUNK_SIZE, channels=(32, 64, 128, 256), feat_dim: int = 256,
                 obs_size: int = OBS_SIZE, state_dim: int = 0, pen_up_weight: float = 1.0, state_embed: int = 0):
        super().__init__()
        self.chunk_size, self.state_dim, self.pen_up_weight = chunk_size, state_dim, pen_up_weight
        self.encoder = build_encoder(channels, feat_dim, obs_size)
        self.state_net = build_state_net(state_dim, state_embed)
        s_dim = state_embed if self.state_net is not None else state_dim
        self.head = nn.Sequential(nn.Linear(feat_dim + s_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, chunk_size * 3))

    def forward(self, obs, state=None):
        f = self.encoder(obs)
        if self.state_dim:
            f = torch.cat([f, self.state_net(state) if self.state_net is not None else state], 1)
        out = self.head(f).view(-1, self.chunk_size, 3)
        return torch.sigmoid(out[..., :2]), out[..., 2]  # xy ∈ [0,1], pen logits

    def loss(self, obs, state, action):
        xy, logit = self(obs, state)
        loss, l_xy, l_pen = policy_loss(xy, logit, action, self.pen_up_weight)
        return loss, {"xy": l_xy, "pen": l_pen}

    @torch.no_grad()
    def predict(self, obs, state=None):
        xy, logit = self(obs, state)
        return xy, torch.sigmoid(logit)


# ---------------------------------------------------------------------------
# v3: 디퓨전 헤드 (DDPM 학습, DDIM 샘플링, 잡음 예측)
# ---------------------------------------------------------------------------
def cosine_alphas_cumprod(T: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0).float()


class SinusoidalEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        a = t.float()[:, None] * freq[None]
        return torch.cat([a.sin(), a.cos()], -1)


class FiLMBlock(nn.Module):
    """잔차 MLP 블록. 조건(이미지 특징 + 펜 상태 + 시간)으로 scale/shift를 만들어 은닉값을 변조."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.film = nn.Linear(cond_dim, dim * 2)
        self.fc1, self.fc2 = nn.Linear(dim, dim * 2), nn.Linear(dim * 2, dim)

    def forward(self, h, c):
        scale, shift = self.film(c).chunk(2, -1)
        x = self.norm(h) * (1 + scale) + shift
        return h + self.fc2(F.mish(self.fc1(x)))


class Denoiser(nn.Module):
    def __init__(self, act_dim: int, cond_dim: int, hidden: int = 512, n_blocks: int = 4, t_dim: int = 128):
        super().__init__()
        self.t_emb = nn.Sequential(SinusoidalEmb(64), nn.Linear(64, t_dim), nn.Mish(), nn.Linear(t_dim, t_dim))
        self.inp = nn.Linear(act_dim, hidden)
        self.blocks = nn.ModuleList(FiLMBlock(hidden, cond_dim + t_dim) for _ in range(n_blocks))
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, act_dim))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, a_t, t, cond):
        c = torch.cat([cond, self.t_emb(t)], -1)
        h = self.inp(a_t)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(h)


class DiffusionPolicy(nn.Module):
    head_type = "diffusion"

    def __init__(self, chunk_size: int = CHUNK_SIZE, channels=(32, 64, 128, 256), feat_dim: int = 256,
                 obs_size: int = OBS_SIZE, state_dim: int = 3, hidden: int = 512, n_blocks: int = 4,
                 train_steps: int = 100, infer_steps: int = 10, noise_repeats: int = 4, state_embed: int = 0):
        super().__init__()
        self.chunk_size, self.state_dim = chunk_size, state_dim
        self.train_steps, self.infer_steps, self.noise_repeats = train_steps, infer_steps, noise_repeats
        self.encoder = build_encoder(channels, feat_dim, obs_size)
        self.state_net = build_state_net(state_dim, state_embed)
        s_dim = state_embed if self.state_net is not None else state_dim
        self.denoiser = Denoiser(chunk_size * 3, feat_dim + s_dim, hidden, n_blocks)
        self.register_buffer("acp", cosine_alphas_cumprod(train_steps), persistent=False)

    # 액션 정규화: x, y ∈ [0,1] → [-1,1], pen ∈ {0,1} → {-1,1}
    @staticmethod
    def _norm(action):
        return action * 2 - 1

    def _cond(self, obs, state):
        f = self.encoder(obs)
        if not self.state_dim:
            return f
        return torch.cat([f, self.state_net(state) if self.state_net is not None else state], 1)

    def loss(self, obs, state, action):
        B, K = action.shape[0], self.noise_repeats
        cond = self._cond(obs, state).repeat_interleave(K, 0)          # 이미지 인코딩 1번 → 잡음 K개
        a0 = self._norm(action).reshape(B, -1).repeat_interleave(K, 0)
        t = torch.randint(0, self.train_steps, (B * K,), device=a0.device)
        noise = torch.randn_like(a0)
        ab = self.acp[t][:, None]
        a_t = ab.sqrt() * a0 + (1 - ab).sqrt() * noise
        loss = F.mse_loss(self.denoiser(a_t, t, cond), noise)
        return loss, {"eps": loss}

    @torch.no_grad()
    def predict(self, obs, state=None, steps: int | None = None, generator: torch.Generator | None = None,
                noise_scale: float = 1.0):
        """noise_scale < 1 = 저온 샘플링: 초기 잡음을 줄여 다양성을 낮춤 (1.0이 정식 DDIM)."""
        cond = self._cond(obs, state)
        B, D = cond.shape[0], self.chunk_size * 3
        a = torch.randn(B, D, device=cond.device, generator=generator) * noise_scale
        ts = torch.linspace(self.train_steps - 1, 0, steps or self.infer_steps, device=cond.device).round().long()
        for i, t in enumerate(ts):  # DDIM (eta=0)
            ab = self.acp[t]
            ab_prev = self.acp[ts[i + 1]] if i + 1 < len(ts) else torch.ones((), device=cond.device)
            eps = self.denoiser(a, t.expand(B), cond)
            a0 = ((a - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            eps = (a - ab.sqrt() * a0) / (1 - ab).sqrt()
            a = ab_prev.sqrt() * a0 + (1 - ab_prev).sqrt() * eps
        a = ((a.clamp(-1, 1) + 1) / 2).view(B, self.chunk_size, 3)
        return a[..., :2], a[..., 2]


# ---------------------------------------------------------------------------
# v5: Transformer 기반 어텐션 헤드 (ACT 스타일 Action Chunking)
# ---------------------------------------------------------------------------
class ConvSpatialEncoder(nn.Module):
    """(B, 2, 128, 128) → (B, N_patches, d_model) 공간 토큰 시퀀스."""
    def __init__(self, channels=(32, 64, 128, 256), d_model: int = 256):
        super().__init__()
        layers, c_in = [], 2
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Conv2d(c_in, d_model, 1)

    def forward(self, x):
        h = self.proj(self.conv(x))  # (B, d_model, 8, 8)
        return h.flatten(2).permute(0, 2, 1)  # (B, 64, d_model)


class TransformerPolicy(nn.Module):
    head_type = "transformer"

    def __init__(self, chunk_size: int = CHUNK_SIZE, channels=(32, 64, 128, 256), d_model: int = 256,
                 obs_size: int = OBS_SIZE, state_dim: int = 3, nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 512, pen_up_weight: float = 10.0, **kwargs):
        super().__init__()
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.pen_up_weight = pen_up_weight
        self.encoder = ConvSpatialEncoder(channels, d_model)
        
        self.state_proj = nn.Linear(state_dim, d_model) if state_dim else None
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.0, activation="relu", batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, 3)

    def forward(self, obs, state=None):
        B = obs.shape[0]
        vis_tokens = self.encoder(obs)  # (B, 64, d_model)
        
        if self.state_proj is not None and state is not None:
            state_token = self.state_proj(state).unsqueeze(1)  # (B, 1, d_model)
            memory = torch.cat([vis_tokens, state_token], dim=1)
        else:
            memory = vis_tokens

        queries = self.action_queries.expand(B, -1, -1)  # (B, 16, d_model)
        h = self.decoder(tgt=queries, memory=memory)      # (B, 16, d_model)
        out = self.out_proj(h)                           # (B, 16, 3)
        
        return torch.sigmoid(out[..., :2]), out[..., 2]  # xy ∈ [0,1], pen logits

    def loss(self, obs, state, action):
        xy, logit = self(obs, state)
        loss, l_xy, l_pen = policy_loss(xy, logit, action, self.pen_up_weight)
        return loss, {"xy": l_xy, "pen": l_pen}

    @torch.no_grad()
    def predict(self, obs, state=None):
        xy, logit = self(obs, state)
        return xy, torch.sigmoid(logit)


# ---------------------------------------------------------------------------
def build_policy(cfg: dict) -> nn.Module:
    common = {"chunk_size": cfg["chunk_size"], "channels": tuple(cfg["channels"]), "obs_size": cfg["obs_size"],
              "state_dim": cfg.get("state_dim", 0), "state_embed": cfg.get("state_embed", 0)}
    if cfg.get("head", "mlp") == "diffusion":
        return DiffusionPolicy(**common, **cfg.get("diffusion", {}))
    if cfg.get("head") == "transformer":
        return TransformerPolicy(**common, pen_up_weight=cfg.get("pen_up_weight", 10.0), **cfg.get("transformer", {}))
    return VisuomotorPolicy(**common, pen_up_weight=cfg.get("pen_up_weight", 1.0))
