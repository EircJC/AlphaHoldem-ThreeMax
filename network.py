"""AlphaHoldem 伪孪生网络（PyTorch）。
两条不共享参数的卷积流分别吃牌张量与动作张量，融合后接策略头 + 价值头。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ACTION_CHANNELS as ACTION_CH, NUM_SCALAR

CARD_CH, RANKS, SUITS = 6, 13, 4
NEG_INF = -1e9


def _conv_stream(in_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, 64, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        nn.Flatten(),
    )


class AlphaHoldemNet(nn.Module):
    def __init__(self, num_actions: int, nb: int):
        super().__init__()
        self.card_stream = _conv_stream(CARD_CH)          # 输出 64*4*13
        self.action_stream = _conv_stream(ACTION_CH)      # 输出 64*4*nb
        # 融合层输入 = 牌特征 + 动作特征 + 标量特征(有效筹码/SPR)
        fused = 64 * SUITS * RANKS + 64 * SUITS * nb + NUM_SCALAR
        self.fuse = nn.Sequential(nn.Linear(fused, 256), nn.ReLU())
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, card, action, scalar):
        feats = torch.cat([self.card_stream(card), self.action_stream(action), scalar], dim=1)
        z = self.fuse(feats)
        return self.policy_head(z), self.value_head(z).squeeze(-1)

    @staticmethod
    def _masked_logits(logits, legal):
        # legal: (B, A) 0/1；非法动作置 -inf
        return torch.where(legal > 0.5, logits, torch.full_like(logits, NEG_INF))

    @torch.no_grad()
    def act(self, card, action, scalar, legal, greedy=False):
        """给定单个/批量局面，采样动作。返回 (action_idx, logp, value)。"""
        logits, value = self.forward(card, action, scalar)
        logits = self._masked_logits(logits, legal)
        dist = torch.distributions.Categorical(logits=logits)
        a = torch.argmax(logits, dim=-1) if greedy else dist.sample()
        return a, dist.log_prob(a), value

    def evaluate_actions(self, card, action, scalar, legal, actions):
        """PPO 用：返回给定动作的 logp、分布熵、价值。"""
        logits, value = self.forward(card, action, scalar)
        logits = self._masked_logits(logits, legal)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


class PrivilegedCritic(nn.Module):
    """【Stage 2】训练期专用的特权评论家（价值网络）。

    与 actor 唯一的差别：牌张量多一个通道（第 6 通道 = 对手底牌），因此它**看得见对手的牌**，
    能真正预测本手的期望收益 → 提供【低方差、无偏】的策略梯度基线（基线只依赖决策前状态、
    与所选动作无关，满足 baseline 定理，不引入偏差，仅降方差）。

    ⚠️ 仅在训练进程存在：只用于算优势基线与价值损失；**不参与推理，不导出 ONNX**。
    actor（AlphaHoldemNet）的输入与结构完全不变，部署链路零改动。
    """

    def __init__(self, num_actions: int, nb: int):
        super().__init__()
        self.card_stream = _conv_stream(CARD_CH + 1)      # 7 通道（含对手底牌）
        self.action_stream = _conv_stream(ACTION_CH)      # 与 actor 同：吃公共动作张量
        fused = 64 * SUITS * RANKS + 64 * SUITS * nb + NUM_SCALAR   # 卷积输出通道恒 64，与 actor 同维
        self.fuse = nn.Sequential(nn.Linear(fused, 256), nn.ReLU())
        self.value_head = nn.Linear(256, 1)

    def forward(self, card7, action, scalar):
        feats = torch.cat([self.card_stream(card7), self.action_stream(action), scalar], dim=1)
        return self.value_head(self.fuse(feats)).squeeze(-1)

    @torch.no_grad()
    def value(self, card7, action, scalar):
        """采样期算基线用（不建图）。"""
        return self.forward(card7, action, scalar)
