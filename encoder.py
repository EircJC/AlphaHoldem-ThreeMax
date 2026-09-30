"""把 HUNLEnv 的局面编码成 AlphaHoldem 的两个张量（numpy, float32）：
  牌张量  card   : (6, 4, 13)  通道 = 底牌/翻牌/转牌/河牌/全部公共牌/全部牌
  动作张量 action : (24, 4, NB) 通道 = 4街×每街最多6动作；4行 = 我/对手/两者之和/当前合法
与 Java 版 StateEncoder 完全一致，便于日后 ONNX 推理复用。
"""
import numpy as np
from engine.cards import card_rank, card_suit
import config
from config import (NB, MAX_STEP, ACTION_CHANNELS as ACTION_CH,
                    NUM_SCALAR, STACK_NORM, SPR_NORM)

CARD_CH = 6
SUITS = 4
RANKS = 13


def _set_card(buf, ch, card):
    buf[ch, card_suit(card), card_rank(card)] = 1.0


def encode_cards(state) -> np.ndarray:
    buf = np.zeros((CARD_CH, SUITS, RANKS), dtype=np.float32)
    hole = state["hole"]
    board = state["board"]
    for c in hole:
        _set_card(buf, 0, c)
    flop, turn, river = board[:3], board[3:4], board[4:5]
    for c in flop:
        _set_card(buf, 1, c)
    for c in turn:
        _set_card(buf, 2, c)
    for c in river:
        _set_card(buf, 3, c)
    for c in board:                 # ch4 全部公共牌
        _set_card(buf, 4, c)
    for c in hole + board:          # ch5 全部牌
        _set_card(buf, 5, c)
    return buf


def encode_opp_channel(opp_hole) -> np.ndarray:
    """【Stage 2 特权评论家】把对手底牌编成额外 1 个通道 (1,4,13)。
    仅训练期喂给特权评论家；actor 永远看不到这个通道。"""
    extra = np.zeros((1, SUITS, RANKS), dtype=np.float32)
    for c in opp_hole:
        _set_card(extra, 0, c)
    return extra


def encode_cards_privileged(state) -> np.ndarray:
    """(7,4,13)：前 6 通道同 actor，第 6 通道(索引6)=对手底牌。需 state['opp_hole']。"""
    return np.concatenate([encode_cards(state), encode_opp_channel(state["opp_hole"])], axis=0)


def encode_actions(state) -> np.ndarray:
    buf = np.zeros((ACTION_CH, SUITS, NB), dtype=np.float32)
    me = state["player"]
    for rnd in range(4):
        for step_i, (p, bucket) in enumerate(state["history"][rnd][:MAX_STEP]):
            ch = rnd * MAX_STEP + step_i          # 0..23
            row = 0 if p == me else 1
            buf[ch, row, bucket] = 1.0            # 行0=我 / 行1=对手
            buf[ch, 2, bucket] = 1.0              # 行2=两者之和
    legal = state["legal"]                        # 行3=当前合法动作（写在最后一通道）
    for a in range(NB):
        if legal[a]:
            buf[ACTION_CH - 1, 3, a] = 1.0
    return buf


def encode_scalars(state) -> np.ndarray:
    """有效筹码/SPR 标量特征（全部归一化到约 [0,1]）：
       [我剩余/NORM, 对手剩余/NORM, 底池/(2·NORM), 有效SPR/SPR_NORM(截断)]。
       归一化基准 NORM 随局面的 stack_norm 传入（= 训练筹码上限），使特征在任意深度都不饱和；
       缺字段时退回默认 STACK_NORM=200（向后兼容）。"""
    norm = float(state.get("stack_norm", STACK_NORM))
    my = float(state.get("my_stack", norm))
    opp = float(state.get("opp_stack", norm))
    pot = float(state.get("pot", 2.0))
    eff = min(my, opp)                         # 有效筹码 = 较少的一方
    spr = eff / max(pot, 1e-6)
    if getattr(config, "LOG_STACK_FEATURES", False):
        # 对数尺度：log1p(x)/log1p(基准)。浅筹被展开、分辨率大增(20/500 线性=0.04 → 对数≈0.49)
        def f(x, d):
            return min(np.log1p(max(x, 0.0)) / np.log1p(d), 1.5)
        return np.array([
            f(my, norm), f(opp, norm), f(pot, 2.0 * norm),
            min(spr / SPR_NORM, 1.0),
        ], dtype=np.float32)
    return np.array([
        min(my / norm, 1.5),
        min(opp / norm, 1.5),
        min(pot / (2.0 * norm), 1.5),
        min(spr / SPR_NORM, 1.0),
    ], dtype=np.float32)


def encode(state):
    """返回 (card(6,4,13), action(24,4,NB), scalar(NUM_SCALAR,), legal_mask(NB,))。"""
    legal = np.array(state["legal"], dtype=np.float32)
    return encode_cards(state), encode_actions(state), encode_scalars(state), legal
