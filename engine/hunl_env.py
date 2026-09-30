"""单挑无限注德州扑克（Heads-Up No-Limit Hold'em）环境。
一手牌 = 一个 episode。奖励只在手牌结束时给出（零和，单位=bb）。

动作离散化（NB=9）：0 弃牌 / 1 过牌或跟注 / 2..7 按底池比例加注 / 8 全下。
说明：这是可训练的参考实现，单挑无边池；对最小加注、全下不足额等极端边界做了简化处理。
"""
from __future__ import annotations
import itertools
import math
import random
from .cards import Deck
from .evaluator import evaluate7
from config import NB, FOLD, CALL, ALLIN, RAISE_FRACTIONS, GameConfig

EPS = 1e-9


class HUNLEnv:
    def __init__(self, cfg: GameConfig | None = None, rng: random.Random | None = None):
        self.cfg = cfg or GameConfig()
        self.rng = rng or random.Random()

    # ---------- 生命周期 ----------
    def reset(self, button: int):
        """开一手牌。button = 小盲/按钮位（翻前先行动），另一方为大盲。"""
        c = self.cfg
        self.button = button
        self.deck = Deck(random.Random(self.rng.random()))
        self.holes = [self.deck.deal(2), self.deck.deal(2)]
        self.board: list[int] = []
        if getattr(c, "random_stacks", False):
            self.stacks = [round(self.rng.uniform(c.min_stack, c.max_stack), 1),
                           round(self.rng.uniform(c.min_stack, c.max_stack), 1)]
        else:
            self.stacks = [c.start_stack, c.start_stack]
        self.start_stacks = list(self.stacks)   # 本手起始筹码(下盲前)，供"有效筹码"归一化用
        self.committed = [0.0, 0.0]
        self.street_bet = [0.0, 0.0]
        self.all_in = [False, False]
        self.acted = [False, False]
        self.folded = None
        self.done = False
        self.winner = None
        self.payoffs = [0.0, 0.0]
        self.street = 0
        self.history = [[], [], [], []]   # 每街 (player, bucket) 列表
        # 下盲注
        self._force(button, c.small_blind)
        self._force(1 - button, c.big_blind)
        self.current_bet = c.big_blind
        self.min_raise = c.big_blind
        self.aggressor = 1 - button
        self.to_act = button              # 翻前小盲先动
        return self

    def _force(self, p, amt):
        amt = min(amt, self.stacks[p])
        self.stacks[p] -= amt
        self.committed[p] += amt
        self.street_bet[p] += amt
        if self.stacks[p] <= EPS:
            self.all_in[p] = True

    def _add(self, p, amt):
        amt = min(amt, self.stacks[p])
        self.stacks[p] -= amt
        self.committed[p] += amt
        self.street_bet[p] += amt
        if self.stacks[p] <= EPS:
            self.all_in[p] = True
        return amt

    # ---------- 合法动作 ----------
    def legal_mask(self):
        p = self.to_act
        opp = 1 - p
        mask = [False] * NB
        to_call = self.current_bet - self.street_bet[p]
        stack = self.stacks[p]
        mask[CALL] = True                       # 过牌/跟注恒合法
        if to_call > EPS:
            mask[FOLD] = True
        if stack > to_call + EPS and not self.all_in[opp]:
            pot_after_call = sum(self.committed) + min(to_call, stack)
            for a, frac in RAISE_FRACTIONS.items():
                raise_add = frac * pot_after_call
                total_add = to_call + raise_add
                if raise_add >= self.min_raise - EPS and total_add < stack - EPS:
                    mask[a] = True
            mask[ALLIN] = True                  # 全下作为激进动作
        return mask

    # ---------- 执行动作 ----------
    def step(self, action: int):
        assert not self.done, "手牌已结束"
        p = self.to_act
        opp = 1 - p
        to_call = self.current_bet - self.street_bet[p]
        self.history[self.street].append((p, action))

        if action == FOLD:
            self.folded = p
            self._settle("fold")
            return self.done, self.payoffs

        prev_current = self.current_bet
        if action == CALL:
            self._add(p, min(to_call, self.stacks[p]))
        elif action == ALLIN:
            self._add(p, self.stacks[p])
        else:  # 2..7 按底池比例加注
            frac = RAISE_FRACTIONS[action]
            pot_after_call = sum(self.committed) + to_call
            total_add = to_call + frac * pot_after_call
            self._add(p, total_add)
        self.acted[p] = True

        # 若形成加注，更新当前注额与最小加注量
        if self.street_bet[p] > prev_current + EPS:
            self.min_raise = max(self.min_raise, self.street_bet[p] - prev_current)
            self.current_bet = self.street_bet[p]
            self.aggressor = p

        # 结算本街是否结束
        if self._street_closed():
            self._advance_or_showdown()
        else:
            self.to_act = opp
            if self.all_in[self.to_act]:        # 安全：对手已全下无法行动
                if self._street_closed():
                    self._advance_or_showdown()
        return self.done, self.payoffs

    def _street_closed(self):
        if abs(self.street_bet[0] - self.street_bet[1]) < EPS and self.acted[0] and self.acted[1]:
            return True
        if self.all_in[0] and self.all_in[1]:
            return True
        if self.all_in[0] or self.all_in[1]:
            ai = 0 if self.all_in[0] else 1
            other = 1 - ai
            if self.acted[other] and self.street_bet[other] >= self.street_bet[ai] - EPS:
                return True
        return False

    def _advance_or_showdown(self):
        if self.street >= 3:
            self._settle("showdown")
            return
        if self.all_in[0] or self.all_in[1]:
            # 【Stage 1】全下且公共牌未发满：用期望权益结算(消 runout 方差)，否则回退到单次 runout。
            if getattr(self.cfg, "allin_ev", False):
                self._settle_allin_ev()
            else:
                self._run_out()
                self._settle("showdown")
            return
        self._advance_street()

    def _allin_equity(self, player: int) -> float:
        """双方底牌已知、公共牌未发满时，player 的权益 = P(赢) + 0.5·P(平)。
        策略：当"精确枚举的组合数 ≤ allin_ev_mc_samples"时直接精确枚举(转牌补1只有≤46种,恒精确)；
        否则(翻牌补2=990、翻前补5≈170万) → MC 采样 allin_ev_mc_samples 次。
        MC 只在'精确更贵'时启用，采样数默认 500 时 SE≈0.02，方差可忽略，却比翻牌全枚举快约 2 倍。"""
        opp = 1 - player
        hp, ho, board = self.holes[player], self.holes[opp], self.board
        need = 5 - len(board)
        if need <= 0:                                   # 已发满(防御)：直接摊牌
            sp, so = evaluate7(hp + board), evaluate7(ho + board)
            return 1.0 if sp > so else (0.0 if sp < so else 0.5)
        used = set(hp) | set(ho) | set(board)
        remaining = [c for c in range(52) if c not in used]
        n_mc = int(getattr(self.cfg, "allin_ev_mc_samples", 500))
        exact_combos = math.comb(len(remaining), need)
        wins = ties = total = 0
        if exact_combos <= n_mc:                        # 精确不比采样贵 → 精确枚举(0 方差)
            combos = itertools.combinations(remaining, need)
        else:                                           # 翻牌/翻前 → MC 采样
            combos = (self.rng.sample(remaining, need) for _ in range(n_mc))
        for combo in combos:
            full = board + list(combo)
            sp, so = evaluate7(hp + full), evaluate7(ho + full)
            if sp > so:
                wins += 1
            elif sp == so:
                ties += 1
            total += 1
        return (wins + 0.5 * ties) / total if total else 0.5

    def _settle_allin_ev(self):
        """按期望权益结算全下局面：E[收益] = matched·(2·equity − 1)，平局贡献 0，零和。
        公共牌保持全下时刻的张数(不发满)，避免复盘时暗示某个具体 runout；收益是对所有 runout 的期望。"""
        matched = min(self.committed)
        eq0 = self._allin_equity(0)
        ev0 = matched * (2.0 * eq0 - 1.0)
        self.payoffs = [ev0, -ev0]
        self.winner = None          # 期望结算无单一赢家
        self.done = True

    def _advance_street(self):
        self.street += 1
        if self.street == 1:
            self.board += self.deck.deal(3)
        elif self.street == 2:
            self.board += self.deck.deal(1)
        elif self.street == 3:
            self.board += self.deck.deal(1)
        self.street_bet = [0.0, 0.0]
        self.current_bet = 0.0
        self.min_raise = self.cfg.big_blind
        self.acted = [False, False]
        self.aggressor = None
        self.to_act = 1 - self.button           # 翻后大盲(非按钮)先动

    def _run_out(self):
        need = 5 - len(self.board)
        if need > 0:
            self.board += self.deck.deal(need)

    def _settle(self, reason):
        matched = min(self.committed)
        if reason == "fold":
            self.winner = 1 - self.folded
        else:
            s0 = evaluate7(self.holes[0] + self.board)
            s1 = evaluate7(self.holes[1] + self.board)
            self.winner = 0 if s0 > s1 else (1 if s1 > s0 else None)
        if self.winner is None:
            self.payoffs = [0.0, 0.0]
        else:
            self.payoffs = [0.0, 0.0]
            self.payoffs[self.winner] = matched
            self.payoffs[1 - self.winner] = -matched
        self.done = True

    # ---------- 供编码器使用 ----------
    def encoding_state(self, player: int):
        return {
            "player": player,
            "hole": self.holes[player],
            "board": list(self.board),
            "history": self.history,
            "legal": self.legal_mask() if not self.done else [False] * NB,
            # 可变筹码所需：当前剩余筹码与底池（供 encoder 计算有效筹码/SPR 标量）
            "my_stack": self.stacks[player],
            "opp_stack": self.stacks[1 - player],
            "pot": sum(self.committed),
            # 归一化基准：cfg.stack_norm>0 则用它(对齐训练口径)，否则用 max_stack(训练默认自动跟随)
            "stack_norm": (self.cfg.stack_norm if getattr(self.cfg, "stack_norm", 0) else
                           getattr(self.cfg, "max_stack", 200.0)),
        }

    # 便捷：当前玩家的收益（bb）
    def reward(self, player: int):
        return self.payoffs[player]
