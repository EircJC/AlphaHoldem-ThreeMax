"""3 人桌 —— 翻前下注引擎(独立模块,完全不依赖/不改动现有 HU 代码)。

目标:把一手 3 人牌打到"翻前结束",算清楚:
  - 谁弃牌、谁存活(survivors)
  - 每家自己投进池里的筹码(committed)
  - 底池总额(pot)
  - 死钱(dead)= 已弃牌玩家留在池里的筹码(= pot - 存活两家的 committed)
这些是"坍缩成 HU → 交现有模型跑翻后"的输入(见 hu_bridge.py)。

座位与顺序(标准 3-max):
  角色:BTN(按钮)/ SB(小盲)/ BB(大盲)。button 传入哪个座位是 BTN。
  sb = (button+1)%3, bb = (button+2)%3。
  翻前行动顺序:BTN → SB → BB(BTN 位于 BB 左侧,先动);有人加注则重新开一圈。
  翻后行动顺序:SB → BB → BTN(按钮最后,位置最好)——供 hu_bridge 做"在位=button"映射。

动作用简单三元组表示(引擎层,与 NB 离散化解耦):
  ("fold", None) / ("call", None) / ("raise", to_bb)   to_bb=本街累计下注到多少 bb。
  raise 到 ≥ 自己剩余 = 全下。

只做翻前:一旦进入翻牌,交回上层(坍缩桥或多路兜底)。
"""
from __future__ import annotations
from dataclasses import dataclass, field

EPS = 1e-9
SB_AMT = 0.5
BB_AMT = 1.0
ROLES = ("BTN", "SB", "BB")
# 翻后行动顺序中的"位置值":越大越靠后(位置越好)。SB 先动=0,BB=1,BTN 最后=2。
POSTFLOP_ORDER = {"SB": 0, "BB": 1, "BTN": 2}


@dataclass
class Player:
    seat: int
    role: str
    stack: float                # 剩余筹码(bb),下盲后
    start_stack: float          # 本手起始筹码(下盲前),供有效筹码归一化
    committed: float = 0.0       # 本手已投入(含盲注)
    street_bet: float = 0.0      # 本街已投入
    folded: bool = False
    allin: bool = False
    acted: bool = False
    pf_actions: list = field(default_factory=list)   # 翻前动作串 [("raise",6.0),("call",None)...]


class Table3:
    """3 人翻前引擎。用法:
        t = Table3(button=0, stacks=[200,200,200])
        t.run_preflop({0: policy0, 1: policy1, 2: policy2})   # policy(table, player)->action
        t.survivors(), t.pot(), t.dead(), t.is_collapsed_hu()
    policy 返回 ("fold"|"call"|"raise", to_bb 或 None)。
    """

    def __init__(self, button: int, stacks, sb=SB_AMT, bb=BB_AMT):
        assert len(stacks) == 3
        self.button = button % 3
        self.sb_seat = (self.button + 1) % 3
        self.bb_seat = (self.button + 2) % 3
        self.sb, self.bb = sb, bb
        role = {self.button: "BTN", self.sb_seat: "SB", self.bb_seat: "BB"}
        self.P = [Player(seat=i, role=role[i], stack=float(stacks[i]),
                         start_stack=float(stacks[i])) for i in range(3)]
        # 下盲
        self._post(self.sb_seat, sb)
        self._post(self.bb_seat, bb)
        self.current_bet = bb
        self.min_raise = bb
        self.aggressor = self.bb_seat
        self.done_preflop = False

    # ---------- 记账 ----------
    def _post(self, seat, amt):
        p = self.P[seat]
        amt = min(amt, p.stack)
        p.stack -= amt
        p.committed += amt
        p.street_bet += amt
        if p.stack <= EPS:
            p.allin = True

    def _put_to(self, seat, target_street_bet):
        """把该玩家本街投入抬到 target(加注/跟注共用);返回实际加投额。"""
        p = self.P[seat]
        add = min(target_street_bet - p.street_bet, p.stack)
        add = max(add, 0.0)
        p.stack -= add
        p.committed += add
        p.street_bet += add
        if p.stack <= EPS:
            p.allin = True
        return add

    # ---------- 查询 ----------
    def alive(self):
        return [p for p in self.P if not p.folded]

    def survivors(self):
        """翻前结束后仍在牌局中的座位号列表。"""
        return [p.seat for p in self.P if not p.folded]

    def pot(self):
        return sum(p.committed for p in self.P)

    def dead(self):
        """死钱 = 底池 - 存活玩家自身投入(= 已弃牌者留下的筹码)。"""
        return self.pot() - sum(self.P[s].committed for s in self.survivors())

    def is_collapsed_hu(self):
        """翻前结束且恰好剩 2 人 → 可坍缩成 HU 交现有模型。"""
        return self.done_preflop and len(self.survivors()) == 2

    def is_walk(self):
        """只剩 1 人(其余翻前全弃)→ 直接收池,不进入翻后。"""
        return self.done_preflop and len(self.survivors()) == 1

    def is_multiway(self):
        """≥3 人进翻牌 → HU 模型无效,走多路兜底。"""
        return self.done_preflop and len(self.survivors()) >= 3

    # ---------- 翻前主循环 ----------
    def _closed(self):
        act = [p for p in self.P if not p.folded and not p.allin]
        if len(self.alive()) <= 1:
            return True
        # 所有可行动者都已行动且本街投入 = 当前注额
        return all(p.acted and abs(p.street_bet - self.current_bet) < EPS for p in act)

    def run_preflop(self, policies):
        """policies: {seat: fn(table, player)->action}。跑到翻前结束。"""
        order = [(self.button + k) % 3 for k in range(3)]     # BTN, SB, BB
        ptr = 0
        guard = 0
        while not self._closed():
            guard += 1
            assert guard < 1000, "翻前循环异常(未收敛)"
            seat = order[ptr % 3]
            ptr += 1
            p = self.P[seat]
            if p.folded or p.allin:
                continue
            if len(self.alive()) <= 1:
                break
            action = policies[seat](self, p)
            self._apply(seat, action)
        self.done_preflop = True
        return self

    def replay_actions(self, actions):
        """按固定动作列表回放翻前(serving 用:动作来自请求,不是策略)。
        actions=[{"actor":s,"type":fold/check/call/bet/raise/allin,"to":bb?}...](只翻前,street==0)。"""
        for a in actions:
            if int(a.get("street", 0)) != 0:
                continue                                  # 只回放翻前
            seat = int(a["actor"]); typ = a["type"]
            if typ == "fold":
                self._apply(seat, ("fold", None))
            elif typ in ("check", "call"):
                self._apply(seat, ("call", None))
            elif typ == "allin":
                self._apply(seat, ("raise", self.P[seat].street_bet + self.P[seat].stack))
            else:                                         # bet/raise
                self._apply(seat, ("raise", float(a["to"])))
        self.done_preflop = True
        return self

    def _apply(self, seat, action):
        p = self.P[seat]
        kind = action[0]
        to_call = self.current_bet - p.street_bet
        if kind == "fold":
            if to_call <= EPS:      # 没人下注时"弃牌"退化为过牌(免得手滑弃掉免费牌)
                p.acted = True
                p.pf_actions.append(("call", None))
                return
            p.folded = True
            p.acted = True
            p.pf_actions.append(("fold", None))
            return
        if kind == "call":
            self._put_to(seat, self.current_bet)
            p.acted = True
            p.pf_actions.append(("call", None))
            return
        if kind == "raise":
            to_bb = float(action[1])
            # 合法化:至少加满 min_raise;超过剩余则全下
            max_to = p.street_bet + p.stack
            target = min(to_bb, max_to)
            min_legal = self.current_bet + self.min_raise
            if target < min_legal - EPS and target < max_to - EPS:
                target = min(min_legal, max_to)     # 不足最小加注→抬到最小合法(除非全下)
            raise_amt = target - self.current_bet
            self._put_to(seat, target)
            if raise_amt > EPS:
                self.min_raise = max(self.min_raise, raise_amt)
                self.current_bet = target
                self.aggressor = seat
                for q in self.P:                     # 加注重开一圈:其余在牌未全下者需再行动
                    if q.seat != seat and not q.folded and not q.allin:
                        q.acted = False
            p.acted = True
            p.pf_actions.append(("raise", target))
            return
        raise ValueError(f"未知动作 {action}")

    # ---------- 供 hu_bridge 用的位置工具 ----------
    def in_position_seat(self, survivors):
        """两存活者中翻后位置更靠后(BTN>BB>SB)的座位 = HU 的 button(在位方)。"""
        return max(survivors, key=lambda s: POSTFLOP_ORDER[self.P[s].role])
