"""PokerBot:把 模型(iter280)+ 宽库DB + 护栏 + 剥削层 + 河解 封装成一个可调用组件。
decide() 的分层顺序【与 play_vs_bot.play() 逐行一致】,保证行为=你验收的 +4.8 那套。

多桌:按 table_id 各维护一个 OpponentTracker(剥削层要跨手累计对手 VPIP/PFR/AFq)。
用法(见 app.py / selftest.py):
    bot = PokerBot(model_path=".../model_iter280.pt", db_path=".../gto_lib_wide.db")
    resp = bot.decide(request_json)          # 轮到机器人时调用
    bot.hand_end(table_id, full_request)     # 一手结束时调用(更新对手读数)
"""
from __future__ import annotations
import os
import platform
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "solver_data"))

import engine_state as es                                     # noqa: E402
from engine import card_str                                   # noqa: E402
from config import FOLD, CALL, ALLIN                          # noqa: E402


def _make_river_cfg(solver_dir, river_range, iters, threads):
    import river_solver as rs
    return {"solver_bin": os.path.join(solver_dir, "console_solver"),
            "resource_dir": os.path.join(solver_dir, "resources"),
            "run_prefix": ["arch", "-x86_64"] if platform.machine() == "arm64" else [],
            "ip_range": river_range or rs.DEFAULT_RANGE,
            "oop_range": river_range or rs.DEFAULT_RANGE,
            "threads": threads, "iters": iters, "accuracy": 0.5, "timeout": 120}


class PokerBot:
    def __init__(self, model_path, db_path=None, norm=500.0, device=None,
                 greedy=False, exploit=True, preflop_guard=True,
                 river_nut_guard=True, river_nut_thresh=0.97,
                 commit_guard=True, commit_min_ratio=6.0,
                 air_fold_guard=True, air_fold_min_bet=0.6,
                 river_solve=False, solver_dir="TexasSolver-v0.2.0-MacOs",
                 river_range=None, river_iters=40, river_threads=4, seed=2026,
                 station_loose_floor=0.40, tight_bluff_cut=False):
        import play_vs_bot as pvb
        from config import pick_device
        self.pvb = pvb
        self.norm = float(norm)
        self.greedy = greedy
        self.device = device or pick_device()
        self.net = pvb.load_net(model_path, self.device)
        self.net_pol = pvb.net_policy(self.net, self.device, greedy)
        # 查库
        self.flopturn = None
        if db_path:
            import gto_lib_db as gdb
            self._gdb = gdb
            self._db = gdb.LibDB(db_path)
            self._ftrng = random.Random(seed + 555)
        # 河解
        self.river_cfg = _make_river_cfg(solver_dir, river_range, river_iters, river_threads) if river_solve else None
        # 剥削层 + 每桌对手跟踪
        self.exploit = exploit
        import exploit_layer as exmod
        self._exmod = exmod
        self._trackers = {}                                   # table_id -> OpponentTracker
        self._fed_hands = set()                               # (table_id,hand_id) 已喂读数的手 → /hand_end 去重
        # 站的 VPIP 下限:HU 默认 0.40;3 人桌(collapse 桥)调用方传 0.0(VPIP 被翻前弃牌稀释)
        self.station_loose_floor = float(station_loose_floor)
        self.tight_bluff_cut = bool(tight_bluff_cut)          # 3 人桌:对紧范围对手砍开口诈唬(HU 默认关)
        # 护栏配置
        self.pf_guard = preflop_guard
        self.river_nut_guard, self.river_nut_thresh = river_nut_guard, river_nut_thresh
        self.commit_guard, self.commit_min_ratio = commit_guard, commit_min_ratio
        self.air_fold_guard, self.air_fold_min_bet = air_fold_guard, air_fold_min_bet
        import preflop_guard as pg
        self._pg = pg
        self._rng = random.Random(seed)

    def _tracker(self, table_id):
        t = self._trackers.get(table_id)
        if t is None:
            t = self._exmod.OpponentTracker(station_loose_floor=self.station_loose_floor)
            self._trackers[table_id] = t
        return t

    def _river_bucket(self, env, hero):
        import river_solver as rs
        board5 = [card_str(c) for c in env.board]
        if len(board5) != 5:
            return None
        pot_river = sum(env.committed) - sum(env.street_bet)
        eff_river = min(env.stacks[0] + env.street_bet[0], env.stacks[1] + env.street_bet[1])
        rh = [(pl == hero, act) for pl, act in env.history[3]]
        hero_is_oop = (env.button != hero)                    # 翻后非按钮先手=OOP
        hole = [card_str(c) for c in env.holes[hero]]
        return rs.river_action(board5, pot_river, eff_river, rh, hero_is_oop, hole, self.greedy, self.river_cfg)

    def decide(self, req):
        """轮到机器人时调用(HU 局面 JSON)。返回 {action,to,add,bucket,meta}。"""
        env, hero = es.build_env(req, self.norm)
        return self.decide_env(env, hero, req.get("table_id", "default"))

    def decide_env(self, env, hero, table_id="default"):
        """核心决策:给定已构造好的 HU env(hero 座位) + 桌号,跑完整分层栈,返回动作。
        3 人桥坍缩出的 env 也走这里,和 HU 完全同一套。"""
        tracker = self._tracker(table_id)
        opp_read = tracker.read() if self.exploit else None
        _is_maniac = bool(self.exploit and opp_read and opp_read["label"] == "maniac")
        st = env.street
        a, source = None, "nn"
        # ① 翻/转查库(非疯子)
        if self.flopturn_on and st in (1, 2) and not _is_maniac:
            fb, _r = self._gdb.flopturn_bucket_db(self._db, env, hero, self.greedy, self._ftrng)
            if fb is not None:
                a, source = fb, "library"
        # ② 河牌现解
        if a is None and self.river_cfg is not None and st == 3:
            rb = self._river_bucket(env, hero)
            if rb is not None:
                a, source = rb, "river_solve"
        # ③ NN 兜底
        if a is None:
            a = self.net_pol(env)
        # ④ 护栏(与 play() 同顺序)
        fired = []
        gc = [0]
        if self.pf_guard and st == 0 and not _is_maniac:
            a = self._pg.guard(env, hero, a, gc)
            if gc[0]:
                fired.append("preflop")
        nc = [0]
        if self.river_nut_guard and st == 3:
            a = self._pg.river_nut_guard_action(env, hero, a, self.river_nut_thresh, nc)
            if nc[0]:
                fired.append("river_nut")
        cc = [0]
        if self.commit_guard and st >= 1:
            a = self._pg.commit_guard_action(env, hero, a, self.commit_min_ratio, rng=self._rng, counters=cc)
            if cc[0]:
                fired.append("commit")
        af = [0]
        if self.air_fold_guard and st >= 1 and cc[0] == 0 and not _is_maniac:
            a = self._pg.air_fold_guard_action(env, hero, a, self.air_fold_min_bet, rng=self._rng, counters=af)
            if af[0]:
                fired.append("air_fold")
        # ⑤ 剥削层(最终决定权;疯子让位给模型)
        ec = {}
        if self.exploit and not _is_maniac:
            a = self._exmod.exploit_adjust(env, hero, a, opp_read, rng=self._rng, counters=ec,
                                           tight_bluff_cut=self.tight_bluff_cut)
            if ec:
                fired.append("exploit")
        out = es.bucket_to_action(env, a, hero)
        out["meta"] = {"source": source,
                       "opp_label": (opp_read["label"] if opp_read else None),
                       "guards": fired, "street": st}
        return out

    @property
    def flopturn_on(self):
        return getattr(self, "_db", None) is not None

    def hand_end(self, table_id, full_req):
        """一手结束后调用,传本手【完整】动作(双方全部),更新该桌对手读数(供下一手剥削判断)。"""
        if not self.exploit:
            return
        hand_id = full_req.get("hand_id")
        if hand_id is not None:                               # 去重:同一手重复调 /hand_end 不重复计数
            key = (table_id, hand_id)
            if key in self._fed_hands:
                return
            self._fed_hands.add(key)
            if len(self._fed_hands) > 20000:                  # 有界:超量清一半旧的
                self._fed_hands = set(list(self._fed_hands)[10000:])
        hero = int(full_req["hero_seat"])
        history = es.build_history_only(full_req)
        self._tracker(table_id).update(history, 1 - hero)     # 跟踪【对手】(1-hero)

    def reset_table(self, table_id):
        """对手换人/换桌:清掉该桌对手读数 + 该桌已喂手记录。"""
        self._trackers.pop(table_id, None)
        self._fed_hands = {k for k in self._fed_hands if k[0] != table_id}
