"""3 人桌 serving 自测。

无 torch(只验路由/桥/多路,pokerbot=None 时坍缩支返回诊断):
    python3 serve/selftest3.py
真跑决策(加载模型 + 库,坍缩支真出招):
    python3 serve/selftest3.py --model runs/spec_mid_L4A6/model_iter330.pt --db gto_lib_wide.db
"""
from __future__ import annotations
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "threemax")):
    sys.path.insert(0, _p)

from table3_bot import ThreeMaxBot, _hist_from_actions        # noqa: E402


def _pf():
    return {"table_id": "t1", "hero_seat": 0, "button": 0, "start_stacks": [200, 200, 200],
            "hero_hole": ["Ah", "Ks"], "board": [], "actions": []}


def _collapsed():
    return {"table_id": "t1", "hero_seat": 0, "button": 0, "start_stacks": [200, 200, 200],
            "hero_hole": ["Ah", "Ks"], "board": ["Qs", "Jh", "2h"],
            "actions": [{"street": 0, "actor": 0, "type": "raise", "to": 3},
                        {"street": 0, "actor": 1, "type": "fold"},
                        {"street": 0, "actor": 2, "type": "call"}]}


def _multiway():
    return {"table_id": "t1", "hero_seat": 0, "button": 0, "start_stacks": [200, 200, 200],
            "hero_hole": ["Ah", "Qd"], "board": ["Ks", "7c", "2s"],
            "actions": [{"street": 0, "actor": 0, "type": "call"},
                        {"street": 0, "actor": 1, "type": "call"},
                        {"street": 0, "actor": 2, "type": "check"},
                        {"street": 1, "actor": 1, "type": "check"},
                        {"street": 1, "actor": 2, "type": "bet", "to": 5}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--norm", type=float, default=500.0)
    args = ap.parse_args()

    pb = None
    if args.model:
        from bot import PokerBot
        pb = PokerBot(model_path=args.model, db_path=args.db, norm=args.norm,
                      station_loose_floor=0.0)                # 3 人桌:放开站的 VPIP 下限
    bot = ThreeMaxBot(pokerbot=pb, norm=args.norm)

    r = bot.decide(_pf())
    print("翻前 AKs BTN 开池:", r)
    assert r["action"] == "raise" and r["meta"]["route"] == "preflop_chart", r

    r = bot.decide(_collapsed())
    print("坍缩HU:", r)
    assert (r.get("route") == "collapsed_hu") or (r.get("meta", {}).get("route") == "collapsed_hu"), r
    if pb is not None:                                        # 真模型:必须给出合法动作
        assert r["action"] in ("fold", "check", "call", "raise", "allin"), r

    r = bot.decide(_multiway())
    print("多路 air 面对下注:", r)
    assert r["meta"]["route"] == "multiway_stub" and r["action"] == "fold", r

    st = _collapsed()
    bot.hand_end(st)                                          # 不崩即可(pokerbot=None 时 no-op)
    bot.reset_table("t1")
    assert _hist_from_actions(st["actions"])[0] == [(0, 2), (1, 0), (2, 1)]

    print("[selftest3] OK ——", "真模型 decide 通过" if pb is not None else "路由/桥/多路/读数 通过(无 torch)")


if __name__ == "__main__":
    main()
