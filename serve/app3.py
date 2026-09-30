"""HTTP 服务(3 人桌):把 ThreeMaxBot(坍缩桥 + HU 决策栈 iter330)暴露成跨语言、多桌可调用接口。

与 HU 版 serve/app.py 平行,但输入是【3 人局面】(3 个座位、button=BTN)。内部按局面分流:
翻前→charts;翻前收成 2 人→坍缩成 HU 交 iter330(强);≥3 人进翻牌→多路启发式兜底(弱);只剩 1 人→walk。

启动(配置走环境变量):
    MODEL_PATH=runs/spec_mid_L4A6/model_iter330.pt DB_PATH=gto_lib_wide.db \
    uvicorn serve.app3:app --host 0.0.0.0 --port 8001 --workers 1

可选环境变量:NORM=500  GREEDY=0  EXPLOIT=1  PREFLOP_GUARD=1
    RIVER_SOLVE=0  SOLVER_DIR=TexasSolver-v0.2.0-MacOs  RIVER_ITERS=40
注意:3 人桌里 HU 模型 norm 500 有效覆盖 ≤500bb;超深筹(1000bb+)未训,serving 端应把有效筹码夹到 ≤500 兜底。
    剥削层站的 VPIP 下限在 3 人桌固定用 0.0(HU 的 0.40 会因翻前弃牌稀释而漏判真站)。

接口:
    POST /decide      body=3 人局面JSON(见下 DecideReq3) → {action,to,add?,meta:{route,...}}
    POST /hand_end    body=本手完整动作JSON(三家全部,需 table_id/hero_seat/actions) → 更新对手读数
    POST /reset_table body={"table_id":...} → 对手换人时清读数
    GET  /health
"""
from __future__ import annotations
import os
import sys
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "threemax")):
    sys.path.insert(0, _p)

from bot import PokerBot                                      # noqa: E402  serve/bot.py:HU 决策栈
from table3_bot import ThreeMaxBot                            # noqa: E402  3 人坍缩桥驱动


def _b(name, default):
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")


app = FastAPI(title="ThreeMaxBot serve (3-max)", version="1.0")
_bot: Optional[ThreeMaxBot] = None
_lock = threading.Lock()


@app.on_event("startup")
def _startup():
    global _bot
    mp = os.environ.get("MODEL_PATH")
    if not mp:
        raise RuntimeError("必须设 MODEL_PATH 环境变量(模型 .pt 路径)")
    norm = float(os.environ.get("NORM", "500"))
    pb = PokerBot(
        model_path=mp,
        db_path=os.environ.get("DB_PATH") or None,
        norm=norm,
        greedy=_b("GREEDY", False),
        exploit=_b("EXPLOIT", True),
        preflop_guard=_b("PREFLOP_GUARD", True),
        river_solve=_b("RIVER_SOLVE", False),
        solver_dir=os.environ.get("SOLVER_DIR", "TexasSolver-v0.2.0-MacOs"),
        river_iters=int(os.environ.get("RIVER_ITERS", "40")),
        station_loose_floor=0.0,                              # 3 人桌:放开站的 VPIP 下限(与 play_3max 一致)
    )
    _bot = ThreeMaxBot(pokerbot=pb, norm=norm)
    print(f"[serve3] 模型={mp} 库={os.environ.get('DB_PATH')} 河解={_b('RIVER_SOLVE', False)} 3人桌就绪")


class Action(BaseModel):
    street: int
    actor: int
    type: str
    to: Optional[float] = None


class DecideReq3(BaseModel):
    table_id: str = "default"
    hand_id: Optional[str] = None                            # 当前手牌 id(同桌可有多手);用于关联 + /hand_end 去重
    hero_seat: int                                           # 0/1/2
    button: int                                              # 0/1/2 = BTN(SB=btn+1, BB=btn+2)
    sb: float = 0.5
    bb: float = 1.0
    start_stacks: List[float]                                # [s0,s1,s2],本手下盲前
    hero_hole: List[str]
    board: List[str] = []
    actions: List[Action] = []


class TableReq(BaseModel):
    table_id: str


def _validate(req: DecideReq3):
    if len(req.start_stacks) != 3:
        raise HTTPException(400, "start_stacks 必须是 3 个座位")
    if req.hero_seat not in (0, 1, 2) or req.button not in (0, 1, 2):
        raise HTTPException(400, "hero_seat / button 必须 ∈ {0,1,2}")
    if len(req.hero_hole) != 2:
        raise HTTPException(400, "hero_hole 必须 2 张")


@app.get("/health")
def health():
    return {"ok": _bot is not None}


@app.post("/decide")
def decide(req: DecideReq3):
    if _bot is None:
        raise HTTPException(503, "bot 未就绪")
    _validate(req)
    body: Dict[str, Any] = req.dict()
    with _lock:
        try:
            out = _bot.decide(body)
        except Exception as e:
            raise HTTPException(400, f"decide 失败: {type(e).__name__}: {e}")
    out["table_id"] = req.table_id                           # 回传桌号+手牌id 供服务端关联/校验幂等
    out["hand_id"] = req.hand_id
    return out


@app.post("/hand_end")
def hand_end(req: DecideReq3):
    if _bot is None:
        raise HTTPException(503, "bot 未就绪")
    with _lock:
        _bot.hand_end(req.dict())
    return {"ok": True}


@app.post("/reset_table")
def reset_table(req: TableReq):
    if _bot is None:
        raise HTTPException(503, "bot 未就绪")
    with _lock:
        _bot.reset_table(req.table_id)
    return {"ok": True}
