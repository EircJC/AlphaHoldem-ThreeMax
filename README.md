# PokerBot 3 人桌服务 —— 部署 / 启动 / 接口文档

模型 **iter885**(3 人坍缩专精训练:从 HU 冠军 iter330 warm-start + BC-anchor,专门在坍缩局面上微调)+ 护栏 + 剥削层,
封装成跨语言、多桌可调用的 HTTP 服务(`serve/app3.py`)。默认端口 8001。与 HU 服务(`app.py`,:8000)完全独立。

**战绩**:7 风格 EV-allin 全正、部署采样口径 worst **+1.6**;未训过的**混搭桌(两个不同风格)也全正 +15~+22**。
详见 [threemax/3人训练方案.md](../threemax/3人训练方案.md)。

**核心特点:不连库、完全自包含。** 3 人局面按局面分流:翻前 charts / 翻前收成 2 人→**坍缩成 HU 交 iter885 训练网** / ≥3 人多路启发式 / 只剩 1 人 walk。

---

## 一、部署文件(最简自包含,不需要 138G 库)

```
AlphaHoldem-ThreeMax/
├── serve/            app3.py bot.py engine_state.py __init__.py requirements.txt
├── threemax/         table3_bot.py table3.py hu_bridge.py charts.py preflop_policy.py multiway.py __init__.py
├── engine/           cards.py evaluator.py hunl_env.py __init__.py
├── network.py encoder.py config.py play_vs_bot.py exploit_layer.py preflop_guard.py
│                     preflop_table.py river_solver.py           (共享底座)
└── runs/threemax_p3/model_iter885.pt                            (3 人定版模型,几 MB)
```

**不含**:HU 的 `gto_lib_wide.db`(3 人不查库)、`solver_data/`(查库层,不需要)、TexasSolver、
以及 `.npz/.jsonl/hu-data/runs 快照` 等一切训练/解算产物。运行时**只加载模型 `.pt` 一个文件**。

## 二、环境安装(Linux / macOS,Python 3.10+)

```bash
pip install torch numpy                 # CPU 版即可,决策每步 ~毫秒
pip install -r serve/requirements.txt   # fastapi / uvicorn / pydantic
```

## 三、启动命令

```bash
cd AlphaHoldem-ThreeMax     # 必须 cd 进目录(相对路径 + 模块名靠当前目录解析)

export MODEL_PATH=runs/threemax_p3/model_iter885.pt
unset DB_PATH                 # 3人不用库,顺手清掉防残留
python3 -m uvicorn serve.app3:app --host 0.0.0.0 --port 8001
```

看到 `[serve3] 模型=... 3人桌就绪` + `Uvicorn running on http://0.0.0.0:8001` 即成功。

> ⚠ **千万不要设 `DB_PATH`**。3 人坍缩走的是训练网 iter885,不走 HU 库;AlphaHoldem-ThreeMax 也没打包查库层(`solver_data/`)。
> 若环境里残留了 HU 的 `DB_PATH`(比如同一终端起过 HU),会报 `ModuleNotFoundError: gto_lib_db`。**起 3 人前先 `unset DB_PATH`,或用独立终端。**

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL_PATH` | (必填) | 模型 `.pt` 路径 = `runs/threemax_p3/model_iter885.pt` |
| `DB_PATH` | **不设** | **3 人桌不用库,保持不设**(设了会因缺查库层报错) |
| `NORM` | `500` | 归一化基准;覆盖 20~500bb |
| `GREEDY` | `0` | 0=采样(更像真人、抗剥削);1=贪心 |
| `EXPLOIT` | `1` | 剥削层(与打分口径一致) |
| `PREFLOP_GUARD` | `1` | 翻前护栏 |
| `RIVER_SOLVE` | `0` | 3 人**不用**开(河牌由训练网处理);默认关即可 |

> 覆盖 20~500bb;`start_stacks` >500bb(超深筹)未训,serving 端建议把有效筹码夹到 ≤500。

## 四、上线自检

```bash
python3 serve/selftest3.py                                      # 无 torch:路由/桥/多路
python3 serve/selftest3.py --model runs/threemax_p3/model_iter885.pt   # 真跑 decide
curl -s localhost:8001/health          # {"ok":true}
```

## 五、为什么不连库、不需要 TexasSolver

3 人桌里能跟进翻牌、坍缩下来的对手**范围比 HU 紧得多**;HU 的 GTO 库(和 iter330)按 HU 范围建,对这种紧范围会漏
(实测 collapsed −20)。**iter885 正是专门在坍缩局面(带死钱、真实 3 人范围)上训的**(collapsed 由 −20 转正),
所以 3 人坍缩**直接用训练网、不走库**。因此:
- **不设 `DB_PATH`**(不加载 138G 库,也不打包查库层);
- **不用 TexasSolver / 不开河解**(`RIVER_SOLVE=0`),河牌由训练网直接决策。

判读 `/decide` 返回的 `meta.source`:3 人坍缩应恒为 **`nn`**(训练网);若看到 `library` 说明误设了 `DB_PATH`(按第三节 `unset`)。

## 六、接口一览

| 方法 | 路径 | 何时调 | 作用 |
|---|---|---|---|
| POST | `/decide` | 轮到机器人行动 | 返回该做的动作 |
| POST | `/hand_end` | **每手结束(必调)** | 传本手三家全部动作,更新对手读数 |
| POST | `/reset_table` | 对手换人/离桌 | 清该桌对手读数 |
| GET | `/health` | 探活 | `{"ok":true}` |

## 七、入参(`/decide` 与 `/hand_end` 同体)

牌用 `"Ah"`:rank ∈ `23456789TJQKA`,suit ∈ `shdc`。**3 人桌**:座位 `0/1/2`,`start_stacks` 三个。

| 字段 | 类型 | 必填 | 默认 | 说明 / 备注 |
|---|---|---|---|---|
| `table_id` | string | 否 | `"default"` | 桌号;多桌用不同值隔离对手读数 |
| `hand_id` | string | 否 | `null` | 手牌 id(同桌可多手)。出参回传;`/hand_end` 按 `(table_id,hand_id)` 去重;不传则不去重 |
| `hero_seat` | int | **是** | — | 机器人座位 `0/1/2` |
| `button` | int | **是** | — | 按钮(BTN)座位 `0/1/2`;`SB=(button+1)%3`,`BB=(button+2)%3` |
| `sb` | float | 否 | `0.5` | 小盲(bb 为单位) |
| `bb` | float | 否 | `1.0` | 大盲 |
| `start_stacks` | float[3] | **是** | — | 三家本手**下盲前**起始筹码 `[seat0,seat1,seat2]`(bb)。>500 超深筹未训 |
| `hero_hole` | string[2] | **是** | — | 机器人两张底牌,如 `["Ah","Ks"]` |
| `board` | string[] | 否 | `[]` | 公共牌 `0/3/4/5` 张;张数须与最深 street 一致 |
| `actions` | Action[] | 否 | `[]` | 到当前为止的**有序**动作(三家);`/hand_end` 须含**本手全部**动作 |

**Action(`actions` 元素):**

| 字段 | 类型 | 必填 | 说明 / 备注 |
|---|---|---|---|
| `street` | int | **是** | `0`翻前 `1`翻牌 `2`转牌 `3`河牌 |
| `actor` | int | **是** | 动作者座位 `0/1/2` |
| `type` | string | **是** | `fold`/`check`/`call`/`bet`/`raise`/`allin` |
| `to` | float | `bet/raise` 必填 | 该街**累计下注到**多少 bb(非本次加注额);其它可省 |

## 八、出参(`/decide`)

| 字段 | 类型 | 说明 / 备注 |
|---|---|---|
| `action` | string | `fold`/`check`/`call`/`raise`/`allin` —— 机器人该做的动作 |
| `to` | float \| null | 该街**累计下注到**多少 bb;check/fold 见 `add` |
| `add` | float | 还需**投入**多少 bb(供你在自己引擎落地)。**`multiway_stub` 路由不返回此字段** |
| `bucket` | int | 内部离散动作号(调试用)。**仅 `collapsed_hu` 路由返回** |
| `meta.route` | string | `preflop_chart` / `collapsed_hu` / `multiway_stub` / `walk` —— 走了哪条支路 |
| `meta.source` | string | 坍缩恒为 `nn`(训练网);`library`/`river_solve` 在 3 人正常不出现 |
| `meta.opp_label` | string\|null | `tag`/`station`/`maniac`/`nit`/`null` —— 剥削层判定(3 人合并两对手为一个读数,近似) |
| `meta.guards` | string[] | 本步触发的护栏;无则 `[]` |
| `meta.street` | int | 决策所在街 `0/1/2/3` |
| `table_id` / `hand_id` | string | 原样回传(供服务端关联 / 校验幂等) |

> 各 route 字段略有差异:`collapsed_hu` 最全(含 `add`/`bucket`/完整 `meta`);`preflop_chart`/`walk` 含 `add` 无 `bucket`;
> `multiway_stub` 仅 `action`/`to`/`meta.route`。所有 route 都回传 `table_id`/`hand_id`。
> `/hand_end` 返回 `{"ok":true}`;`/reset_table`(body `{"table_id":"t1"}`)返回 `{"ok":true}`;`/health` 返回 `{"ok":bool}`。

> **幂等**:`/decide` 默认采样、非确定性。网络重试请服务端按 `(table_id,hand_id,决策点)` 缓存首次响应、勿重复调 bot;
> `/hand_end` 由 bot 侧按 `(table_id,hand_id)` 去重,重复调不重复计读数。

## 九、调用示例

```bash
# 探活
curl -s localhost:8001/health

# /decide —— 坍缩局面(BTN=seat0 开池、SB=seat1 弃、BB=seat2 跟 → 翻牌两人,问 hero=seat0)
curl -s localhost:8001/decide -H 'Content-Type: application/json' -d '{
  "table_id":"t1","hand_id":"20260930-0001","hero_seat":0,"button":0,
  "sb":0.5,"bb":1.0,"start_stacks":[200,200,200],
  "hero_hole":["Ah","Ks"],"board":["Qs","Jh","2h"],
  "actions":[{"street":0,"actor":0,"type":"raise","to":3},
             {"street":0,"actor":1,"type":"fold"},{"street":0,"actor":2,"type":"call"}]}'
# → {"action":"raise","to":4.5,"add":4.5,"bucket":6,
#    "meta":{"source":"nn","opp_label":"unknown","guards":[],"street":1,"route":"collapsed_hu"},
#    "table_id":"t1","hand_id":"20260930-0001"}

# 一手结束(传三家全部动作,喂对手读数)——必调
curl -s localhost:8001/hand_end -H 'Content-Type: application/json' -d '{
  "table_id":"t1","hand_id":"20260930-0001","hero_seat":0,"button":0,
  "start_stacks":[200,200,200],"hero_hole":["Ah","Ks"],"board":["Qs","Jh","2h","7c","9d"],
  "actions":[{"street":0,"actor":0,"type":"raise","to":3},{"street":0,"actor":1,"type":"fold"},
             {"street":0,"actor":2,"type":"call"},{"street":1,"actor":2,"type":"check"},
             {"street":1,"actor":0,"type":"bet","to":4},{"street":1,"actor":2,"type":"call"},
             {"street":2,"actor":2,"type":"check"},{"street":2,"actor":0,"type":"check"},
             {"street":3,"actor":2,"type":"check"},{"street":3,"actor":0,"type":"check"}]}'

# 对手换人 → 清读数
curl -s localhost:8001/reset_table -H 'Content-Type: application/json' -d '{"table_id":"t1"}'
```

## 十、多桌 & 运营 / 边界

- 每桌用不同 `table_id` → 对手读数各自独立;每手结束**必调 `/hand_end`**(传全手动作);换对手调 `/reset_table`。
- 多进程 `--workers N` 时**同一桌必须路由到同一进程**(否则对手读数分裂)——网关按 `table_id` 一致性哈希。
- **强度边界(诚实)**:只有**坍缩成 HU 的翻后**是强的(训练网);翻前是近似 charts、多路(≥3人)是"安全不倒钱"启发式。
  超深筹 >500bb 未训,建议夹到 ≤500。战绩是打这套风格 bot 的口径,真人 reg 会自适应、数字会缩水。
