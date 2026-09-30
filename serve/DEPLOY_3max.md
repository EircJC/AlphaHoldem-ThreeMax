# 3 人桌 serving 部署清单(serve/app3.py + iter885)

## 一、要打包的(全是代码 + 一个模型,很小)

在**训练机器上**(模型/代码所在处),项目根 `alphaholdem-python/` 下打包:

```bash
cd /path/to/alphaholdem-python
tar czf 3max_deploy.tgz \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='*.log' --exclude='*.bak' \
  serve/app3.py serve/bot.py serve/engine_state.py serve/selftest3.py serve/requirements.txt \
  threemax/table3_bot.py threemax/table3.py threemax/hu_bridge.py \
  threemax/charts.py threemax/preflop_policy.py threemax/multiway.py \
  engine \
  network.py encoder.py config.py play_vs_bot.py exploit_layer.py preflop_guard.py \
  preflop_table.py river_solver.py \
  runs/threemax_p3/model_iter885.pt
```

清单说明(依赖闭环,少一个都会 ImportError):
- **serving 代码**:`serve/{app3,bot,engine_state,selftest3}.py` + `requirements.txt`。
- **3 人逻辑层**:`threemax/{table3_bot,table3,hu_bridge,charts,preflop_policy,multiway}.py`
  (坍缩桥 + 翻前 charts + 多路桩;`engine3/style3/play_3max` 是测试用,不进服务)。
- **共享底座**:`engine/`(牌/评估/HUNLEnv 包)、`network.py`、`encoder.py`、`config.py`、
  `play_vs_bot.py`(bot.py 用它 load_net)、`exploit_layer.py`、`preflop_guard.py`。
- **兜底带上**:`preflop_table.py`、`river_solver.py`(是条件 import,不设 db/不开河解不会真加载,
  但带上防某些代码路径触发 ImportError;体积极小)。
- **定版模型**:`runs/threemax_p3/model_iter885.pt`(几 MB)。

## 二、【不要】打包的(大 / 用不到 / 跨平台跑不了)

- `gto_lib_wide.db`(138G,HU 库)—— **3 人坍缩走训练网、不走库,坚决不带**。
- `gto_libdir*` / `solver_data/`(库构建产物/脚本)—— 不需要。
- `TexasSolver-v0.2.0-MacOs/`(153M,且是 **Mac x86 二进制,Linux 服务器跑不了**)—— 3 人不开河解,不需要。
- `hu-data/`(677M 训练数据)、`.venv/`(1.3G,服务器上重装)、其它 `runs/`(HU 模型等)。
- HU 那套(`serve/app.py` + iter330 + 库)是**另一条独立产品线**;要不要在同机部署 HU 单独说。

## 三、服务器环境(Linux)

Python 3.10+,装依赖(CPU 推理即可,poker 决策每步 ~毫秒级):
```bash
pip install torch numpy                       # CPU 版 torch 即可;有 GPU 可装 cuda 版
pip install -r serve/requirements.txt         # fastapi / uvicorn / pydantic
```

## 四、启动(定版配置:iter885 + 不加载 HU 库)

```bash
tar xzf 3max_deploy.tgz
MODEL_PATH=runs/threemax_p3/model_iter885.pt \
python3 -m uvicorn serve.app3:app --host 0.0.0.0 --port 8001
```
- **不设 `DB_PATH`** → 坍缩决策走 iter885 训练网(不是 HU 库)。这是定版口径,勿加 db。
- 默认 `EXPLOIT=1`(与打分口径一致)、`GREEDY=0`(采样,抗剥削)。`NORM=500`。
- 有效筹码 >500bb(超深筹)未训,serving 端建议把喂进来的有效筹码夹到 ≤500 兜底。

## 五、上线前自检

```bash
# 1) 决策栈自检(无需起 HTTP,确认模型加载 + 四支路由;坍缩应显示 source:nn)
python3 serve/selftest3.py --model runs/threemax_p3/model_iter885.pt
# 2) 起服务后
curl -s localhost:8001/health
curl -s localhost:8001/decide -H 'Content-Type: application/json' -d '{
  "table_id":"t1","hand_id":"20260930-0001","hero_seat":0,"button":0,"start_stacks":[200,200,200],
  "hero_hole":["Ah","Ks"],"board":["Qs","Jh","2h"],
  "actions":[{"street":0,"actor":0,"type":"raise","to":3},
             {"street":0,"actor":1,"type":"fold"},{"street":0,"actor":2,"type":"call"}]}'
```
`/decide` 返回里 `meta.source` 应为 `nn`(坍缩走训练网)、`meta.route` 为 `collapsed_hu`。

## 六、接口一览

| 方法 | 路径 | 何时调用 | 作用 |
|---|---|---|---|
| POST | `/decide` | 轮到机器人行动时 | 返回机器人该做的动作 |
| POST | `/hand_end` | **每手结束后(必调)** | 传本手三家全部动作,更新对手读数(剥削层靠它) |
| POST | `/reset_table` | 对手换人 / 离桌时 | 清掉该桌对手读数(不调会串味) |
| GET | `/health` | 探活 | 返回 `{"ok":true}` |

> 多桌并发:每桌用不同 `table_id` → 对手读数各自独立。同一桌必须每手都调 `/hand_end`,否则剥削层学不到对手、退化为纯模型。

## 七、`/decide`(与 `/hand_end`)入参

请求体(JSON)。牌用 `"Ah"`:rank ∈ `23456789TJQKA`,suit ∈ `shdc`。

| 字段 | 类型 | 必填 | 默认 | 说明 / 备注 |
|---|---|---|---|---|
| `table_id` | string | 否 | `"default"` | 桌号;多桌用不同值隔离对手读数 |
| `hand_id` | string | 否 | `null` | 当前手牌 id(同一桌可有多手)。用途:① 出参原样回传供关联;② **`/hand_end` 去重**——同一 `(table_id,hand_id)` 重复调只计一次读数。不传则不去重(向后兼容) |
| `hero_seat` | int | **是** | — | 机器人座位,取值 `0/1/2` |
| `button` | int | **是** | — | 按钮(BTN)座位 `0/1/2`;`SB=(button+1)%3`,`BB=(button+2)%3` |
| `sb` | float | 否 | `0.5` | 小盲(bb 为单位) |
| `bb` | float | 否 | `1.0` | 大盲 |
| `start_stacks` | float[3] | **是** | — | 三家本手【下盲前】起始筹码(bb),顺序 `[seat0,seat1,seat2]`。**>500bb 超深筹未训,建议后端夹到 ≤500** |
| `hero_hole` | string[2] | **是** | — | 机器人两张底牌,如 `["Ah","Ks"]` |
| `board` | string[] | 否 | `[]` | 公共牌,`0/3/4/5` 张;张数须与最深 street 一致 |
| `actions` | Action[] | 否 | `[]` | 本手到当前为止的【有序】动作。`/hand_end` 时须含**本手全部**动作(三家) |

**Action(`actions` 数组元素):**

| 字段 | 类型 | 必填 | 说明 / 备注 |
|---|---|---|---|
| `street` | int | **是** | `0`翻前 `1`翻牌 `2`转牌 `3`河牌 |
| `actor` | int | **是** | 动作者座位 `0/1/2` |
| `type` | string | **是** | `fold` / `check` / `call` / `bet` / `raise` / `allin` |
| `to` | float | `bet/raise` 必填 | 该街**累计下注到**多少 bb(不是本次加注额);`check/call/fold/allin` 可省 |

## 八、`/decide` 出参

| 字段 | 类型 | 说明 / 备注 |
|---|---|---|
| `table_id` | string | 原样回传请求里的 `table_id`(供服务端关联 / 校验幂等) |
| `hand_id` | string \| null | 原样回传请求里的 `hand_id`(供后端把响应关联回该手) |
| `action` | string | `fold` / `check` / `call` / `raise` / `allin` —— 机器人该做的动作 |
| `to` | float \| null | 该街**累计下注到**多少 bb(`raise/call/allin`);`check/fold` 时见 `add` |
| `add` | float | 还需从筹码里**投入**多少 bb(供你在自己引擎落地)。**注:`multiway_stub` 路由不返回此字段**,后端按 `to` 自行算 |
| `bucket` | int | 内部离散动作号(调试用)。**仅 `collapsed_hu` 路由返回** |
| `meta.route` | string | `preflop_chart` / `collapsed_hu` / `multiway_stub` / `walk` —— 走了哪条支路 |
| `meta.source` | string | `nn`(训练网)/ `library` / `river_solve` —— 决策来源。**3 人定版恒为 `nn`**(坍缩走训练网) |
| `meta.opp_label` | string \| null | `tag`/`station`/`maniac`/`nit`/`null` —— 剥削层对对手的判定 |
| `meta.guards` | string[] | 本步触发的护栏,如 `["commit","air_fold","exploit"]`;无则 `[]` |
| `meta.street` | int | 决策所在街 `0/1/2/3` |

> 不同 route 的返回字段略有差异:`collapsed_hu` 最全(含 `add`/`bucket`/完整 `meta`);`preflop_chart`/`walk` 含 `add` 但无 `bucket`;`multiway_stub` 仅 `action`/`to`/`meta.route`。后端按 `action` + `to`(或 `add`)落地即可。所有 route 都回传 `table_id`/`hand_id`。

> **幂等说明**:`/decide` 默认采样出招(`GREEDY=0`,同一局面两次调用可能给不同动作)。若网络重试,**服务端应按 `(table_id, hand_id, 决策点)` 缓存首次响应、重试直接返回缓存,不要重复调 `/decide`**(bot 已回传 `table_id`/`hand_id` 供你建 key)。`/hand_end` 则由 bot 侧按 `(table_id, hand_id)` 去重(重复调不重复计读数),幂等安全。

## 九、其它接口出参

| 接口 | 入参 | 出参 |
|---|---|---|
| `POST /hand_end` | 同 `/decide` 请求体(`actions` 含全手动作) | `{"ok": true}` |
| `POST /reset_table` | `{"table_id": "t1"}` | `{"ok": true}` |
| `GET /health` | — | `{"ok": true/false}` |

## 十、示例

```bash
# /decide:BTN(hero=0)开池、SB弃、BB跟 → 翻牌坍缩成 HU
curl -s localhost:8001/decide -H 'Content-Type: application/json' -d '{
  "table_id":"t1","hand_id":"20260930-0001","hero_seat":0,"button":0,"start_stacks":[200,200,200],
  "hero_hole":["Ah","Ks"],"board":["Qs","Jh","2h"],
  "actions":[{"street":0,"actor":0,"type":"raise","to":3},
             {"street":0,"actor":1,"type":"fold"},{"street":0,"actor":2,"type":"call"}]}'
# → {"action":"raise","to":4.5,"add":4.5,"bucket":6,
#    "meta":{"source":"nn","opp_label":"unknown","guards":[],"street":1,"route":"collapsed_hu"}}

# 该手打完后(传三家全部动作):
curl -s localhost:8001/hand_end -H 'Content-Type: application/json' -d '{...本手完整 actions...}'
# 对手换人:
curl -s localhost:8001/reset_table -H 'Content-Type: application/json' -d '{"table_id":"t1"}'
```
