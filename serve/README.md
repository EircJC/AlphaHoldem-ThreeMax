# PokerBot 服务组件

把 **模型 iter330 + 宽库 gto_lib_wide.db + 护栏 + 剥削层 + (可选)河解** 封装成一个
**跨语言、多桌**可调用的 HTTP 服务。决策分层与 `play_vs_bot.play()` **逐行一致**,行为 = 你验收的
7 风格 +4.8/全正那套。

## 组成
- `engine_state.py` —— 局面重建(JSON→引擎快照)+ 动作换算。无 torch,可单测。
- `bot.py` —— `PokerBot` 类:加载模型/库/河解,按 `table_id` 维护对手读数,`decide()` 复刻决策栈。
- `app.py` —— FastAPI 服务(`/decide` `/hand_end` `/reset_table` `/health`)。
- `selftest.py` —— 自测。

## 安装 & 启动
```bash
pip install -r serve/requirements.txt   # fastapi/uvicorn/pydantic;torch 你环境已有
# (可选)开河解才需要:装/校验 TexasSolver —— 见下方"河解依赖"
# 启动(配置走环境变量)
MODEL_PATH=runs/spec_mid_L4A6/model_iter330.pt DB_PATH=gto_lib_wide.db \
EXPLOIT=1 PREFLOP_GUARD=1 RIVER_SOLVE=0 \
uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

### 河解依赖(仅 RIVER_SOLVE=1 需要)
河解用 TexasSolver 二进制现解河牌树。**默认关(RIVER_SOLVE=0)则完全不需要,跳过本节。**
一键装/校验(项目根目录下跑):
```bash
bash serve/setup_river_solver.sh
```
它会:定位 `console_solver` → `chmod +x` → (Apple Silicon)装/查 Rosetta → 冒烟解一手河牌确认可用。
- 项目自带 macOS 版 `TexasSolver-v0.2.0-MacOs/`(x86 二进制,arm 机经 Rosetta 跑)。
- **部署到 Linux 服务器**:mac 二进制跑不了,需从官方 Releases 下 Linux 版 `console_solver`(含 `resources/`)
  放到 `SOLVER_DIR`(官方:https://github.com/bupticybee/TexasSolver)。脚本会检测并提示。
开河解启动:`RIVER_SOLVE=1 SOLVER_DIR=TexasSolver-v0.2.0-MacOs RIVER_ITERS=40 ...`。
环境变量:`MODEL_PATH`(必填) `DB_PATH` `NORM=500` `GREEDY=0`(0=采样,更像真人、抗剥削;1=贪心)
`EXPLOIT=1` `PREFLOP_GUARD=1` `RIVER_SOLVE=0` `SOLVER_DIR` `RIVER_ITERS=40`。

## 调用契约

### POST /decide —— 轮到机器人时调用
请求(牌用 `Ah`,rank∈`23456789TJQKA`,suit∈`shdc`):
```json
{
  "table_id": "t1",
  "hand_id": "20260930-0001",
  "hero_seat": 1,
  "button": 0,
  "sb": 0.5, "bb": 1.0,
  "start_stacks": [200.0, 200.0],
  "hero_hole": ["As","Ad"],
  "board": ["Qs","Jh","2h"],
  "actions": [
    {"street":0,"actor":0,"type":"raise","to":3.0},
    {"street":0,"actor":1,"type":"call"}
  ]
}
```
- `hand_id`(可选):当前手牌 id,同桌可有多手。**出参原样回传 `table_id`/`hand_id`** 供服务端关联/校验幂等;
  `/hand_end` 按 `(table_id,hand_id)` 去重(重复调不重复计读数)。不传则不去重(向后兼容)。与 3 人 `app3` 接口一致。
- `/decide` 默认采样、非确定性:网络重试请服务端按 `(table_id,hand_id,决策点)` 缓存首次响应,勿重复调。
- `hero_seat` 机器人座位;`button` 按钮/小盲座位(翻前先动、翻后有位置)。
- `start_stacks` 本手**下盲前**起始筹码 [seat0,seat1](bb)。
- `actions` 本手到当前为止的**有序**动作;`type`∈`fold/check/call/bet/raise/allin`,
  `bet/raise` 带 `to`=该街累计下注**到**多少 bb。
- `street`:0翻前 1翻牌 2转牌 3河牌,须与 `board` 张数(0/3/4/5)一致。

响应:
```json
{
  "action": "raise",          // fold|check|call|raise|allin
  "to": 9.0,                  // 该街累计下注到多少 bb(raise/call/allin);check/fold 见字段
  "add": 6.0,                 // 还需投入多少 bb(供你在自己引擎里落地)
  "bucket": 7,                // 内部离散动作号(调试)
  "meta": {"source":"library|nn|river_solve", "opp_label":"tag|station|maniac|nit|null", "guards":["preflop","commit",...], "street":1}
}
```

### POST /hand_end —— 一手结束后调用(必须,喂对手读数)
body 同 /decide,但 `actions` 要含**本手全部**动作(双方)。剥削层靠它累计对手 VPIP/PFR/AFq。
不调用则剥削层学不到对手、退化为纯模型。

### POST /reset_table —— 对手换人/离桌
`{"table_id":"t1"}` 清掉该桌对手读数(换对手后必调,否则读数串味)。

## curl 示例
```bash
curl -s localhost:8000/decide -H 'Content-Type: application/json' -d '{
  "table_id":"t1","hand_id":"20260930-0001","hero_seat":1,"button":0,"start_stacks":[200,200],
  "hero_hole":["As","Ad"],"board":["Qs","Jh","2h"],
  "actions":[{"street":0,"actor":0,"type":"raise","to":3},{"street":0,"actor":1,"type":"call"}]}'
```

## 3 人桌 serving(`serve/app3.py`,端口 8001)
把 `ThreeMaxBot`(坍缩桥 + HU 决策栈 iter330)暴露成 3 人桌 HTTP 服务,和 HU 版平行。内部按局面分流:
翻前→charts;翻前收成 2 人→**坍缩成 HU 交 iter330(强)**;≥3 人进翻牌→多路启发式兜底(弱);只剩 1 人→walk。

启动(**定版模型 = 3 人专精训练的 iter885,且不加载 HU 库**):
```bash
MODEL_PATH=runs/threemax_p3/model_iter885.pt \
uvicorn serve.app3:app --host 0.0.0.0 --port 8001
```
- **关键:不设 `DB_PATH`** → 坍缩后的翻后决策走**训练好的 3 人网(iter885)**,而不是 HU-GTO 库(HU 库按 HU 范围,
  对 3 人紧范围会漏;iter885 正是为治这个训的)。preflop 仍 charts、多路仍桩(冻结)。
- 内部自动用 `station_loose_floor=0.0` 构造 PokerBot;`EXPLOIT=1`(与打分口径一致)。
- iter885 战绩:7 风格全正、worst +1.6(部署采样口径),详见 `threemax/3人训练方案.md`。
- **HU 产品不受影响**:HU 服务仍用 `serve/app.py` + iter330 + gto_lib_wide.db,与本服务互不干涉。
- 环境变量与 HU 版相同(`NORM/GREEDY/EXPLOIT/PREFLOP_GUARD/...`);默认采样(`GREEDY=0`,抗剥削)。

### 接口(3 人局面)
`POST /decide`:body 是 3 人局面(3 座位、`button`=BTN、`start_stacks` 三个),响应 `{action,to,add?,meta:{route,...}}`,
`meta.route ∈ {preflop_chart, collapsed_hu, multiway_stub, walk}`。
`POST /hand_end`:传本手**三家全部**动作,更新对手读数(3 人合并两对手进一个 tracker)。
`POST /reset_table`:换对手清读数。`GET /health`。

```bash
# BTN(hero=0)开池、SB弃、BB跟 → 翻牌坍缩成 HU
curl -s localhost:8001/decide -H 'Content-Type: application/json' -d '{
  "table_id":"t1","hand_id":"20260930-0001","hero_seat":0,"button":0,"start_stacks":[200,200,200],
  "hero_hole":["Ah","Ks"],"board":["Qs","Jh","2h"],
  "actions":[{"street":0,"actor":0,"type":"raise","to":3},
             {"street":0,"actor":1,"type":"fold"},{"street":0,"actor":2,"type":"call"}]}'
```
自测:`python3 serve/selftest3.py`(无 torch 验路由)/ `--model ... --db ...`(真跑 decide)。

### 3 人桌 serving 的已知边界
- **只有坍缩成 HU 的翻后是强的**;多路(≥3人)是"安全不倒钱"启发式、翻前是手写近似 charts(非 GTO)。
- 剥削层把两对手**合并**成一个读数(两对手风格差异大时是近似);真实对手多为 reg(=tag,剥削层不介入、走基础打法)。
- **超深筹**:norm 500 有效覆盖 ≤500bb;$50-70=1000bb+ 越界,serving 端应把有效筹码夹到 ≤500 兜底。
- 压测画像见 `threemax/README.md`(F +2~+9、G ~0、N ~−17,worst 是坍缩对紧范围岩石的结构性天花板)。

## 剥削层:HU vs 3人桌的 `station_loose_floor`
`PokerBot(station_loose_floor=0.40)` 是站(跟注站)判定的 VPIP 下限。**HU 保持默认 0.40**——把"紧范围但翻后
粘"的 reg/nit(VPIP 低)排除在站外,避免误砍对它们本该有效的诈唬(曾因删掉此下限导致 HU 的 D/N 回退)。
**接 3 人桌 serving(collapse 桥)时构造 PokerBot 必须传 `station_loose_floor=0.0`**——3 人桌 VPIP 被翻前
弃牌稀释、HU 的 0.40 够不到真站,改由 `af`+`call_ratio` 判站(与 `threemax/play_3max.py` 一致)。

## 验收口径(HU 7 风格回归,改 exploit_layer/护栏后必跑)
`exploit_layer.py` 等是 HU 产品共享文件,任何改动后用**官方配置**回归,确认 7 档仍全正(iter330 基准 ~+56.5):
```bash
python3 play_vs_styles.py --depth mid --model runs/spec_mid_L4A6/model_iter330.pt \
    --hands 500000 --greedy --preflop-guard --flopturn-db gto_lib_wide.db --exploit \
    --styles A,C,D,N,E,F,G
```
注意:**必带 `--depth mid --greedy --preflop-guard`,不加 `--fixed`/`--ev-allin`**——换配置(如采样+固定200bb+
关翻前护栏)会把 reg 打成负、造成"假回退"。

## 多桌 & 并发
- 每桌用不同 `table_id` → 对手读数各自独立。
- 服务内用锁串行化推理(单 net + 每桌 tracker,保证安全)。单进程吞吐够中小并发;
  高并发要多进程 `--workers N`,但**同一桌必须路由到同一进程**(否则对手读数分裂)——
  用网关按 `table_id` 做一致性哈希,或干脆单进程 + 加大机器。

## 覆盖范围(重要)
- **模型 iter330 = norm 500,有效覆盖 20~500bb**。`start_stacks` 超过 ~500bb(如微额 $50-70=1000bb+)
  会特征越界、决策不可靠——那是**未训的超深筹**(见项目 Layer4 记录)。当前组件对超深筹不保证质量;
  serving 端可自行把喂进来的有效筹码**夹到 ≤500** 兜底,或等超深专精就绪。
- **河解**默认关。开(`RIVER_SOLVE=1`)每次河牌卡几秒、需机器装 TexasSolver;A 增益约 +6bb/100。

## 自测
```bash
python3 serve/selftest.py                                             # 无 torch:局面重建
python3 serve/selftest.py --model runs/spec_mid_L4A6/model_iter330.pt --db gto_lib_wide.db   # 真跑 decide
```
