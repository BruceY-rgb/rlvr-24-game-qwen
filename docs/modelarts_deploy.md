# ModelArts 昇腾部署与评估手册（B 同学：评估 / 基线 / 消融 / 幻觉检测）

本手册面向 **B 同学**的任务：在 ModelArts 昇腾（Ascend NPU）上完成
全量评估、精确搜索基线对比、消融实验、unsolvable 幻觉检测。
命令均基于本仓库实际代码（`src/twentyfour_rl/`、`scripts/`）。

---

## 0. 你手上的产物 & 关键事实

- `checkpoint-400/` 是一个 **LoRA adapter（不是完整模型）**，基座为
  `Qwen/Qwen2.5-1.5B-Instruct`（见 `checkpoint-400/adapter_config.json`）。
  评估时必须 **同时给基座 + adapter**：`--model-path Qwen/Qwen2.5-1.5B-Instruct --adapter-path checkpoint-400`。
- 该 adapter 用 **TRL `GRPOTrainer`** 训练，约 400 step（≈0.33 epoch），
  训练 reward 从 0.026 → 0.40，`training_args.bin` 里设备为 `npu`，
  即 **原训练就在昇腾上完成**。
- ⚠️ 训练日志里 `completions/mean_length` 恒为 256（=上限），模型不太会主动输出
  EOS。评估默认 `--max-new-tokens 256`，`<answer>` 可能被截断、压低正确率。
  若发现正确率异常低，先把 `--max-new-tokens` 调到 384/512 复测。

### 可评估的数据 split（`data/processed/`）

| split | 文件 | 样本数 | 说明 |
|---|---|---|---|
| dev | `dev.jsonl` | 136 | 同分布开发集 |
| ood_test | `ood_test.jsonl` | 136 | ToT 风格保留集 |
| hard_test | `hard_test.jsonl` | 100 | 难题集 |
| unsolvable | `unsolvable.jsonl` | 100 | 无解集（幻觉检测用） |
| countdown_ood | `countdown_ood.jsonl` | 2000 | 3–4 数任意目标 OOD |
| countdown_unique | `countdown_unique.jsonl` | 2000 | 更严格 OOD |

> ⚠️ 任务表写"7 个 split"，但仓库实际只有上面 **6 个可评估**。差的那 1 个
> 请先找组里确认（可能把 base 模型或 train 集也算作一个对照条件）。

---

## 1. 前置：OBS 上传

ModelArts 的数据/代码出入口是 OBS（对象存储）。先建桶，按下面布局上传整个仓库：

```text
obs://<your-bucket>/24game/code/      # 整个 repo：src/ scripts/ data/ tests/ requirements*.txt pyproject.toml
obs://<your-bucket>/24game/models/    # checkpoint-400/  （那个 LoRA adapter 目录）
obs://<your-bucket>/24game/outputs/   # 评估/训练结果回传
```

上传方式任选其一：

- **OBS Browser+** 客户端拖拽（最直观）。
- 命令行 `obsutil`：
  ```bash
  obsutil cp -r ./ obs://<your-bucket>/24game/code/ -f -r
  obsutil cp -r checkpoint-400/ obs://<your-bucket>/24game/models/checkpoint-400/ -f -r
  ```

`data/processed/*.jsonl` 已在仓库里，随 `code/` 一起传即可；也可单独放 `data/`。

---

## 2. 方式 A：Notebook —— 用于全量评估 / solver 基线 / 幻觉检测

评估是交互式、反复看结果的工作，用 Notebook 最顺手。

1. **ModelArts 控制台 → 开发环境 → Notebook → 创建**。
2. **镜像**：选**昇腾 PyTorch 公共镜像**（内置 `torch` + `torch-npu` + CANN 已版本对齐）。
   **规格**：选 Ascend（如 snt9b / 910）。
3. **存储**：挂载 `obs://<your-bucket>/24game/`（默认挂到 `/home/ma-user/work`）。
4. 打开 Notebook 的 **Terminal**：
   ```bash
   cd /home/ma-user/work/24game/code
   pip install -r requirements-modelarts.txt    # ⚠️ 切勿安装 torch / torch-npu
   export PYTHONPATH=$PWD/src
   python -m unittest discover -s tests          # 冒烟测试，应全过
   ```
   > ⚠️ **铁律**：不要 `pip install torch` 或 `pip install torch-npu`，会破坏镜像里
   > CANN 的版本匹配（`requirements-modelarts.txt` 已故意不含 torch）。
5. （首次）若 `data/processed/` 不全，可重建数据：
   ```bash
   python -m twentyfour_rl.data --output-dir data/processed --include-countdown --allow-synthetic
   ```
6. 之后按 **第 4 节** 的命令跑评估。

---

## 3. 方式 B：训练作业 —— 用于消融实验的 3 次重训

消融需要从基座重新训练 3 个 adapter（分别关掉一个 reward 分量），算力消耗大，
用训练作业更合适。

1. **ModelArts → 训练管理 → 创建训练作业**。
2. **创建方式**：自定义镜像（推荐）或预置框架 PyTorch(Ascend)。
3. **代码目录**：`obs://<your-bucket>/24game/code/`。
4. **启动命令 / 启动文件**：
   - 自定义镜像最简单 —— 启动命令设为：
     ```bash
     bash scripts/modelarts_bootstrap.sh
     ```
     该脚本已封装：装依赖 → 校验 torch_npu → 建数据 → 调用 `train_grpo`。
   - 预置框架要求启动文件是 `.py` —— 需要一个把 `src/` 加进 `sys.path` 再调
     `twentyfour_rl.train_grpo.main()` 的小启动器（仓库暂无，可让维护者补）。
5. **环境变量**（`modelarts_bootstrap.sh` 通过它们控制训练）：

   | 变量 | 默认 | 说明 |
   |---|---|---|
   | `TRAINER` | `trl` | 保持 `trl` 与原 checkpoint 一致；昇腾不兼容时退 `manual` |
   | `MODEL_NAME_OR_PATH` | `Qwen/Qwen2.5-1.5B-Instruct` | 基座 |
   | `OUTPUT_DIR` | `outputs/qwen24-grpo` | 每次消融换不同目录 |
   | `NUM_GENERATIONS` | `4` | 与原 run 一致 |
   | `MAX_STEPS` | `-1` | 复现可设 `400` 与原 checkpoint 对齐 |
   | `INCLUDE_COUNTDOWN` | `1` | 是否带 countdown 数据 |

6. **资源**：选 Ascend 规格；**训练输出**回写 `obs://<your-bucket>/24game/outputs/`。

> 复现原训练的关键超参（来自 `train_grpo.py` 默认值，与 checkpoint 一致）：
> lr `5e-6`、`num_generations 4`、`max_completion_length 256`、
> `gradient_accumulation_steps 4`、LoRA `r=16, alpha=32`、约 400 step。
> 消融时**只改 reward，其余全部保持不变**，对比才公平。

---

## 4. 评估命令清单（任务①②③）

设置一次：
```bash
export PYTHONPATH=$PWD/src
```

为得到有意义的对比，**同一套 split 跑 3 种条件**：base（无 adapter）/ trained（带 checkpoint-400）/ solver。

### 4.1 一键跑 6 个 split（trained 模型，pass@k 用 k=4）

`scripts/run_eval_suite.sh` 已串好多 split：
```bash
ADAPTER_PATH=../models/checkpoint-400 \
MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct \
MODEL_NAME=grpo K=4 OUT_DIR=outputs/eval_grpo \
bash scripts/run_eval_suite.sh
```
（`ADAPTER_PATH` 用 checkpoint-400 在你机器上的实际路径。）

### 4.2 base 模型对照（证明 RL 真有提升）

同上去掉 adapter：
```bash
MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct \
MODEL_NAME=base K=4 OUT_DIR=outputs/eval_base \
ADAPTER_PATH= bash scripts/run_eval_suite.sh   # ADAPTER_PATH 留空
```
若脚本对空 adapter 报错，则逐 split 手跑、不带 `--adapter-path`。

### 4.3 精确搜索基线（任务②，逐 split）

`eval.py` 内置 `--solver-baseline`（用 `solver.py` 的 `solve_24`），确定性，k=1：
```bash
for s in dev ood_test hard_test unsolvable countdown_ood countdown_unique; do
  python -m twentyfour_rl.eval \
    --dataset data/processed/$s.jsonl \
    --output outputs/eval_solver/$s.jsonl \
    --metrics-output outputs/eval_solver/${s}_metrics.json \
    --model-name exact-search --solver-baseline --k 1
done
```
solver 给出每个 split 的"可解上界"，用来衡量模型离最优有多远。

### 4.4 幻觉检测（任务③，已内置）

无需额外脚本：`eval.py` 的 `aggregate_metrics` 会在含 `solvable=false` 的 split
（即 `unsolvable`）上自动算 **`hallucination_rate`** = 无解题里模型仍硬编出非空
`<answer>` 的比例。跑完 4.1 后直接看：
```bash
cat outputs/eval_grpo/unsolvable_metrics.json   # 看 hallucination_rate
```
定性分析（看模型瞎编了什么）：
```bash
python -c "import json;[print(r['numbers'],'->',r['answer']) for r in (json.loads(l) for l in open('outputs/eval_grpo/unsolvable.jsonl')) if not r['solvable'] and r['answer']]"
```

### 单条命令模板（备查）

```bash
python -m twentyfour_rl.eval \
  --dataset data/processed/<split>.jsonl \
  --output outputs/<cond>/<split>.jsonl \
  --metrics-output outputs/<cond>/<split>_metrics.json \
  --model-name <grpo|base|exact-search> \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path checkpoint-400 \   # base/solver 不加
  --k 4 \                            # solver 用 1
  --max-new-tokens 256               # 若疑似截断，调 384/512
```

---

## 5. 消融实验（任务④）—— 需先合入一处代码改动

**现状**：`rewards.py` 的权重 `DEFAULT_REWARD_WEIGHTS` 写死，`grpo_reward_func`
永远用全量权重，`train_grpo.py` 没有控制权重的入口。**因此无法直接消融，必须先改代码。**

### 5.1 需要的最小改动（建议）

让 `grpo_reward_func` 能读到一组权重（可走环境变量 `RLVR_DISABLE`，
或给 `train_grpo.py` 加 `--reward-weights`），用工厂函数生成 reward：

```python
# rewards.py 思路示例
def make_reward_func(weights):
    def _fn(completions, numbers=None, target=None, **kwargs):
        ...  # 同 grpo_reward_func，但 score_output(..., weights=weights)
    return _fn
```
训练时把某个分量权重置 0（`exact` 是核心可验证信号，**不要**置 0）：

| 消融组 | 置 0 的分量 | 其余权重 |
|---|---|---|
| no_format | `format=0` | exact 1.0 / legal 0.3 / closeness 0.2 |
| no_closeness | `closeness=0` | exact 1.0 / legal 0.3 / format 0.2 |
| no_legal | `legal=0` | exact 1.0 / format 0.2 / closeness 0.2 |

> 这处插桩可以让维护者/我来加；本手册先假定它以环境变量 `RLVR_DISABLE` 形式存在。

### 5.2 三次重训（训练作业，每次只改 reward）

每个消融开一个训练作业，设环境变量：
```bash
# 例：去掉 format
RLVR_DISABLE=format
TRAINER=trl
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-1.5B-Instruct
OUTPUT_DIR=outputs/ablation_no_format
NUM_GENERATIONS=4
MAX_STEPS=400          # 与原 checkpoint 对齐
```
另两组把 `RLVR_DISABLE` 换成 `closeness` / `legal`，`OUTPUT_DIR` 改名。

### 5.3 评估每个消融模型并对比

对 3 个新 adapter 各跑一遍 4.1（`ADAPTER_PATH` 指向对应 `outputs/ablation_*`，
`MODEL_NAME` 改成 `no_format` 等），比较各 split 的：

- `pass_at_1` / `pass_at_k`：正确率受哪个分量影响最大
- `legal_rate`：去掉 legal 后是否更容易出非法表达式
- `format_rate`：去掉 format 后是否破坏 `<think>/<answer>` 结构
- `hallucination_rate`：哪个分量更能抑制无解题瞎编

由此得出**每个 reward 分量的贡献**，即任务④的结论。

---

## 6. 结果回传与可视化

- 把 `outputs/` 回传 OBS：`obsutil cp -r outputs/ obs://<your-bucket>/24game/outputs/ -f -r`。
- 本地用 Streamlit 看曲线/对比/失败分析：
  ```bash
  streamlit run streamlit_app.py
  ```
  侧边栏指向 `train_metrics.jsonl` / `eval_results.jsonl` / `eval_metrics.json`。

---

## 7. 常见坑

1. **别重装 torch / torch-npu** —— 破坏 CANN 匹配，是昇腾最常见翻车点。
2. **LoRA 要带基座** —— 只给 `--adapter-path` 不给 `--model-path` 会失败。
3. **截断** —— `completions/mean_length=256` 说明模型爱写满，评估 `--max-new-tokens`
   太小会截掉 `<answer>`；正确率异常低时先调大复测。
4. **split 数量** —— 仓库只有 6 个可评估 split，"7 个"的口径先与组里对齐。
5. **消融要公平** —— 除被消融的 reward 外，trainer / 超参 / step 数必须和原 run 完全一致。
6. **大 split 慢** —— countdown 两个各 2000 条，本地 Mac(MPS) 只适合
   `--max-samples 8` 冒烟；全量放昇腾。
