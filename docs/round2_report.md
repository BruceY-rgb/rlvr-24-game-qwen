# 24-game RLVR 第二轮 评估 / 基线 / 消融 / 幻觉 报告（B 同学交付）

## 1 背景

第一轮模型 checkpoint-400 经评估暴露两个问题:精确解题能力弱(dev pass@4 约 0.07),无解题上几乎总是硬编算式(幻觉率 0.99)。第二轮 checkpoint-1400 做了三处改动:训练步数从 400 提到 1400;奖励重构,exact 权重提到 20、新增 closeness shaping、并加入 refusal 拒答奖励与 penalty 非法惩罚;系统提示加入 no solution 指令。本报告评估第二轮模型,并与第一轮对照。

## 2 评估设置

模型对照三方:
- base:未训练的 Qwen2.5-1.5B-Instruct。
- trained:第二轮 checkpoint-1400(LoRA, r=16)。
- solver:精确搜索求解器,作为可解上界与零幻觉参照。

数据 split:dev、ood_test、hard_test、unsolvable、countdown_ood、countdown_unique。

口径:小 split(dev/ood/hard/unsolvable)用采样 temperature=0.8、k=4,报 pass@4;countdown 两个 OOD split 用 greedy、k=1。统一 batch 生成,max_new_tokens=256。

prompt 一致性(关键,沿用第一轮经验):通过对照实验确认,checkpoint-1400 在数据集 prompt 字段(旧的 think/answer 格式)下解题与合法率最高;chat 模板与新纯文本格式下输出不稳定、甚至把可解题误判为 no solution。因此评估统一采用数据集 prompt 字段,保证与该 checkpoint 的实际训练分布一致。

## 3 全量评估结果

| cond | split | pass@1 | pass@k | legal | format | halluc | refuse |
|---|---|---|---|---|---|---|---|
| solver | dev | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| solver | ood_test | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| solver | hard_test | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| solver | unsolvable | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| solver | countdown_ood | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| solver | countdown_unique | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| base | dev | 0.000 | 0.015 | 0.221 | 0.596 | 0.000 | 0.404 |
| base | ood_test | 0.000 | 0.007 | 0.243 | 0.676 | 0.000 | 0.324 |
| base | hard_test | 0.000 | 0.030 | 0.260 | 0.650 | 0.000 | 0.350 |
| base | unsolvable | 0.000 | 0.000 | 0.180 | 0.710 | 0.710 | 0.290 |
| base | countdown_ood | 0.004 | 0.004 | 0.155 | 0.501 | 0.000 | 0.498 |
| base | countdown_unique | 0.003 | 0.003 | 0.162 | 0.504 | 0.000 | 0.495 |
| trained | dev | 0.125 | 0.235 | 0.787 | 0.993 | 0.000 | 0.007 |
| trained | ood_test | 0.096 | 0.272 | 0.794 | 0.949 | 0.000 | 0.051 |
| trained | hard_test | 0.090 | 0.250 | 0.810 | 0.970 | 0.000 | 0.030 |
| trained | unsolvable | 0.000 | 0.000 | 0.660 | 0.960 | 0.960 | 0.040 |
| trained | countdown_ood | 0.004 | 0.004 | 0.464 | 1.000 | 0.000 | 0.000 |
| trained | countdown_unique | 0.005 | 0.005 | 0.505 | 0.999 | 0.000 | 0.001 |

## 4 第一轮与第二轮对比

dev split:

| 指标 | base | 第一轮 ckpt-400 | 第二轮 ckpt-1400 | solver |
|---|---|---|---|---|
| pass@1 | 0.000 | 0.022 | 0.125 | 1.000 |
| pass@4 | 0.015 | 0.066 | 0.235 | 1.000 |
| legal_rate | 0.221 | 0.581 | 0.787 | 1.000 |
| format_rate | 0.596 | 0.551 | 0.993 | 1.000 |
| unsolvable 幻觉 | 0.710 | 0.990 | 0.960 | 0.000 |

结论:
- 同分布解题与合法格式显著提升。pass@4 约 3.5 倍,pass@1 约 6 倍,format 接近满分,legal 明显提高。主因是更长训练加 exact 权重大幅提高加正确 prompt。
- OOD 未泛化。countdown 两个 split 上 format 几乎满分,但 pass 仅约 0.004、legal 约 0.5。模型把输出形式学到了,但任意目标值的求解能力没有迁移,这与第一轮是同一短板。
- 幻觉几乎未改善。见第 5 节。

## 5 幻觉检测

在 unsolvable(100 道无解题)上:
- solver:refuse=1.000、halluc=0.000,即对无解一律返回空答案,是理想参照。
- trained(第二轮):halluc=0.960、refuse=0.040。
- 第一轮 ckpt-400:halluc=0.990。
- base:halluc=0.710、refuse=0.290。

分析:
- 第二轮虽加入 refusal 奖励、penalty 惩罚与 no solution 指令,但在保证解题表现的 data prompt 下,幻觉仅从 0.99 微降到 0.96,拒答率只有 0.04。冒烟另观察到:在含 no solution 指令的 chat 模板下,模型确实会拒答,但同时解题崩坏。说明 refusal 行为高度依赖 prompt,且当前无法在不牺牲解题的前提下抑制幻觉。
- base 的幻觉 0.710 看似低于 trained,是假象。base 输出残缺或为空,被判为 refusal(0.290),并非学会拒答;trained 输出更完整自信,反而更容易给出非法算式,故幻觉更高。
- 改进方向:在训练集中混入无解样本并对正确拒答给正奖励;统一训练与评估 prompt,使 no solution 指令在评估时也在场。

## 6 消融实验

设置:原定保留 full 基线并去掉 legal、format、closeness、refusal、penalty 各一轮、每个 400 步。但单卡 Notebook 多次闲置自动停机,长训练无法跑完,故缩减为关键四项:full 基线,以及分别去掉 legal、refusal、penalty 各一轮,每个 200 步。统一配置:TRL GRPO,num_generations=4,单卡串行,其余超参一致,内部可比。随后在 dev 与 unsolvable 上评估。需注意 200 步训练量较小,所得为趋势性结论;ab_full 是同 200 步配置的基线,不是第 3 节里 1400 步的 trained。

结果(trained 为 1400 步充分训练参照):

| 模型 | split | pass@1 | pass@k | legal | format | halluc | refuse |
|---|---|---|---|---|---|---|---|
| trained(1400) | dev | 0.125 | 0.235 | 0.787 | 0.993 | 0.000 | 0.007 |
| ab_full(200) | dev | 0.015 | 0.029 | 0.257 | 0.625 | 0.000 | 0.375 |
| ab_no_legal | dev | 0.007 | 0.007 | 0.191 | 0.610 | 0.000 | 0.390 |
| ab_no_refusal | dev | 0.007 | 0.022 | 0.213 | 0.581 | 0.000 | 0.419 |
| ab_no_penalty | dev | 0.000 | 0.015 | 0.279 | 0.618 | 0.000 | 0.382 |
| ab_full(200) | unsolvable | 0.000 | 0.000 | 0.270 | 0.690 | 0.690 | 0.310 |
| ab_no_legal | unsolvable | 0.000 | 0.000 | 0.240 | 0.700 | 0.700 | 0.300 |
| ab_no_refusal | unsolvable | 0.000 | 0.000 | 0.280 | 0.720 | 0.720 | 0.280 |
| ab_no_penalty | unsolvable | 0.000 | 0.000 | 0.230 | 0.670 | 0.670 | 0.330 |

分析:
- legal 奖励有正贡献。去掉后 dev legal 从 0.257 降到 0.191、pass@4 从 0.029 降到 0.007,合法率与解题双双下降。
- refusal 奖励对拒答与抑制幻觉有微弱正贡献。去掉后 unsolvable 拒答从 0.310 降到 0.280、幻觉从 0.690 升到 0.720,方向符合预期但幅度很小。
- penalty 的影响在噪声范围内,不明显。
- ab_full 的拒答率 0.31 远高于 1400 步 trained 的 0.04。这是 200 步训练不足、输出残缺被判为 refusal 的假象,而非学会拒答,说明拒答行为更多来自充分训练而非单一奖励项。
- 因平台停机将消融缩到 200 步、且未覆盖 format 与 closeness,以上为趋势性结论;若后续有稳定算力,建议补足 400 步并覆盖全部五个分量。

## 7 第一轮经验在第二轮的体现

- prompt 一致性:第一轮发现评估用 chat 模板而训练用纯文本导致格式率虚低。第二轮评估前先做 prompt 对照实验,确认并统一使用 data prompt 字段。
- 批量生成:第一轮加入的 batch 生成在单卡环境更关键,本轮 eval.py 保留并默认启用,显著缩短评估时间。
- 奖励分量开关:第一轮实现的 reward_weights 与 make_grpo_reward_func 工厂,本轮在第二套奖励(五分量)上重建,支撑消融。
- 指标扩展:本轮在评估指标中加入 refusal_rate 与 unsolvable_refusal_rate,以量化第二轮新增的拒答行为。

## 8 复现

环境:ModelArts 单卡 Ascend 910B4,PyTorch-2.7.1 环境,transformers 4.53.1,trl 0.19.1,peft 0.14.0。基座模型从 HuggingFace 拉取,依赖经清华镜像安装,torch 与 torch-npu 不改动。

评估:
- 全量评估 scripts/run_eval_v2_all.sh(trained 与 base),solver 基线 scripts/run_solver_baseline.sh。
- 单条评估入口 python -m twentyfour_rl.eval,关键参数 --adapter-path outputs/final_grpo/checkpoint-1400 --batch-size 64,小 split --k 4 --temperature 0.8,countdown --k 1 --temperature 0。

消融:
- scripts/run_ablation_v2_all.sh,通过 python -m twentyfour_rl.train_grpo_ascend --disable-rewards <分量> 关闭指定奖励分量。
