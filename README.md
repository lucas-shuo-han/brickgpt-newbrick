# BrickGPT 新零件项目（CS ICL 期末项目）

**研究问题：** BrickGPT（LLaMA-3.2-1B + LoRA，根据文字描述生成乐高积木结构）能不能只用少量样本学会一种它没见过的新零件？加了新零件之后，结构会不会更好？需要多少样本？

**现状（2026-11-05）：** 阶段 0 和阶段 1 已完成，停在阶段 1 的 Go / No-Go。新零件**没有让结构更稳**，只让用砖数变少。需要全队决定方向，见 [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md)。
Proposal **11 月 12 日截止**（CMT + Gradescope），草稿在 [proposal/](proposal/)。

> 给队友的同步消息：[docs/TEAM_UPDATE.md](docs/TEAM_UPDATE.md)

---

## 主要结果

所有数字都来自仓库里的输出文件，出处写在各报告里。

### 1. 新零件对稳定性没有显著帮助（阶段 1）
40 个"难形状"，每种零件各 440 个拼法（10 个随机拼法 + 1 个原算法拼法），与原 8 种砖配对比较（Wilcoxon 检验）。

| | 原 8 种 | +1x3 | +2x3 | +1x10 | +1x12 |
|---|---|---|---|---|---|
| Gurobi 稳定性分数中位数（↓） | 0.064 | 0.072（p=0.94） | 0.075（p=0.91） | 0.053（p=0.72） | 0.041（p=0.086） |
| 砖数中位数（↓） | 62.5 | **60.5（p=0.0005）** | **60.5（p=0.0001）** | 61.0（p=0.15） | **59.5（p=0.001）** |
| 拿掉 1 块砖后最差剩余比例（↑） | 0.969 | 0.964 | 0.965 | 0.958（**更差**，p=0.008） | 0.942（**更差**，p=0.013） |

STABLE 论文的 interlocking 和 seam 指标，四种零件都没有显著变好。
→ [newbrick/STAGE1_REPORT.md](newbrick/STAGE1_REPORT.md)

### 2. BrickGPT 的 Gurobi 稳定性检查有一个数值 bug
`stability_analysis.py` 要求力的残差**严格等于 0**（`.X > 0`），但 Gurobi 会返回 1e-14 量级的浮点噪声，于是一些稳定的结构被整体判成 1.0（不稳定）。我们的修正：容差改成 1e-9，见 `newbrick/tolerant_stability.py`（运行时替换，不改原文件）。

| 测试 | 原代码 | 修正后 |
|---|---|---|
| 500 个数据集结构，判为不稳定 | 44（8.8%） | 10（2.0%） |
| BrickGPT 生成 100 个（不开回滚），判为稳定 | 24 | 30 |
| BrickGPT 生成 12 个（开回滚，同一随机种子），结果不同 | — | 2 / 12 |

在 BrickGPT 的 GitHub issue 和 PR 里没有人报告过这个问题（2026-11-05 查过）；论文没有系统查过。
→ [newbrick/investigate_score1/REPORT.md](newbrick/investigate_score1/REPORT.md)、[newbrick/MORNING_REPORT.md](newbrick/MORNING_REPORT.md)

### 3. 更早的 pilot：只靠 prompt，模型几乎不用新零件
few-shot、加描述、把 1x3 加进允许列表，模型选 1x3 的概率都约 0.1%；训练一个小 LoRA（150 例）后升到约 9–11%。
→ [pilot/RESULTS.md](pilot/RESULTS.md)

---

## 仓库结构

```
lego_new_brick_prompt.md   实验 spec（阶段 0–4、Go/No-Go 规则、"不许编造数字"）
requirements.md            课程要求原文（ICL 主题帖 + 行政帖）
spec-整理版.md             课程要求整理版（时间线、评分）
proposal/                  proposal 草稿（LaTeX + PDF）
pilot/                     早期 pilot：prompt / few-shot / 小 LoRA 探针
  common.py                  公共工具：加载模型、运行时加 1x3、把 1x1+1x2 合并成 1x3
  probe*.py, gen.py, lora_tiny.py, results_*.json, RESULTS.md
newbrick/                  阶段 0 起的正式实验
  STAGE0_REPORT.md           阶段 0：环境、Gurobi 计时、可复用的 STABLE 指标
  STAGE1_REPORT.md           阶段 1：新零件物理可行性（1x3 + 补充 2x3/1x10/1x12）
  MORNING_REPORT.md          11-04 通宵实验汇总
  investigate_score1/        Gurobi bug 的调查
  stage1.py                  拼法 + 打分（STABLE 的 mesh2brick 拼法算法 + Gurobi）
  analyze_ab.py              A/B 配对统计
  perturb_metric.py          扰动测试（不用 Gurobi 的独立稳定性检查）
  tolerant_stability.py      Gurobi 容差修正
  bug_impact.py, gen_bug.py, compare_gen.py, score_colab.py   bug 相关实验
  build_stage2_data.py       阶段 2 数据构建（还没正式跑）
  colab/                     Colab 笔记本（GPU 生成 + LoRA 训练）
  stage1_out/, bug_out/, colab_out/, stage2_data/   实验输出
docs/
  SETUP.md                   环境搭建
  NEXT_STEPS.md              待决定的事
  TEAM_UPDATE.md             给队友的同步
```

## 怎么复现
环境搭建见 [docs/SETUP.md](docs/SETUP.md)。常用命令（在 `newbrick/` 下）：

```bash
python stage1.py A 40 10 stage1_out/ab_A_jit_0.jsonl     # 拼法 + 打分；环境变量 JITTER=1、TOL_FIX=1、SHAPES=stage1_out/hard_shapes.json
python analyze_ab.py B2x3                                 # A 组 vs 某个新零件组
python bug_impact.py 500 10                               # 500 个数据集结构，原代码 vs 修正后
python compare_gen.py                                     # BrickGPT 生成：原代码 vs 修正后
python score_colab.py colab_out/gen.jsonl 12              # 给 Colab 生成的结构打分
```
