# 阶段 0 汇报（2026-11-04，本机 CPU，无 GPU）

## 结论：Gurobi 已解决（11-04 装上学术许可证）。基线推理和拼法代码仍未完成。

> **更新（11-05）：** 下面的待办都已处理。
> - "10 个里有 3 个 = 1.0"：查明是 Gurobi 容差 bug，见 `investigate_score1/REPORT.md`。
> - 拼法代码：找到了，用在阶段 1（`stage1.py`）。
> - GPU：用 Colab（T4），见 `../docs/SETUP.md`。
> - 基线推理：不开回滚的 100 个已在 Colab 跑完并打分（原代码 24/100 稳定，修正后 30/100），见 `MORNING_REPORT.md` 第 4 节。开回滚的 100 个**还没跑**（Colab 上估计约 28 小时）。
> - CLIP 分数：仍未运行。

| 项目 | 结果 |
|---|---|
| 分支 | `pilot/BrickGPT` 上新建 `newbrick-1x3`。原始数据没改。新文件都放在 `icl/newbrick/` |
| Gurobi 版检查器 | 装上学术 WLS 许可证后能跑（第一次用 pip 自带的限制版许可证时，10/10 个结构都报 "Model too large"）。10 个 test 结构（32–239 块）每次耗时 0.97–52.8 s，中位数 3.5 s，平均 8.4 s；239 块的那个用了 52.8 s。**另外：10 个里有 3 个 max 分数 = 1.0（按代码 `is_stable` 的判定是不稳定），但数据集号称已经过稳定性过滤。原因不确定**，可能是版本或参数不同，需要排查 |
| 连通性版检查器 | 能跑，每个结构 0–11 ms（10 个结构，见 `stage0_checker_timing.json`）。但它只有连通 / 不连通，10 个结构全部是 0。**不能代替 Min / Mean Stability** |
| STABLE 指标 | 能直接复用。`STABLE/src/brickgpt/stability_analysis/interlocking_analysis.py` 已复制成 `stable_metrics.py`，在 BrickGPT 的输出上能直接跑（5 个 test 结构：interlocking 0.40–0.56，seam 0.61–0.75）。加了 1x3 的结构也能跑。注意：不开 Gurobi 时，`physics_stable` 会固定为 False、`max_physics_score` 为 inf，这只是占位，不是真实结果 |
| 随机拼法代码（split-and-remerge） | **更新：找到了**，在 `STABLE/src/mesh2brick/src/mesh2brick/voxel2brick.py`（README 写明来自 BrickGPT 原作者的 mesh2brick）。流程：贪心铺砖，然后对连通性 / 稳定性的关键区域做 split + 随机顺序 re-merge，带随机种子。能不能合并由 `brick_library.json` 决定，所以在副本里加一条 1x3 就能用。输入是体素，可以直接用数据集结构自己的体素，不需要 ShapeNet。旧记录：原先以为**不在 BrickGPT 仓库里**，仓库里只有训练和推理代码。零件库定义在 `src/brickgpt/data/brick_library.json`，运行时可以加入新尺寸（pilot 里已经加过 1x3，id 99，零件号 3622）。拼法代码需要另找（论文引用的来源 / 作者），或者我们自己重写 |
| 基线推理（开 / 不开回退） | **未运行**。之前 pilot 的实测是每个物体在 CPU 上 2–6 分钟。估算：100 个 prompt × 2 种设置 ≈ 7–20 小时。超过"几小时"，先报告 |
| CLIP 分数 | **未运行**。需要先用 Blender 渲染，本机没装 Blender |

## 为什么必须先解决 Gurobi
阶段 1 的 best-of-K 选择、阶段 4 的主要指标（Min / Mean Stability、不开回退稳定率），都要靠 Gurobi 的受力分数。连通性检查器全部输出 0，没有区分度，用它挑拼法等于随机挑。

## 需要你们决定
1. ~~Gurobi 许可证~~：已解决。许可证放在用户目录 `gurobi.lic`（WLS 类型，需要联网）。阶段 1 耗时估算：500 个形状 × K=10 × 2 组 = 1 万次检查 × 每次 3.5–8.4 s，串行约 10–23 小时，可以多进程并行（学术 WLS 是否限制并发会话：不确定）。
2. **GPU**：阶段 0 的基线和阶段 3 的 LoRA 在 CPU 上太慢（pilot 中 250 例 LoRA 用了 77 分钟；N=2000 估计要 10 小时以上，每个档位都要训）。需要课程 / 实验室的 GPU，或者 Colab。
3. **拼法代码**：找原作者要，还是我们自己写一个"随机切分 + 稳定性过滤"的版本？自己写的话，和原数据集的分布不完全一样，需要在报告里注明。

## 文件
`time_checker.py`、`stage0_checker_timing.json`、`stable_metrics.py`（复制自 STABLE）、`STABLE/`（克隆）
