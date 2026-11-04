# 为什么"稳定"的数据集结构 max score 会等于 1.0

（子代理调查结果，由主会话存档。所有数字都来自本目录脚本的实测。）

**结论：是数值容差 bug。** 结构本身是稳定的，求解也没有失败。

> **更新（11-04 晚 / 11-05）：** 扩大样本后的数字（详见 `../MORNING_REPORT.md` 第 2–4 节）：
> - 500 个 test 结构（`bug_impact.py`）：原代码判 44 个不稳定，修正后 10 个；**34 个（6.8%）是误判**。下面"约 15%"的估计来自 33 个的小样本，以 6.8% 为准。
> - BrickGPT 生成 100 个（不开回滚，Colab）：原代码 24 个稳定，修正后 30 个。
> - BrickGPT 生成 12 个（开回滚、同一种子，`gen_bug.py` / `compare_gen.py`）：2 个的生成结果被 bug 改变。
> - BrickGPT 的 GitHub issue / PR 里没有人报告过这个问题（11-05 查过）。

`pilot/BrickGPT/src/brickgpt/stability_analysis/stability_analysis.py` 第 502–507 行：只要某块砖的力或力矩残差 `.X > 0`（严格大于 0，没有容差），或者 `min_c <= 0`，这块砖的分数就被设成 1.0。在出问题的结构上，Gurobi 返回的是 OPTIMAL（status=2），但残差是 1e-31 到 1e-14 的浮点噪声，结果**整个结构的每块砖**都被判成 1.0。

证据：
- 被标记的砖占全部砖块：109/109、78/78、120/120、45/45、78/78。
- 这些砖的 `min_c` 都等于 T，说明拉力判据并没有触发。
- 33 次运行的 status 全部是 2（OPTIMAL）。
- 结果是确定的：每个结构重复跑 3 次，结果完全一样。
- 正常结构的残差正好是 0.0。
- 三份 `stability_analysis.py`（BrickGPT、STABLE/brickgpt、STABLE/mesh2brick）完全相同。本机 Gurobi 版本是 13.0.3。原作者的环境是否会得到精确的 0：未验证。

实测数字：
- 原始算法下，33 个 test 结构中有 5 个（15%）g_max = 1.0。
- 容差取 1e-12、1e-9 或 1e-6，都是 0/33。
- 修正后，那几个结构的 g_max 分别为 0.00563、0.00203、1.3e-10、0.00575、2.3e-10。

影响和处理：
- 原始的 `is_stable()` 和 max/mean 分数会把大约 15% 的稳定结构误判为不稳定。
- 修正方法：`newbrick/tolerant_stability.py` 在运行时把那 5 处 `.X > 0` 改为 `.X > 1e-9`，BrickGPT 和 mesh2brick 两份代码都会改到，原文件不动。
- 已验证：上面 3 个结构修正后的分数为 0.0056、0.0020、1.3e-10；jitter 试验中真正不稳定的那个拼法仍然是 1.0。

脚本：`diag.py`、`step1.py`、`step2.py`；结果：`step1_out.json`、`step2_*.json`。
