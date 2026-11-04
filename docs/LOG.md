# 工作日志

按日期记录做了什么、得到什么、决定了什么。细节和数字见各报告（链接在 README 的"文档索引"）。

## 2026-11-01 至 11-03：选题和 proposal 草稿
- 整理课程要求：`requirements.md`（原文）→ `spec-整理版.md`（时间线、评分）。
- 选题：BrickGPT 能否通过 in-context learning 或少量训练学会新零件。
- proposal 草稿 v1：`proposal/`（当时的设想是把相邻的砖合并成 4x4、2x8、4x6 等大砖）。

## 2026-11-04 白天：pilot（只靠 prompt）
- 在本地 CPU 上跑了 BrickGPT（Llama-3.2-1B + 官方 LoRA），运行时把 1x3 加进积木库。
- 结果：只靠 prompt 或 few-shot，模型选 1x3 的概率约 0.1%；训练一个小 LoRA（150 例）后约 9–11%。→ `pilot/RESULTS.md`
- 结论：必须训练，纯 prompting 不再投入（实验 spec 也这样要求）。

## 2026-11-04 白天：实验 spec 和阶段 0
- 实验 spec：`lego_new_brick_prompt.md`（阶段 0–4、Go/No-Go 规则、"不许编造数字"）。
- 阶段 0：拿到 Gurobi 学术许可证；测了检查器速度；确认 STABLE 的指标能复用；找到原作者的拼法代码（在 STABLE 仓库的 mesh2brick 里）。→ `newbrick/STAGE0_REPORT.md`
- 发现数据集里一部分"稳定"结构被打成 1.0，查明是 Gurobi 容差 bug。→ `newbrick/investigate_score1/REPORT.md`
- 搭好 Colab 笔记本（T4 GPU），跑通计时：不开回滚每个物体约 30 秒；开回滚最慢的一个要 49 分钟。

## 2026-11-04 下午：阶段 1（1x3）
- 筛选 300 个形状，取最难的 40 个；A（原 8 种砖）和 B（+1x3）各 440 个拼法。
- 结果：稳定性没有显著差别，只有用砖数显著变少。停在 Go/No-Go。→ `newbrick/STAGE1_REPORT.md`

## 2026-11-04 晚（通宵）：四组实验
1. 补测 2x3、1x10、1x12：都没让结构更稳；长砖在"拿掉一块砖"测试里更差。→ `STAGE1_REPORT.md`"补充实验"一节
2. bug 影响面：500 个数据集结构里，6.8% 被误判。
3. BrickGPT 生成，原代码 vs 修正后（开回滚、同一种子，12 个物体）：2 个的结果被 bug 改变。
4. Colab 基线：BrickGPT 不开回滚，生成 100 个物体。
→ `newbrick/MORNING_REPORT.md`

## 2026-11-05
- 从 Drive 取回 Colab 的生成结果并打分：原代码 24/100 稳定，修正后 30/100；65/100 有悬空的砖。→ `MORNING_REPORT.md` 第 4 节（脚本 `newbrick/score_colab.py`）
- 查了 BrickGPT 的 GitHub：17 个 issue、6 个 PR 里没有人报告过容差 bug（issue 16 讲的是论文分数等于"1 减代码分数"，和这个 bug 无关）。
- 确认 BrickGPT 的格式和稳定性检查**只支持高度一格的长方形砖**（每块砖只有长、宽、朝向）。所以长方形新砖只是换一种切法；转角砖、薄板、拱砖这类常用零件都需要改检查代码。→ `docs/NEXT_STEPS.md`
- **决定：** bug 不当项目主题（风险大，也偏离 ICL 主题）。修正后的打分只在评测里使用，加脚注说明。
- **决定：** 北大未名一号用不了；GPU 只考虑 Colab（免费版或 Pro）。未名教学二号只有 CPU。
- 建了公开仓库，写好 README、`docs/SETUP.md`、`docs/NEXT_STEPS.md`、`docs/TEAM_UPDATE.md`，以及本日志。
- **待全队决定：** 方向选 A（省砖，1x3 或 2x3）还是 B（2x2 转角砖，要改检查代码）。proposal 11 月 12 日截止。
- 讨论"是不是选的零件太奇怪"：查了零件目录。普通的 4x4 方砖不是标准件（常见的 4x4 是薄板 3031、转角砖、圆砖）。常用但没测过的长方形砖有 2x8（3007）、2x10（3006）、4x6（2356，2010 年停产）。
- `stage1.py` 改成一组可以加多种零件；开始补测 2x8、2x10、4x6，以及"所有新尺寸一起加"（Ball 组，共 7 种）。同样 40 个难形状，每组 440 个拼法，预计约 4 小时（`run_parts2.sh`）。**结果未出。**
- 新零件筛选（`docs/BRICK_SCREENING.md`）：在产的普通砖共 16 种，BrickGPT 缺 8 种。按常见程度，缺的砖里前三名是 1x3（近期排名 39，和 2x4 的 33 差不多）、2x2 转角砖（66）、2x3（104）；2x8 及以下明显少见。用合并法造训练数据：1x3 平均每个结构 4.9 处，2x3 3.8 处。结论：最早选的 1x3、2x3 正是最常见的两种。
