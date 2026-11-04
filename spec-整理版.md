# ICL 期末项目 Spec（整理版）

> 来源：`requirements.md`（ICL 主题帖 + 行政帖 #287）。所有截止时间均为 **10:59 PM PT**，除非 #287 另有更新。

---

## 1. 时间线 & 待办（按时间排序）

| 日期 | 事项 | 备注 |
|---|---|---|
| **Nov 2 那周** | 第 1 次 1:1 会议（10 分钟，必须、计分） | 和 Matteo Guarrera 约（或 Prof. Ranade）；全队一起到；**带打印好的 proposal 草稿** |
| 会议前 | 完成 **HW9 Problem 4（Hardware Check）** | 单独提交，只上传 `hardware-results.csv` 到 "Homework 9 — Hardware Check CSV"；参考帖 #294、#292 |
| **Nov 4 (Wed)** | HW9 截止 | 若会议早于此日，P4 要在会议前做完 |
| **Nov 12 (Wed)** | **Proposal 截止**（含确定的队伍） | 提交到 Microsoft CMT **和** Gradescope；只需一人提交 |
| **Nov 16 那周** | 第 2 次 1:1 会议（10 分钟，必须、计分） | 拿反馈 |
| **Nov 23 (Mon)** | Final report **V1** 截止（用于互评） | 感恩节周 |
| **Nov 30 (Mon)** | 第 1 轮个人互评截止（CMT） | |
| **Dec 7 (Mon)** | Final report **V2** 截止（回应评审） | RRR 周 |
| **Dec 9 (Wed)** | 第 2 轮个人互评截止（CMT，新分配评审对象） | |
| TBD | Oral exam / poster session | 每位成员都要能回答项目任何部分（含代码） |
| **Dec 12 (Sat)** | **最终提交**：报告 + GitHub repo（代码 + checkpoints） | |

**会议准备清单：** 打印的 proposal 草稿（或几个备选方向）、研究问题、计划实验、可用算力。

---

## 2. 硬性规则

- **队伍 4 人**（3 或 5 人极少例外）。
- **必须训练东西**：从头训练的模型 / soft prompt 或 prefix / LoRA adapter / 大系统中的小模块 —— 要能画出 train & val 曲线。**只调 API 的项目不接受。**
- **两轮互评都必须做**，漏一次 → 项目 **0 分**。
- 项目成绩全队共享，但会 **乘以个人互评质量**。
- 每次提交附带一份自评 checklist（peer-review form）。
- 可以用生成式 AI（代码、文献检索、写作、画图），但必须：引用所用工具 + 附录说明用法（附录不计页数）。**报告中的幻觉会被严重扣分**。
- 在 ICL 主题 Ed 帖里帮助其他队可以拿参与分。

---

## 3. Proposal 要求（Nov 12 截止）

**格式：** LaTeX → PDF，1 inch 页边距，12pt 字体，**1–2 页**。

**必须包含的章节：**
1. **Team** — 4 名成员全部列出
2. **Abstract** — 最多 6 句
3. **Introduction & Background** — 至少 **8 篇**相关论文的文献综述，每篇至少一句话总结核心观点
4. **Key questions** — 你要回答的具体问题，并由上一节支撑
5. **Hypothesis** — 你预期会发生什么（结果不同也没关系，但必须有预期）
6. **Methods** — 如何验证假设、做哪些实验、预计算力

ICL 主题帖额外强调要写：基于哪些 **codebase**、**如何生成训练/测试数据**、**资源预算**（清楚自己有多少算力非常重要）。

---

## 4. Final Report 要求

- 有连贯的故事：清楚提出问题、引用相关文献、同时自成一体、分析系统。
- 引用相关工作，放进学术语境。
- 提供 **train / val loss 曲线** 并做必要对比；可以和同主题其他队共享 baseline。
- **必须有有意义的 error bars。**
- 提供 GitHub repo 链接：全部代码 + 训练好的 checkpoints，足以高效复现论文中所有图表。
- 附录：生成式 AI 使用说明。

---

## 5. 项目方向：In-Context Learning

**必读：**
- Garg et al. (2022) *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes* — 小 transformer 从头学线性函数等，匹配最小二乘；单 GPU 可跑，仍是标准测试平台。
- Wang, Shi, Fox (2025) *Test-time regression* — softmax attention、linear attention、SSM 都是在 context 中解同一个回归问题。

**参考方向（不是题目，要自己找问题）：**

| # | 方向 | 要点 / 注意事项 |
|---|---|---|
| A | 训练好的模型在 context 中跑什么算法？ | 单层 linear attention ≈ 预条件 GD；更深模型可能是 Newton / PCR / Bayes；预训练 LLM 不像 GD。kernel 化线性模型、分布偏移已有文献 —— 从已知出发找空白。**不鼓励**无明确问题地在新函数类上重跑 Garg。 |
| B | 哪些架构能做 ICL，为什么？ | associative recall、copying、state tracking 能区分 attention 和 RNN，线性回归不能。Gated DeltaNet、Mamba-3；用 `flash-linear-attention` 库。若用 nanoGPT speedrun，**固定某个版本的 recipe**，注意有些组件把信息存进权重。 |
| C | 训练选择如何影响 ICL？ | ICL 会突然出现又消失；取决于任务多样性和训练动力学。优化器（Muon vs AdamW）、参数化（muP）的影响研究很少。可与 Project (1) 结合。 |
| D | 放 context 还是放权重？ | Qwen3.5 0.8B–9B 单 GPU 可跑。many-shot vs fine-tuning vs test-time training；能否用少量训练过的 token 替代长 context？ |
| E | Agent 从 harness 构建的 context 学到什么？ | harness = 在 token 预算下决定冻结模型看什么的策略。可训练部分：context 本身（soft prompt / prefix / 压缩 token）或 harness 小模块。GEPA 等纯文本 prompt 优化只能当 **baseline**。 |
| F | 小模型能否从真实装配序列 ICL？ | LEGO LDraw 数据，每步可精确检验碰撞/连接；重复子装配；能否 in-context 学 pose estimation。 |

**其他参考论文：**

| 论文 | 一句话 |
|---|---|
| In-context Learning and Induction Heads | induction head 电路，训练中相变式出现 |
| Transformers learn in-context by gradient descent | 构造出前向即 GD 的 linear attention，训练模型接近它 |
| Do pretrained Transformers Learn In-Context by GD? | GD 观点不适用于预训练 LLM（与上篇对照读） |
| What learning algorithm is in-context learning? | 依深度/噪声表现为 GD、ridge 或最小二乘 |
| Data Distributional Properties Drive Emergent ICL | bursty + 大量稀有类 → ICL；与 in-weights 学习竞争 |
| Training Dynamics of ICL in Linear Attention | 精确学习曲线；参数化决定一次跳变还是多次 |
| Strategy Coopetition | 解释 ICL 的出现与消失 |
| Zoology | associative recall 区分 attention 与高效 RNN；附合成 benchmark |
| Gated Delta Networks | 作为在线回归求解器的循环层 |
| From Markov to Laplace | Mamba 如何 in-context 学 Markov 链最优预测 |
| Decomposing Prediction Mechanisms for In-Context Recall | 线性动力系统回归 + recall，两种机制不同动力学 |
| Many-Shot In-Context Learning | 上百/上千例子持续提升，可覆盖预训练偏置 |
| On the generalization of LMs from ICL and finetuning | 等量数据下 ICL 泛化更灵活；加 in-context 推断可补齐 |
| Surprising Effectiveness of Test-Time Training | 测试时在例子上训练优于 ICL（如 ARC） |
| GEPA | 自然语言反思优化 prompt，少量 rollout 匹敌 RL |
| Meta-Harness | 把 harness 代码本身作为优化对象 |
| Effective context engineering for AI agents (Anthropic, 2025) | 定义 context engineering：compaction、笔记、子 agent |

---

## 6. 联系渠道

- ICL 主题帖：ICL 问题 + 本主题组队
- #267：跨主题找队友
- #287：行政问题
- ICL 主题 staff：**Matteo Guarrera**（日历见原帖）
