# 环境搭建

我们在 Windows 11、Python 3.12.10、只有 CPU（18 线程，32 GB 内存）的机器上跑通。GPU 部分用 Colab（T4）。

## 1. Python 环境
```bash
cd pilot
python -m venv .venv
.venv/Scripts/pip install -r requirements-pilot.txt   # Linux/Mac: .venv/bin/pip
.venv/Scripts/pip install datasets                    # 只在 Colab 笔记本里用到
```
`requirements-pilot.txt` 是我们实际用的版本。torch 装的是 CPU 版（2.15.0+cpu）。

## 2. BrickGPT 代码（不放进仓库，单独 clone）
```bash
cd pilot
git clone https://github.com/AvaLovelace1/BrickGPT.git
git -C BrickGPT checkout 9c53c7cb65f9de0df89562cc74b2d2e270fbcab4
```
- 脚本通过 `sys.path` 引用 `pilot/BrickGPT/src`，不需要 pip install。BrickGPT 自己的依赖见它的 `pyproject.toml`。
- 我们**没有改 BrickGPT 的任何文件**。新零件是运行时加进积木库的（`pilot/common.py`、`newbrick/stage1.py`）；Gurobi 容差修正也是运行时替换的（`newbrick/tolerant_stability.py`）。

## 3. STABLE 代码（阶段 1 用它的拼法算法）
```bash
cd newbrick
git clone https://github.com/miniHuiHui/STABLE.git
git -C STABLE checkout a42b0c810b62e6e269829b7e52b6d90d19214de1
```
`stage1.py` 用的是 `STABLE/mesh2brick/voxel2brick.py`（贪心合并，再在不稳定或不连通的地方随机拆开重拼）。`stable_metrics.py` 是 STABLE `interlocking_analysis.py` 的副本。

## 4. 数据
StableText2Brick（Hugging Face：`AvaLovelace/StableText2Brick`）。放到 `pilot/data/`：
- `test.parquet`：test 集
- `train0.parquet`：train 集的第一个分片

## 5. 模型
- 底座：`unsloth/Llama-3.2-1B-Instruct`（公开镜像，权重和 Meta 的一样，不需要申请权限）
- BrickGPT 官方 LoRA：`AvaLovelace/BrickGPT`
- 第一次运行 `pilot/common.py` 的 `load_model()` 时，会把两者合并后存到 `pilot/models/brickgpt-merged/`（约 4.7 GB，float32）。

## 6. Gurobi（稳定性打分必需）
- 免费的受限版许可证不够用（会报 "Model too large"），需要**学术 WLS 许可证**（gurobi.com，用学校邮箱免费申请）。
- 下载 `gurobi.lic`，放到用户主目录（Windows：`C:\Users\<你>\gurobi.lic`）。
- **不要把 `gurobi.lic` 提交到仓库，也不要把内容贴到笔记本里**（`.gitignore` 已排除）。
- 我们用的是 Gurobi 13.0.3。一个结构打分要 1–53 秒，中位数约 3.5 秒（`newbrick/stage0_checker_timing.json`）。

## 7. Colab（GPU）
笔记本：`newbrick/colab/brickgpt_newbrick.ipynb`（由 `build_notebook.py` 生成，改动请改 `build_notebook.py` 再重新生成）。
- 运行前上传 `gurobi.lic` 到 `/content/`，并挂载 Google Drive（结果存在 `MyDrive/brickgpt_newbrick/`，可以断点续跑）。
- 装完 BrickGPT 之后要 `pip uninstall -y torchao`，否则 peft 会报版本不兼容（笔记本里已经写好）。
- 免费 Colab 一段时间不操作就会断开；生成结果每完成一个物体就写入 Drive，所以重连后能接着跑。
- 速度（T4）：不开回滚每个物体约 30 秒；开回滚很慢，最慢的一个要 49 分钟。

## 8. 本地速度参考
- 阶段 1 拼法 + 打分：3 种新零件共 1320 个拼法，开 9 个进程，从 15:38 跑到 19:34，约 4 小时。
- 在 CPU 上用 BrickGPT 生成（开回滚）：每个物体 3 分钟到 6 小时，每个进程约占 6 GB 内存。
