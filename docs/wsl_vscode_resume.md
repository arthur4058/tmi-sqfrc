# WSL、VS Code 启动与 GeoLife 基线续跑

> 更新时间：2026-07-30  
> 用途：明天从当前进度继续，不重复执行已经完成的数据预处理。

## 1. 当前进度

- 仓库：`~/research/tmi-sqfrc`
- Conda 环境：`tmi-sqfrc`
- 当前实验分支：`exp/geolife-baseline-reproduction`
- 当前实验提交：`62e2166`（已推送到 GitHub）
- 环境基线：`v0.1.0-env`
- GeoLife 数据基线：`v0.2.0-data`
- S1–S4 数据预处理已经完成，不要重跑。
- 全量数据 1 epoch 试运行已经通过，CUDA、数据加载、训练、验证和模型保存均正常。
- 正式基线训练尚未开始。

全量 1 epoch 试运行结果：

- 训练 / 验证 / 测试样本数：39,191 / 4,355 / 5,321
- Validation accuracy：0.730654
- Validation F1：0.728653
- 单个 epoch 约 21 秒；首次运行连同缓存构建共约 112 秒。

正式配置文件：

- `configs/baseline/geolife_upstream_seed10086.json`

## 2. 明天如何进入 WSL

先打开 Windows PowerShell，执行：

```powershell
wsl -d Ubuntu-22.04
```

进入 WSL 后，执行：

```bash
conda activate tmi-sqfrc
cd ~/research/tmi-sqfrc
```

检查当前位置、环境和分支：

```bash
pwd
python --version
which python
git branch --show-current
git status
```

预期结果：

- `pwd` 是 `/home/yc/research/tmi-sqfrc`
- Python 是 3.10.20
- Python 路径位于 `/home/yc/miniconda3/envs/tmi-sqfrc/`
- 分支是 `exp/geolife-baseline-reproduction`
- 如果 `git status` 只显示本启动文档尚未提交，这是正常的，不影响训练。

## 3. 从 WSL 启动 VS Code

仍在 WSL 的仓库目录中执行：

```bash
code .
```

`code .` 启动 VS Code 后命令很快返回是正常现象。

在 VS Code 中确认：

- 左下角显示 `WSL: Ubuntu-22.04`
- Git 分支显示 `exp/geolife-baseline-reproduction`
- 新建终端的提示符前有 `(tmi-sqfrc)`

如果新终端没有激活环境：

```bash
conda activate tmi-sqfrc
cd ~/research/tmi-sqfrc
```

## 4. 明天开始正式基线训练

### 4.1 先确认没有重复任务

```bash
tmux ls
ps -ef | grep '[p]ython main.py'
nvidia-smi
```

- 如果 `tmux ls` 提示没有服务器或没有会话，说明可以新建训练会话。
- 如果已经存在 `geolife-baseline` 会话，直接恢复它，不要重复启动训练：

```bash
tmux attach -t geolife-baseline
```

### 4.2 新建 tmux 会话

```bash
tmux new -s geolife-baseline
```

进入 tmux 后执行：

```bash
conda activate tmi-sqfrc
cd ~/research/tmi-sqfrc

time python main.py \
  --task dual_branch_classification_from_scratch \
  --config configs/baseline/geolife_upstream_seed10086.json
```

启动时应看到训练数据和测试数据从已有的 DataFrame 缓存读取。正式训练会写入：

```text
experiments/geolife_upstream_baseline_seed10086/
```

### 4.3 让训练在后台继续

退出 tmux 但不终止训练：

1. 按 `Ctrl+B`
2. 松开
3. 再按 `D`

稍后恢复：

```bash
tmux attach -t geolife-baseline
```

查看会话：

```bash
tmux ls
```

### 4.4 监控训练

可以在 VS Code 的另一个 WSL 终端中运行：

```bash
watch -n 2 nvidia-smi
```

按 `Ctrl+C` 退出监控。

查看训练日志：

```bash
tail -f experiments/geolife_upstream_baseline_seed10086/output.log
```

同样按 `Ctrl+C` 退出日志跟踪，这不会停止 tmux 中的训练。

预计最多约 70–75 分钟；如果提前停止条件触发，可能更早结束。

## 5. 训练完成后的检查

```bash
tail -n 80 experiments/geolife_upstream_baseline_seed10086/output.log

find experiments/geolife_upstream_baseline_seed10086 \
  -maxdepth 2 -type f -printf '%p
' | sort

git status
```

保留 `model_best.pth` 和实验结果用于后续独立测试与指标记录。`experiments/` 已被 Git 忽略，不要把大型 checkpoint 提交到仓库。

## 6. Clash Verge 的“允许局域网连接”要不要开

结论：**今晚可以关，明天本地训练时也可以保持关闭。**

以下操作不需要 Clash，也不需要“允许局域网连接”：

- 启动 WSL
- 启动 VS Code WSL
- 读取本地 GeoLife 数据
- 使用 RTX 5070 训练
- 查看本地日志和 checkpoint

只有当 WSL 必须通过 Windows 上的 Clash 联网时，才临时打开“允许局域网连接”，例如：

- `git pull` / `git push`
- `pip install`
- 从 GitHub 或其他网站下载文件

如果不开 Clash 也能正常访问 GitHub，就一直关闭即可。关闭后更少暴露局域网代理端口，也更稳妥。

如果关闭“允许局域网连接”后，WSL 的联网命令一直超时，可能是当前 shell 仍保留代理环境变量。先检查：

```bash
env | grep -i proxy
```

本次终端不需要代理时，可临时清除：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
```

以后确实需要 WSL 使用 Clash 时，再打开“允许局域网连接”并使用正确的 Windows 主机地址和代理端口（此前使用的是 7897）。

## 7. 今天收尾

今天没有启动正式训练，因此可以：

- 保存并关闭 VS Code
- 关闭 Clash 的“允许局域网连接”
- 关闭 WSL 窗口

如果要彻底停止 WSL，可在 Windows PowerShell 中执行：

```powershell
wsl --shutdown
```

**注意：以后 tmux 中正在训练时，绝对不要执行 `wsl --shutdown`，否则训练会立即终止。**

## 8. 明天不要做的事

- 不要重跑 S1、S2、S3、S4。
- 不要删除 `data/geolife_features` 中已经生成的 NPY 和 PKL 缓存。
- 不要切换到 `main` 分支启动本次实验。
- 正式训练开始后不要修改实验配置。
- 不要同时启动两个 `python main.py`。
- 训练进行时不要关闭 WSL，也不要执行 `wsl --shutdown`。
