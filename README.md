# 多生理信号驱动的心源性猝死智能预测预警与应急响应 Agent 系统

> Multi-Physiological Signal Driven Intelligent Prediction, Early-Warning and Emergency-Response Agent System for Sudden Cardiac Death (SCD)

本项目实现了一条从**原始生理信号 → 信号预处理 → 深度学习模型推理 → 风险评估 → 应急响应决策**的完整闭环，用于对潜在心律异常（尤其是可能诱发心源性猝死的恶性心律失常）进行识别、风险分级与响应策略输出。

## 1. 系统目标

- 以心电信号 (ECG) 为核心输入，预留多模态扩展接口（PPG、血压、血氧、加速度等）。
- 基于真实公开数据集 **MIT-BIH Arrhythmia Database** 与 **PTB-XL** 进行训练与评估。
- 使用 PyTorch 实现 **1D-CNN + LSTM** 混合模型完成心律失常多分类。
- 在模型输出之上构建 **风险评估模块**，将分类概率加权映射为连续风险值并分级。
- 构建 **基于规则的 Agent 决策模块**，针对不同风险等级输出结构化响应策略。
- 代码模块化组织，各模块解耦，可独立测试并通过主程序统一串联。

## 2. 数据来源

**重要**：所有数据必须由用户提前下载至本地，代码通过配置文件读取本地路径，不进行任何在线下载。

### 2.1 MIT-BIH Arrhythmia Database

- PhysioNet 标准格式：每条记录包含 `.dat`（波形）、`.hea`（头文件）、`.atr`（标注）。
- 使用 `wfdb.rdrecord` 读取波形，使用 `wfdb.rdann` 读取心拍级标注。
- 标签来自注释符号（N、V、L、R、A、...），在代码中会被映射到少数几个分类：
  - `N` → 正常窦性搏动
  - `S` → 室上性异位搏动（A、a、J、S 等）
  - `V` → 室性异位搏动（V、E 等）
  - `F` → 融合搏动（F）
  - `Q` → 未知/起搏（/、f、Q 等）

### 2.2 PTB-XL

- 同为 PhysioNet 标准格式，共 12 导联，10 秒记录。
- 附带 `ptbxl_database.csv` 提供记录级诊断标签（`scp_codes`），**无 `.atr` 文件**。
- 当前版本仅提供 `load_ptbxl_index` 索引读取作为多模态/记录级扩展接口，训练流程默认使用 MIT-BIH。

### 2.3 数据下载（需手动执行，代码不做任何下载）

从 PhysioNet 官网手动下载（推荐浏览器或 `wget` + 需要 PhysioNet 账号同意协议）：

- MIT-BIH Arrhythmia Database: <https://physionet.org/content/mitdb/1.0.0/>
- PTB-XL: <https://physionet.org/content/ptb-xl/1.0.3/>

示例（下载 + 解压 MIT-BIH）：

```bash
# 需要先注册 PhysioNet 账号并同意数据使用协议
wget -r -N -c -np --user=<YOUR_PN_USER> --ask-password \
     https://physionet.org/files/mitdb/1.0.0/
```

下载完成后将本地路径填入 [config/config.yaml](config/config.yaml) 的 `data.mitbih_dir`，或通过 CLI `--mitbih-dir` 覆盖。

### 2.4 本地目录约定（示例）

```
<你的数据根目录>/
├── mitbih/                # MIT-BIH 解压后的目录，内含 100.dat/100.hea/100.atr ...
└── ptbxl/                 # PTB-XL 解压后的目录
    ├── ptbxl_database.csv
    └── records100/ or records500/
```

数据路径由 `config/config.yaml` 配置，默认使用 MIT-BIH 作为训练数据（心拍级分类），PTB-XL 路径预留以支持扩展。

## 3. 项目结构

```
chuangke/
├── README.md                    # 本文档：设计说明与使用指南
├── pyproject.toml               # uv 管理的依赖声明
├── .python-version
├── config/
│   └── config.yaml              # 数据路径、模型超参、风险阈值、决策模板
├── src/
│   └── scd_agent/
│       ├── __init__.py
│       ├── utils/
│       │   ├── __init__.py
│       │   └── config.py        # YAML 配置加载 + 路径校验
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py        # wfdb 读取 MIT-BIH / PTB-XL
│       │   └── preprocess.py    # 带通滤波、归一化、R 峰切分
│       ├── models/
│       │   ├── __init__.py
│       │   └── cnn_lstm.py      # 1D-CNN + LSTM 分类器
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py       # 训练/评估循环、ckpt 保存/加载
│       ├── risk/
│       │   ├── __init__.py
│       │   └── assessor.py      # 加权风险值 + 等级划分
│       └── agent/
│           ├── __init__.py
│           └── decision.py      # 规则引擎：低/中/高风险 → 决策
├── scripts/
│   ├── train.py                 # 训练入口
│   └── infer.py                 # 推理入口（单条信号 → 决策）
└── main.py                      # 端到端流水线演示（训练 + 推理 + 决策）
```

## 4. 核心模块说明

### 4.1 数据加载 `data/loader.py`
- `load_mitbih_record(record_path)`：读取单条记录的 12 导联之一 + 标注。
- `build_mitbih_beats(...)`：遍历记录目录，以 R 峰为中心截取固定长度窗口（默认 360 点 ≈ 1 s），返回 `(X, y)`。
- `load_ptbxl_index(csv_path)`：读取 PTB-XL 索引 CSV，供扩展使用。

### 4.2 预处理 `data/preprocess.py`
- `bandpass_filter(signal, fs, low=0.5, high=40)`：Butterworth 带通去噪。
- `z_normalize(signal)` / `minmax_normalize(signal)`：按样本归一化。
- `segment_by_rpeak(signal, rpeaks, win_before, win_after)`：R 峰居中切分。
- `segment_by_window(signal, fs, window_sec)`：固定时长滑动切分（扩展用）。

### 4.3 模型 `models/cnn_lstm.py`
- `CNN_LSTM(nn.Module)`：
  - 3 层 1D 卷积 + BatchNorm + ReLU + MaxPool 提取局部波形特征；
  - 2 层双向 LSTM 建模时间依赖；
  - 全连接分类头输出 `num_classes` 维 logits。
- 输入形状：`(batch, channels, seq_len)`；输出：`(batch, num_classes)`。

### 4.4 训练 `training/trainer.py`
- `Trainer.fit(...)`：CrossEntropy + Adam，按 epoch 统计 loss 与 acc。
- `Trainer.evaluate(...)`：在验证集上计算 loss/acc，可选返回混淆矩阵。
- `save_checkpoint / load_checkpoint`：支持模型保存与恢复。

### 4.5 风险评估 `risk/assessor.py`
- `RiskAssessor.score(probs)`：`risk = Σ wᵢ · pᵢ`，权重由配置给出（恶性类别权重更高）。
- `RiskAssessor.grade(risk)`：按阈值 `(low, medium)` 划分为 `low / medium / high`。

### 4.6 Agent 决策 `agent/decision.py`
- `RuleBasedAgent.decide(risk_score, risk_level, extra)`：
  - `low` → `"normal_monitoring"`
  - `medium` → `"early_warning"`
  - `high` → `"emergency_response"`
- 输出字典字段：`timestamp / risk_score / risk_level / action / message / class_probs / meta`。

## 5. 运行流程

### 5.1 环境准备（uv）

```bash
uv sync                       # 根据 pyproject.toml + uv.lock 安装依赖
```

### 5.2 配置数据路径

编辑 [config/config.yaml](config/config.yaml) 中的 `data.mitbih_dir`，指向你本地 MIT-BIH 解压目录；也可以在命令行用 `--mitbih-dir /abs/path` 覆盖。

### 5.3 常用命令

```bash
# 小规模快速跑通（~30s，仅用于验证流水线完整性，非真实训练）
uv run python main.py --smoke --mitbih-dir /abs/path/to/mitbih

# 正式训练
uv run python scripts/train.py --config config/config.yaml

# CLI 覆盖参数（不改 YAML）
uv run python scripts/train.py --epochs 10 --batch-size 128 --device cuda \
    --class-weight inverse_freq --checkpoint checkpoints/v1.pt

# 单条记录推理 + 风险 + Agent 决策
uv run python scripts/infer.py --record 100 --max-beats 20 --output out.json

# 端到端演示（训练 + 保存 + 加载 + 推理 + 决策）
uv run python main.py
```

### 5.4 日志级别

设置环境变量或 CLI 参数：

```bash
SCD_LOG_LEVEL=DEBUG uv run python main.py --smoke
# 或
uv run python main.py --smoke --log-level DEBUG
```

### 5.5 单元测试

```bash
uv run pytest -q
```

测试涵盖预处理（滤波、归一化、切分）、模型前向、训练/保存/加载往返、风险评估阈值、Agent 决策映射以及配置校验；不依赖真实数据集。

## 6. 开发原则

- **不做在线下载**：数据必须本地化，路径缺失时直接报错。
- **严格解析 PhysioNet 格式**：使用 `wfdb`，不绕过解析读 CSV。
- **可运行优先**：允许限制训练样本数量，但不得跳过关键环节（切分、滤波、归一化、前反向传播、保存/加载）。
- **模块化解耦**：数据/预处理/模型/风险/Agent 各自可独立 import 与单元测试。
- **接口预留多模态**：模型第一维通道数可配置，便于后续接入多通道/多模态输入。
