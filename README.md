# 吞宝 · Tunbao-Omni

吞下多种模态，学会理解与表达。

当前版本提供基于 T2M-GPT 的**文字生成动作基线**：训练 VQ-VAE，将动作编码为离散编号，再训练自回归 GPT 根据英文描述生成动作。后续逐步扩展语音等模态。

本仓库只包含这条核心流程。数据、权重、网页和实验临时文件不在仓库中。

## 环境

Linux，Python 3.10，CUDA GPU。已使用的 PyTorch 版本为 2.5.1 / CUDA 12.4。

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

训练中的动作评估需要原始 [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) 提供的 HumanML3D 评估器和 GloVe 资源。按上游说明获取后，保持其目录结构放到 `vendor/t2m_gpt/checkpoints/` 和 `vendor/t2m_gpt/glove/`；也可以通过 `T2M_GPT_ROOT` 指向已经准备好的上游项目目录。

另外准备 OpenAI CLIP ViT-B/32 权重，并设置实际路径：

```bash
export CLIP_MODEL_PATH="$HOME/.cache/clip/ViT-B-32.pt"
# 可选：数据、断点和缓存默认写入本仓库 data/。
export TUNBAO_DATA_ROOT="$(pwd)/data"
```

CLIP 权重可通过安装的官方 CLIP 包获取：`python -c "import clip; clip.load('ViT-B/32', device='cpu')"`。

## 数据准备

输入为已经处理好的 **20fps、22 身体关节、263 维 HumanML3D 动作特征**，每段一个 `[帧数, 263]` 的 `.npy` 文件。原始 BVH、SMPL 参数和关节坐标不能直接当作这些特征输入；其转换不属于本次发布范围。

准备一个 JSONL 描述文件，每行对应一个动作：

```json
{"id":"clip_001","group":"capture_take_001","captions":["A person walks forward.","Someone is walking ahead."]}
```

`id` 对应 `clip_001.npy`；`group` 必须标识原始采集 take，同一 take 的片段使用同一个 group，避免跨训练/验证集。`captions` 保留该动作的全部英文描述。

```bash
python baseline/prepare_data.py --features /path/to/features --captions /path/to/captions.jsonl
python baseline/prepare_vq.py
```

划分按原始 take 分组并固定 seed=123；只在训练部分拟合均值和标准差。VQ 训练使用至少 64 帧的动作窗口；完整编码时保留短动作，只裁掉末尾不足 4 帧的部分。

## 训练与推理

```bash
# 1. 训练动作压缩模型
python baseline/train_vq.py --precision fp32 --steps 300000

# 2. 用验证 FID 最佳的 VQ 权重编码动作，并准备文本特征
python baseline/encode.py
python baseline/prepare_gpt.py

# 3. 训练文字到动作 GPT，每 1000 步验证和保存
python baseline/train_gpt.py --steps 300000 --eval-every 1000

# 4. 从 GPT 最佳权重生成动作，默认使用随机采样
python baseline/infer.py --text "A person walks forward." --out outputs/walk
```

VQ 使用 512 项码本，每 4 帧约对应一个动作编号。GPT 最长生成 49 个编号，即 196 帧、约 9.8 秒；训练 GPT 时排除超过该长度的整段动作，不把长动作随意切碎。CLIP 文本编码器保持冻结。

推理输出 `codes.npy`、`joints.npy`（`[帧数,22,3]`，20fps）以及生成参数 `motion.json`。设置 `--seed 123` 可复现采样；仅在显式加入 `--greedy` 时使用贪心解码。

训练支持 SIGTERM/SIGINT 保存后退出。恢复完整断点：

```bash
python baseline/train_vq.py --precision fp32 --steps 300000 --resume "$TUNBAO_DATA_ROOT/vq/resume_latest.pt"
python baseline/train_gpt.py --steps 300000 --eval-every 1000 --resume "$TUNBAO_DATA_ROOT/gpt/resume_latest.pt"
```

断点同时保存优化器、学习率计划、批次位置和随机数状态。`net_best_fid.pth` 用于推理；`resume_latest.pt` 用于续训。

## 评估范围

训练日志中的 FID 和 R@1/2/3 使用 HumanML3D 预训练评估器，在当前数据的验证划分上计算，**不是官方 HumanML3D 测试集成绩，也不能与 Kimodo 官方榜单直接比较**。验证集同时用于选择最佳权重，正式泛化评估还需要独立测试集。

## 来源与许可

模型结构及必要依赖来自 [T2M-GPT](https://github.com/Mael-zys/T2M-GPT)，保留在 `vendor/t2m_gpt/`，遵循其 Apache-2.0 许可证。项目集成代码采用仓库根目录的 MIT 许可证，详见 [NOTICE](NOTICE)。数据及外部模型资源需单独获取并遵守各自许可。
