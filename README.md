# 吞宝 · Tunbao-Omni

吞下多种模态，学会理解与表达。目前实现基于 T2M-GPT 的文字生成动作，后续扩展语音等模态。

## 安装

需要 Linux 和 CUDA 显卡。

```bash
conda create -n tunbao python=3.10 -y
conda activate tunbao
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import clip; clip.load('ViT-B/32', device='cpu')"
```

## 数据准备

动作使用 20fps、263 维 HumanML3D 特征，每段保存为一个 `.npy`。描述文件使用 JSONL，每行如下；`id` 对应文件名，同一次采集的动作使用相同 `group`。

```json
{"id":"clip_001","group":"take_001","captions":["A person walks forward."]}
```

按 [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) 的说明获取评估器和 GloVe，保持目录结构放入 `vendor/t2m_gpt/checkpoints/` 和 `vendor/t2m_gpt/glove/`。

## 训练

```bash
python baseline/prepare_data.py --features /path/to/features --captions /path/to/captions.jsonl
python baseline/prepare_vq.py
python baseline/train_vq.py
python baseline/encode.py
python baseline/prepare_gpt.py
python baseline/train_gpt.py
```

依次完成数据准备、VQ-VAE 训练、动作编码和 GPT 训练。断点与缓存保存在 `data/`，日志指标为当前数据的验证集成绩。

## 推理

```bash
python baseline/infer.py --text "A person walks forward." --out outputs/walk
```

默认随机采样，输出动作编号和 22 关节坐标。需自行准备训练数据与权重。

模型结构来自 T2M-GPT，原始许可证保留在 `vendor/t2m_gpt/LICENSE`，详见 [NOTICE](NOTICE)。
