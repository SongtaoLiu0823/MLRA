# Multi-Head Low-Rank Attention

This repository provides the official implementation of [**Multi-Head Low-Rank Attention**](https://openreview.net/pdf?id=vBJKZ19XGY). MLRA is a novel attention mechanism that natively supports 4-way tensor parallelism and significantly reduces the key-value (KV) cache size, enabling efficient long-context inference at scale.

📝 [**Paper**](https://openreview.net/pdf?id=vBJKZ19XGY) | 📖 [**Blog**](https://songtaoliu0823.github.io/mlra/)

---

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Pretrained Weights](#pretrained-weights)
- [Pretraining](#pretraining)
- [Resume Training](#resume-training)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Hardware Requirements

- **GPU**: NVIDIA Hopper architecture (e.g., H100) is required for FlashAttention-3 and FlashMLA support.
- **VRAM**: At least **8 × 80 GB** GPU memory is needed for pretraining at the 2.9B scale.

---

## Installation

Follow the steps below to set up the environment. All commands should be run from the root of the repository unless otherwise specified.

### 1. Clone the Repository

```bash
git clone https://github.com/SongtaoLiu0823/MLRA.git
cd MLRA
```

### 2. Create and Activate a Virtual Environment

```bash
conda create -n mlra python=3.10.14
conda activate mlra
```

### 3. Install Core Dependencies

```bash
pip3 install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip3 install transformers==4.51.3 datasets==3.5.1 tiktoken==0.9.0 wandb==0.19.10 tqdm==4.67.1 accelerate==1.6.0 deepspeed==0.16.7
```

### 4. Install FlashAttention-3

FlashAttention-3 is required for efficient attention computation on Hopper GPUs.

```bash
cd flash-attention
git clone https://github.com/NVIDIA/cutlass.git csrc/cutlass
cd hopper
MAX_JOBS=16 python3 setup.py install
cd ../..
```

> **Note:** `MAX_JOBS` controls the number of parallel compilation jobs. Reduce this value if your system runs out of memory during compilation.

### 5. Install lm-evaluation-harness

Used for downstream zero-shot common-sense reasoning evaluation.

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip3 install -e .
cd ..
```

### 6. Install FlashMLA

FlashMLA provides a highly optimized CUDA kernel for MLA decoding.

```bash
git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla
cd flash-mla
git submodule update --init --recursive
pip3 install -v .
cd ..
```

---

## Dataset

We provide preprocessed datasets on Hugging Face for convenience, as well as scripts to reproduce the datasets from scratch.

### Option A: Download Preprocessed Data (Recommended)

```bash
cd data
python3 download_train_data.py
python3 download_eval_data.py
cd ..
```

### Option B: Reproduce from Scratch

If you prefer to prepare the datasets yourself from original sources:

```bash
cd data
python3 get_train_data.py   # downloads and tokenizes training corpora
python3 get_eval_data.py    # downloads and tokenizes evaluation corpora
cd ..
```

Evaluation data includes: Wikipedia, C4, Pile, RefinedWeb, Cosmopedia, FineWeb, and FineWeb-Edu.

---

## Pretrained Weights

We provide pretrained weights for all model variants on Hugging Face at [Soughing/MLRA](https://huggingface.co/Soughing/MLRA). To download the weights locally, run:

```bash
python3 output/down_model.py
```

The weights will be saved under the `output/` directory and can be used directly for evaluation without rerunning pretraining.

---

## Pretraining

We use `torchrun` for distributed pretraining across multiple GPUs. Config files for all model variants are provided under the `config/` directory.

### Basic Command

```bash
torchrun --standalone --nproc_per_node=8 \
    train_mlra.py \
    config/train_mlra_4.py
```

- `--nproc_per_node=8`: number of GPU processes to launch (should match the number of available GPUs).
- `train_mlra.py`: main training script.
- `config/train_mlra_4.py`: configuration file for the MLRA-4 variant.

To train other attention variants (e.g., MHA, MQA, GQA, MLA), replace `train_mlra.py` and the config file accordingly.

## Resume Training

To resume training from a checkpoint, update the config file by uncommenting and filling in the following fields:

```python
checkpoint_step = 20000   # step number of the checkpoint to resume from
resume_dir = f"output/{output_name}/checkpoint-{checkpoint_step}"
init_from = 'resume'
```

Then launch training with the same `torchrun` command as before. The trainer will automatically skip the first `checkpoint_step` steps and resume from the saved state.

> **Note:** Make sure `output_name` matches the original run, and that the hardware configuration (number of GPUs, gradient accumulation steps, batch size, etc.) is identical to the original run to ensure correctness.

---

## Evaluation

We provide evaluation scripts for perplexity, zero-shot common-sense reasoning, and decoding speed.

### Perplexity Evaluation

Evaluates the model on seven datasets: Wikipedia, C4, Pile, RefinedWeb, Cosmopedia, FineWeb, and FineWeb-Edu.

```bash
python3 eval_ppl.py \
    --checkpoint_dir output/main/mlra_4 \
    --model mlra_4
```

### Common-Sense Reasoning Evaluation

Evaluates zero-shot performance on seven benchmarks: ARC-E, ARC-C, OpenBookQA, BoolQ, HellaSwag, Winogrande, and PIQA.

```bash
python3 eval_downstream.py \
    --checkpoint_dir output/main/mlra_4 \
    --model mlra_4
```

### Decoding Speed Benchmark

Measures decoding latency and throughput for GQA, MLA, GLA-2, and MLRA across different sequence lengths and tensor parallelism configurations.

```bash
python3 benchmark_decoding_speed.py
```

### Arguments

| Argument | Description |
|---|---|
| `--checkpoint_dir` | Path to the model checkpoint directory |
| `--model` | Model variant identifier (e.g., `mlra_2`, `mlra_4`, `mha`, `mla`) |

---

## Acknowledgements

This project builds upon the following open-source works:

- [nanoGPT](https://github.com/karpathy/nanoGPT) — training framework foundation.
- [TPA](https://github.com/tensorgi/TPA) — Tensor Product Attention reference implementation.
- [GLA](https://github.com/Dao-AILab/grouped-latent-attention) — Grouped Latent Attention reference implementation.
- [FlashAttention-3](https://github.com/Dao-AILab/flash-attention/tree/main/hopper) — efficient attention kernel for Hopper GPUs.
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA) — optimized MLA decoding kernel.
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — standardized LM evaluation framework.
- [Hugging Face Datasets](https://huggingface.co/): [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), [C4](https://huggingface.co/datasets/allenai/c4), [Pile](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated), [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb), [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia), [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/).

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{liu2026multi,
  title     = {Multi-Head Low-Rank Attention},
  author    = {Liu, Songtao and Peng, Hongwu and Zhang, Zhiwei and Chen, Zhengyu and Guo, Yue},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```
