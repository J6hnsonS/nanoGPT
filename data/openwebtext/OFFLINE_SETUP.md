# Offline OpenWebText Dataset Preparation

This guide explains how to prepare the OpenWebText dataset when you have limited internet access.

## What You Need to Download (on a machine WITH internet)

### 1. GPT-2 Tokenizer Files (~2 MB)

Download these two files:
```bash
wget https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe
wget https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json
```

Or download manually from your browser:
- https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe
- https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json

### 2. OpenWebText Dataset (~54 GB)

**Option A: Download using HuggingFace CLI (recommended)**
```bash
pip install huggingface_hub
huggingface-cli download openwebtext --repo-type dataset --local-dir ./openwebtext_data
```

**Option B: Download parquet files manually**
Go to: https://huggingface.co/datasets/openwebtext/tree/main
Download all the `.parquet` files

**Option C: Use git-lfs**
```bash
git lfs install
git clone https://huggingface.co/datasets/openwebtext
```

## Setup on Your Training Server

1. **Create directories:**
```bash
cd /workspace/data/openwebtext
mkdir -p gpt2_tokenizer
mkdir -p openwebtext_data
```

2. **Transfer files:**

Transfer the tokenizer files to:
```
/workspace/data/openwebtext/gpt2_tokenizer/vocab.bpe
/workspace/data/openwebtext/gpt2_tokenizer/encoder.json
```

Transfer the dataset to:
```
/workspace/data/openwebtext/openwebtext_data/
```

3. **Run the offline preparation script:**
```bash
cd /workspace/data/openwebtext
python prepare_offline.py
```

This will generate:
- `train.bin` (~17 GB) - Training data
- `val.bin` (~8.5 MB) - Validation data  
- `meta.pkl` - Metadata file

## Expected Output

After successful preparation:
```
data/openwebtext/
├── train.bin          # ~17 GB, ~9B tokens
├── val.bin            # ~8.5 MB, ~4M tokens
├── meta.pkl           # Vocabulary metadata
├── prepare.py         # Original script (requires internet)
├── prepare_offline.py # Offline script (this one)
└── ...
```

## Troubleshooting

**If tiktoken still tries to download:**
The script will try to use cached tokenizer first. If that fails, it will look for local files in `gpt2_tokenizer/`.

**If dataset loading fails:**
Make sure the dataset is in one of these formats:
- HuggingFace cache format (has `dataset_info.json`)
- Parquet files (`*.parquet`)
- Arrow files

**Quick test with Shakespeare (much smaller):**
If you want to test the training pipeline first with a smaller dataset:
```bash
cd /workspace/data/shakespeare
python prepare.py  # Only needs ~1MB, might work even with limited internet
```

## After Preparation

Once you have the `.bin` files, you can start training:
```bash
cd /workspace
torchrun --standalone --nproc_per_node=1 train.py config/train_gpt2.py --wandb_log=False
```
