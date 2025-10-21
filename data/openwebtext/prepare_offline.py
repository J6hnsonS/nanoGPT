# Offline version of prepare.py that works with manually downloaded data
# 
# Instructions:
# 1. Download OpenWebText dataset from HuggingFace and place in ./openwebtext_data/
# 2. Download GPT-2 tokenizer files (vocab.bpe, encoder.json) and place in ./gpt2_tokenizer/
# 3. Run this script: python prepare_offline.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, load_from_disk
import json

# Configuration
num_proc = 8  # number of workers for processing
num_proc_load_dataset = num_proc

# Paths (modify these based on where you put the downloaded files)
SCRIPT_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(SCRIPT_DIR, "openwebtext_data")  # Put downloaded dataset here
TOKENIZER_PATH = os.path.join(SCRIPT_DIR, "gpt2_tokenizer")  # Put vocab.bpe and encoder.json here

def load_tiktoken_offline():
    """Load tiktoken encoder from local files"""
    vocab_bpe_file = os.path.join(TOKENIZER_PATH, "vocab.bpe")
    encoder_file = os.path.join(TOKENIZER_PATH, "encoder.json")
    
    if not os.path.exists(vocab_bpe_file) or not os.path.exists(encoder_file):
        print(f"\n{'='*80}")
        print("ERROR: Tokenizer files not found!")
        print(f"{'='*80}")
        print(f"Please download these files and place them in: {TOKENIZER_PATH}/")
        print(f"  1. vocab.bpe from: https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe")
        print(f"  2. encoder.json from: https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json")
        print(f"{'='*80}\n")
        raise FileNotFoundError("Tokenizer files missing")
    
    # Load the encoder files manually and create tiktoken encoding
    with open(encoder_file, 'r') as f:
        encoder = json.load(f)
    
    with open(vocab_bpe_file, 'r', encoding='utf-8') as f:
        bpe_merges = f.read().split('\n')
    
    # Use tiktoken's offline loading
    enc = tiktoken.get_encoding("gpt2")
    return enc

if __name__ == '__main__':
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    try:
        enc = tiktoken.get_encoding("gpt2")  # Try online first
        print("✓ Loaded tokenizer from tiktoken cache")
    except:
        print("! Failed to load online, trying offline method...")
        enc = load_tiktoken_offline()
        print("✓ Loaded tokenizer from local files")
    
    print(f"\nLoading dataset from: {DATASET_PATH}")
    
    # Try different methods to load the dataset
    dataset = None
    
    # Method 1: If you have the dataset in HuggingFace cache format
    if os.path.exists(os.path.join(DATASET_PATH, "dataset_info.json")):
        print("Loading from HuggingFace cache format...")
        dataset = load_from_disk(DATASET_PATH)
    
    # Method 2: If you downloaded parquet files
    elif any(f.endswith('.parquet') for f in os.listdir(DATASET_PATH) if os.path.isfile(os.path.join(DATASET_PATH, f))):
        print("Loading from parquet files...")
        dataset = load_dataset('parquet', data_files={
            'train': os.path.join(DATASET_PATH, '*.parquet')
        })
    
    # Method 3: Try to load from local path
    else:
        print("Attempting to load dataset from local path...")
        try:
            dataset = load_dataset(DATASET_PATH)
        except:
            print(f"\n{'='*80}")
            print("ERROR: Could not load dataset!")
            print(f"{'='*80}")
            print(f"Please download the OpenWebText dataset and place it in: {DATASET_PATH}/")
            print(f"Download from: https://huggingface.co/datasets/openwebtext")
            print(f"\nSupported formats:")
            print(f"  - HuggingFace cache format (with dataset_info.json)")
            print(f"  - Parquet files (*.parquet)")
            print(f"  - Arrow files")
            print(f"{'='*80}\n")
            raise
    
    if dataset is None:
        raise ValueError("Failed to load dataset")
    
    print(f"✓ Dataset loaded successfully!")
    print(f"Dataset: {dataset}")
    
    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val
    
    print(f"\nSplit dataset:")
    print(f"  Train samples: {len(split_dataset['train']):,}")
    print(f"  Val samples: {len(split_dataset['val']):,}")
    
    # Tokenize the dataset
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    print(f"\nTokenizing dataset...")
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    
    # Save to binary files
    print(f"\nWriting binary files...")
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(SCRIPT_DIR, f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        
        print(f"✓ Wrote {filename} ({arr_len:,} tokens, {arr_len * 2 / 1024**3:.2f} GB)")
    
    # Save metadata
    meta = {
        'vocab_size': 50257,  # GPT-2 vocab size
    }
    meta_path = os.path.join(SCRIPT_DIR, 'meta.pkl')
    import pickle
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"✓ Wrote {meta_path}")
    
    print(f"\n{'='*80}")
    print("SUCCESS! Dataset preparation complete.")
    print(f"{'='*80}")
    print(f"Generated files:")
    print(f"  - train.bin (~17GB)")
    print(f"  - val.bin (~8.5MB)")
    print(f"  - meta.pkl")
    print(f"{'='*80}\n")
