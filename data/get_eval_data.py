import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
import time
from datasets import load_dataset, Dataset
from tqdm import tqdm

def write_datafile(filename, toks):
    """Saves token data as a .bin file."""
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = len(toks)
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    print(f"\nWriting {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

# ------------------------------------------
# Datasets to be sampled
DATASETS_TO_SAMPLE = [
    {"name": "wikimedia/wikipedia", "subset": "20231101.en", "split": "train", "text_col": "text", "data": "Wikipedia"},
    {"name": "allenai/c4", "subset": "en", "split": "train", "text_col": "text", "data": "C4"},
    {"name": "HuggingFaceTB/cosmopedia", "subset": "web_samples_v2", "split": "train", "text_col": "text", "data": "Cosmopedia"},
    {"name": "EleutherAI/the_pile_deduplicated", "split": "train", "text_col": "text", "data": "Pile"},
    {"name": "tiiuae/falcon-refinedweb", "split": "train", "text_col": "content", "data": "RefinedWeb"},
    {"name": "HuggingFaceFW/fineweb", "split": "train", "text_col": "text", "data": "FineWeb"},
]
# ------------------------------------------
seed = 42

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    """Tokenizes a single document."""
    tokens = [eot] + enc.encode_ordinary(doc["text"])
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

def process_dataset(d_info, output_dir, target_tokens, subset_gb):
    """
    Processes a dataset. Uses a streaming path for all other datasets.
    """
    dataset_clean_name = d_info['data']
    output_filename = os.path.join(output_dir, f"{dataset_clean_name}_eval.bin")
    
    print("-" * 80)
    print(f"Processing dataset: {d_info['name']} ({d_info.get('subset', 'default')})")

    # --- Default path for all other datasets: Stream and sample ---
    print(f"-> Step 1: Collecting ~{subset_gb}GB of text data into RAM from stream.")
    try:
        ds_stream = load_dataset(d_info["name"], name=d_info.get("subset"), split=d_info["split"], streaming=True, trust_remote_code=True)
        if d_info["text_col"] != "text":
            ds_stream = ds_stream.rename_column(d_info["text_col"], "text")
        ds_stream = ds_stream.filter(lambda x: len(x['text']) > 0)
    except Exception as e:
        print(f"Could not load dataset stream for {d_info['name']}. Skipping. Error: {e}")
        return
    
    subset_docs = []
    current_bytes = 0
    target_subset_bytes = subset_gb * 1024**3
    with tqdm(total=target_subset_bytes, desc="Streaming data", unit='B', unit_scale=True, unit_divisor=1024) as pbar:
        for doc in ds_stream:
            doc_bytes = len(doc['text'].encode('utf-8'))
            if current_bytes + doc_bytes > target_subset_bytes:
                break
            subset_docs.append(doc)
            current_bytes += doc_bytes
            pbar.update(doc_bytes)

    if not subset_docs:
        print("No documents were collected from the stream. Skipping dataset.")
        return

    print(f"Collected {len(subset_docs)} documents, total size ~{current_bytes / 1024**3:.2f} GB.")
    print("-> Step 2: Converting to a local dataset, shuffling, and tokenizing.")
    ds = Dataset.from_list(subset_docs)
    ds = ds.shuffle(seed=seed)
    
    # --- The tokenization loop is now common for both paths ---
    # It operates on the `ds` object, which is prepared correctly by the logic above.
    print("-> Final Step: Tokenizing the data to create the .bin file.")
    all_tokens_np = np.empty((target_tokens,), dtype=np.uint16)
    token_count = 0
    
    nprocs = max(1, 16)
    with mp.Pool(nprocs) as pool:
        with tqdm(total=target_tokens, unit="tokens", desc=f"Tokenizing {dataset_clean_name}") as progress_bar:
            # pool.imap iterates through the `ds` object provided by either the
            # the streaming path.
            for tokens in pool.imap(tokenize, ds, chunksize=16):
                space_left = target_tokens - token_count
                if len(tokens) > space_left:
                    tokens = tokens[:space_left]
                
                if len(tokens) > 0:
                    all_tokens_np[token_count : token_count + len(tokens)] = tokens
                    token_count += len(tokens)
                    progress_bar.update(len(tokens))

                if token_count >= target_tokens:
                    break
    
    final_tokens = all_tokens_np[:token_count]
    if len(final_tokens) > 0:
        write_datafile(output_filename, final_tokens)
    else:
        print(f"No tokens were processed for {dataset_clean_name}. No file written.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create evaluation datasets by sampling from a large in-memory subset.")
    parser.add_argument("--tokens_per_dataset", type=float, default=1e8, help="Target tokens to sample for the final .bin file (e.g., 1e8 for 0.1B).")
    parser.add_argument("--subset_gb", type=int, default=10, help="Size in GB of the temporary subset to collect in RAM for sampling.")
    parser.add_argument("--output_dir", type=str, default="eval_datasets", help="Directory to save the processed .bin files.")
    parser.add_argument("--delay", type=int, default=5, help="Seconds to wait between processing each dataset.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Remind the user about memory usage for streaming datasets
    print("="*80)
    print(f"IMPORTANT: For streaming datasets, this script will collect ~{args.subset_gb}GB of data into RAM.")
    print("Ensure your system has enough available memory and disk space.")
    print("="*80)
    time.sleep(5) # Give user time to read the warning

    for i, d_info in enumerate(DATASETS_TO_SAMPLE):
        process_dataset(
            d_info=d_info,
            output_dir=args.output_dir,
            target_tokens=int(args.tokens_per_dataset),
            subset_gb=args.subset_gb
        )
        
        if i < len(DATASETS_TO_SAMPLE) - 1:
            print(f"\nWaiting for {args.delay} seconds before starting the next dataset...")
            time.sleep(args.delay)
    
    print("\nAll datasets have been processed.")
    print(f"Check the directory for results: {args.output_dir}")