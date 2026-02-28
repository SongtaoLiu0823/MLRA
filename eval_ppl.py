import os
import math
import argparse
import importlib
import numpy as np
import torch
from tqdm import trange
from glob import glob
from contextlib import nullcontext

# ------------------- Argument Parsing -------------------
parser = argparse.ArgumentParser(description="Evaluate a model's loss on multiple validation datasets.")
parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the model checkpoint directory.")
parser.add_argument("--data_dir", type=str, default="data/eval_datasets", help="Directory containing the .bin evaluation files.")
parser.add_argument("--block_size", type=int, default=2048, help="Block size for context.")
parser.add_argument("--batch_size", type=int, default=6, help="Batch size for evaluation.")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Precision for calculations.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cuda', 'cpu').")
parser.add_argument("--model", type=str, default="base_model", help="Model module name under model/.")
parser.add_argument("--eval_tokens", type=float, default=1e8, help="Number of tokens to evaluate per dataset (e.g., 1e8 for 0.1B).")

args = parser.parse_args()

# ------------------- Device and Precision Setup -------------------
device_type = 'cuda' if 'cuda' in args.device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

# ------------------- Load Model -------------------
# This is done once, as the model is the same for all evaluations.
print("="*80)
print(f"Loading model from: {args.checkpoint_dir}")
model_file = importlib.import_module(f'model.{args.model}')
GPTConfig = model_file.GPTConfig
GPT = model_file.GPT

try:
    config = GPTConfig.from_json_file(os.path.join(args.checkpoint_dir, 'config.json'))
    model = GPT.from_pretrained(args.checkpoint_dir, config=config)
    model.to(args.device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)
print("="*80)


# ------------------- Helper Functions -------------------
def get_batch_factory(val_data):
    """
    Creates a get_batch function for a specific validation dataset.
    This encapsulates the data and offset, making it cleaner.
    """
    offset = 512  # Initial offset to avoid starting at the very beginning

    def get_batch_val():
        nonlocal offset
        x_list, y_list = [], []
        for _ in range(args.batch_size):
            start = offset
            # Ensure we don't read past the end of the file
            if start + args.block_size + 1 > len(val_data):
                # Reset offset if we are at the end of the data
                print("\nWarning: Reached end of validation data. Resetting offset.")
                offset = 512
                start = offset
            
            x = val_data[start : start + args.block_size]
            y = val_data[start + 1 : start + 1 + args.block_size]
            x_list.append(torch.from_numpy(x.astype(np.int64)))
            y_list.append(torch.from_numpy(y.astype(np.int64)))
            offset += args.block_size
        
        x = torch.stack(x_list)
        y = torch.stack(y_list)

        if device_type == 'cuda':
            return x.pin_memory().to(args.device, non_blocking=True), y.pin_memory().to(args.device, non_blocking=True)
        else:
            return x.to(args.device), y.to(args.device)
    
    return get_batch_val

@torch.no_grad()
def estimate_loss(get_batch_func, eval_iters):
    """Evaluates the loss for a given number of iterations."""
    losses = torch.zeros(eval_iters)
    for i in trange(eval_iters, desc="Evaluating"):
        X, Y = get_batch_func()
        with ctx:
            _, loss = model(X, Y)
        losses[i] = loss.item()
    return losses.mean().item()

# ------------------- Main Evaluation Loop -------------------
# Find all .bin files in the specified data directory
eval_files = glob(os.path.join(args.data_dir, '*.bin'))

if not eval_files:
    print(f"Error: No .bin files found in '{args.data_dir}'. Please check the path.")
    exit(1)

print(f"Found {len(eval_files)} evaluation datasets. Starting evaluation loop...\n")

results = {}

for val_path in eval_files:
    dataset_name = os.path.basename(val_path)
    print("-" * 80)
    print(f"EVALUATING DATASET: {dataset_name}")
    print("-" * 80)

    # Load the validation data for the current dataset
    val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
    
    # Calculate evaluation iterations for this dataset
    tokens_target = int(args.eval_tokens)
    tokens_per_iter = args.batch_size * args.block_size
    # Ensure we don't evaluate more tokens than exist in the file
    max_iters_for_file = (len(val_data) - 512) // tokens_per_iter
    eval_iters = min(tokens_target // tokens_per_iter, max_iters_for_file)

    print(f"Tokens per iteration: {tokens_per_iter}")
    print(f"Target tokens: {tokens_target: ,}")
    print(f"Available tokens in file: {len(val_data):,}")
    print(f"Setting eval_iters = {eval_iters}")

    if eval_iters == 0:
        print("Not enough tokens in the dataset to perform a single evaluation iteration. Skipping.")
        continue

    # Create a new get_batch function for the current data
    get_batch_val = get_batch_factory(val_data)

    # Run the evaluation
    val_loss = estimate_loss(get_batch_val, eval_iters)
    perplexity = math.exp(val_loss)
    results[dataset_name] = {'loss': val_loss, 'perplexity': perplexity}

    print(f"\n--- Results for {dataset_name} ---")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Perplexity:      {perplexity:.4f}\n")

# ------------------- Final Summary -------------------
print("=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)
for name, metrics in results.items():
    print(f"Dataset: {name:<40} | Loss: {metrics['loss']:.4f} | Perplexity: {metrics['perplexity']:.4f}")
print("=" * 80)
