import os
import sys
import json
import argparse
import importlib
import logging
import numpy as np
import torch
import lm_eval
import transformers
from lm_eval.models.huggingface import HFLM
from dataclasses import dataclass

# ------------------- Model Wrapper -------------------
@dataclass
class CausalLMOutput:
    """Wrapper to match HuggingFace output format"""
    logits: torch.Tensor
    loss: torch.Tensor = None

class HFCompatibleWrapper(torch.nn.Module):
    """Wrap custom GPT model to be compatible with HFLM"""
    def __init__(self, model, device='cuda', dtype=torch.bfloat16):
        super().__init__()
        self.model = model
        self.config = model.config
        self.device_type = 'cuda' if 'cuda' in device else 'cpu'
        self.dtype = dtype
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        with torch.autocast(device_type=self.device_type, dtype=self.dtype):
            logits, loss = self.model(input_ids, targets=None, return_logits=True, output_all_seq=True)
        return CausalLMOutput(logits=logits, loss=loss)
    
    def __getattr__(self, name):
        # Delegate attribute access to the wrapped model
        if name in ['model', 'config', 'training']:
            return super().__getattr__(name)
        return getattr(self.model, name)

# ------------------- Logging Setup -------------------
def setup_logging(verbosity):
    logging.basicConfig(
        level=verbosity.upper(), format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)

def load_task(task):
    return None, task.split(",") if task else []


# ------------------- Argument Parsing -------------------
parser = argparse.ArgumentParser(description="Evaluate a model on downstream tasks using lm_eval.")
parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the model checkpoint directory.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Precision for calculations.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cuda', 'cpu').")
parser.add_argument("--model", type=str, default="base_model", help="Model module name under model/.")
parser.add_argument("--result_dir", type=str, default="result_dir", help="Directory to save results.")
parser.add_argument("--verbosity", default="INFO", help="Logging level: CRITICAL, ERROR, WARNING, INFO, DEBUG.")
parser.add_argument("--show_config", action="store_true", default=False, help="If True, shows the full config of all tasks at the end of the evaluation.")

args = parser.parse_args()

# ------------------- Main -------------------
# Initialize logger
logger = setup_logging(args.verbosity)

# ------------------- Precision Setup -------------------
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]

# ------------------- Load Model -------------------
print("=" * 80)
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
    sys.exit(1)

# ------------------- Load Tokenizer -------------------
print(f"Loading GPT-2 tokenizer...")

try:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "gpt2",
        trust_remote_code=True,
        use_fast=True,
        model_max_length=2048,
    )
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    sys.exit(1)

print("=" * 80)

# ------------------- Wrap Model for lm_eval -------------------
wrapped_model = HFCompatibleWrapper(model, device=args.device, dtype=ptdtype)
lm = HFLM(pretrained=wrapped_model, tokenizer=tokenizer, max_length=2048)

# ------------------- Define Tasks -------------------
# Default task list with few-shot settings
task_num_fewshot_list = [
    ("arc_easy", 0),
    ("arc_challenge", 0),
    ("openbookqa", 0),
    ("boolq", 0),
    ("hellaswag", 0),
    ("winogrande", 0),
    ("piqa", 0),
]

# ------------------- Create Result Directory -------------------
model_name = os.path.basename(args.checkpoint_dir.rstrip('/'))
result_subdir = os.path.join(args.result_dir, model_name)
os.makedirs(result_subdir, exist_ok=True)

# ------------------- Evaluation Loop -------------------
print(f"\nStarting evaluation on {len(task_num_fewshot_list)} tasks...")
print("-" * 80)

all_results = {}

for task, num_fewshot in task_num_fewshot_list:
    print(f"\n{'='*80}")
    print(f"EVALUATING TASK: {task} (num_fewshot={num_fewshot})")
    print(f"{'='*80}")
    
    # Set output path for this task
    current_output_path = os.path.join(result_subdir, f"{args.model}-{task}-{num_fewshot}shot")
    file_path = current_output_path + "-results.json"
    
    # Skip if results file already exists
    if os.path.exists(file_path):
        print(f"Output file {file_path} already exists. Skipping.")
        continue
    
    try:
        # Load task and evaluate
        task_manager, task_list = load_task(task)
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=task_list,
            num_fewshot=num_fewshot,
            batch_size=args.batch_size,
            device=args.device,
            task_manager=task_manager
        )
        
        # Save results
        results_str = json.dumps(results, indent=2, default=_handle_non_serializable)
        if args.show_config:
            logger.info(results_str)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(results_str)
        print(f"Results saved to {file_path}")
        
        # Store for summary
        all_results[task] = results.get("results", {})
        
    except Exception as e:
        logger.error(f"Error evaluating task {task}: {e}")
        continue

# ------------------- Final Summary -------------------
print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)

for task, task_results in all_results.items():
    if task in task_results:
        metrics = task_results[task]

        if "acc_norm,none" in metrics:
            acc = metrics["acc_norm,none"]
            label = "Accuracy Norm"
        elif "acc,none" in metrics:
            acc = metrics["acc,none"]
            label = "Accuracy"
        elif "exact_match,none" in metrics:
            acc = metrics["exact_match,none"]
            label = "Exact Match"
        else:
            acc = "N/A"
            label = "Accuracy"

        print(f"Task: {task:<30} | {label}: {acc}")

print("=" * 80)
print(f"All results saved to: {result_subdir}")