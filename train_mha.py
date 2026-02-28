import os
import time
import math
import pickle
import importlib
import numpy as np
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim import AdamW
from contextlib import nullcontext
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on Fineweb-edu 100B
# I/O
data_path = "data"
out_dir = 'output/out'
resume_dir = '.'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# init_from = 'resume'
# wandb logging
output_name = ""
wandb_log = False # disabled by default
wandb_project = 'KV'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'fineweb-edu100B'
gradient_accumulation_steps = 5 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 2048
# model
n_layer = 12
n_head = 12
head_dim = 64
n_embd = 768
intermediate_size = 3072
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# optimizer
optimizer_name = 'adamw' 
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
rho = 0.1
interval = 10
variant = 4 
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
checkpoint_step = -20000
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
schedule = 'cosine'
model_type = 'base_model'
# System configuration
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

# Fixed seed for reproducibility
base_seed = 5000  # Global seed for all randomization
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
model_file = importlib.import_module(f'model.{model_type}')
GPTConfig = model_file.GPTConfig
GPT = model_file.GPT

def get_num_params(self, non_embedding=False):
    """
    Return the number of parameters in the model.
    For non-embedding count (default), the position embeddings get subtracted.
    The token embeddings would too, except due to the parameter sharing these
    params are actually used as weights in the final layer, so we include them.
    """
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
        n_params -= self.transformer.wpe.weight.numel()
    return n_params

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}, RANK: {os.environ.get('RANK')}, LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    world_size = 1
    gradient_accumulation_steps *= 8 # simulate 8 gpus

# Calculate total tokens in billions
tokens_per_iter = batch_size * block_size * gradient_accumulation_steps * world_size
total_tokens_B = tokens_per_iter * max_iters / (1000 ** 3)

# Add after the initial variable declarations
tokens_trained = 0  # track total tokens trained

# Initialize random seed and torch settings
torch.manual_seed(base_seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

# Data directory
data_dir = os.path.join(data_path, dataset)

# Simple data loading implementation without DataLoader
train_file_list = sorted([os.path.join(data_dir, x) for x in os.listdir(data_dir) 
                         if x.endswith('.bin') and x.startswith('fineweb_train')])
val_path = os.path.join(data_dir, 'fineweb_val_000000.bin')

print(f"Found {len(train_file_list)} training files")
train_data_list = [np.memmap(file, dtype=np.uint16, mode='r') for file in train_file_list]
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')

# Initialize data order tracking variables
# For partitioning, each GPU gets its own subset of files
current_offset = 0  # Track position within the assigned files

# Get rank for current process
rank_id = ddp_rank if ddp else 0

# Each GPU is assigned a subset of files based on its rank
# GPU 0 gets files 0, 0+world_size, 0+2*world_size, ...
# GPU 1 gets files 1, 1+world_size, 1+2*world_size, ...
assigned_indices = [i for i in range(len(train_file_list)) if i % world_size == rank_id]
print(f"Rank {rank_id}: Assigned {len(assigned_indices)} files out of {len(train_file_list)}")
print(f"Rank {rank_id}: First 5 assigned files: {assigned_indices[:min(5, len(assigned_indices))]}")

def get_batch(split):
    global current_offset
    
    if split == 'train':
        # Training data handling remains unchanged
        file_idx = assigned_indices[current_offset % len(assigned_indices)]
        data = train_data_list[file_idx]
        
        # Initialize sequence tracking if not already done
        if not hasattr(get_batch, 'seq_offsets'):
            get_batch.seq_offsets = {}
        if file_idx not in get_batch.seq_offsets:
            get_batch.seq_offsets[file_idx] = 512  # Starting offset
        
        # Check if we have enough space to get a batch
        if get_batch.seq_offsets[file_idx] + (batch_size * block_size) >= len(data):
            # Reset the offset and move to the next file
            get_batch.seq_offsets[file_idx] = 512
            current_offset = (current_offset + 1) % len(assigned_indices)
            # Recursively call to get a batch from the next file
            return get_batch(split)
        
        # Sequential extraction of a batch
        sequences_x = []
        sequences_y = []
        for i in range(batch_size):
            start_idx = get_batch.seq_offsets[file_idx]
            x_seq = data[start_idx:start_idx+block_size]
            y_seq = data[start_idx+1:start_idx+1+block_size]
            
            # Convert to the correct data type and add to batch
            sequences_x.append(torch.from_numpy(x_seq.astype(np.int64)))
            sequences_y.append(torch.from_numpy(y_seq.astype(np.int64)))
            
            # Update sequence offset
            get_batch.seq_offsets[file_idx] += block_size
    else:
        # Validation data handling - improved to ensure different processes use different data
        
        # Initialize global validation offset as a static variable if not exists
        if not hasattr(get_batch, 'global_val_offset'):
            get_batch.global_val_offset = 512
        
        # Check if we need to reset validation offset (only master process updates global offset)
        if get_batch.global_val_offset + (batch_size * block_size) >= len(val_data):

            # All processes reset independently but synchronously
            get_batch.global_val_offset = 512
        
        data = val_data
        sequences_x = []
        sequences_y = []
        
        # Extract sequences from this process's assigned section
        for i in range(batch_size):
            start_idx = get_batch.global_val_offset

            x_seq = data[start_idx:start_idx+block_size]
            y_seq = data[start_idx+1:start_idx+1+block_size]
            
            sequences_x.append(torch.from_numpy(x_seq.astype(np.int64)))
            sequences_y.append(torch.from_numpy(y_seq.astype(np.int64)))
    
            get_batch.global_val_offset += block_size
    
    # Stack sequences and move to device
    x = torch.stack(sequences_x)
    y = torch.stack(sequences_y)
    
    if device_type == 'cuda':
        # Pin arrays x,y, which allows us to move them to GPU asynchronously
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y

# Init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
clip_time = 0
best_val_loss = 1e9

# Attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Model initialization
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, intermediate_size=intermediate_size,
                  block_size=block_size, bias=bias, head_dim=head_dim, vocab_size=None, dropout=dropout) # start with model_args from command line
                  
if init_from == 'scratch':
    # Init a new model from scratch
    print("Initializing a new model from scratch")
    # Determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {resume_dir}")
    # Resume training from a checkpoint.
    config = GPTConfig.from_json_file(os.path.join(resume_dir, 'config.json'))
    model = GPT.from_pretrained(resume_dir, config=config)
    
    # Force these config attributes to be equal otherwise we can't resume training
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(config, k)
    model.transformer.wte.weight = model.lm_head.weight
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # Initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # Read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
        
model.to(device)

# Now calculate non-embedding parameters
param_count = get_num_params(model, non_embedding=False)
param_count_m = param_count / 1_000_000  # convert to millions

# Update wandb run name and out_dir if not resuming
if init_from != 'resume':
    # Update wandb run name
    wandb_run_name = f"{output_name}"
    # Update output directory
    out_dir = f"output/{output_name}"
else:
    try:
        wandb_run_name = f"{output_name}"
        out_dir = f"output/{output_name}"
    except:
        pass
        
# Now create the output directory
if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer
params = list(model.parameters())
optimizer = AdamW(params, lr=learning_rate, betas=(beta1, beta2), eps=1e-8, weight_decay=weight_decay)

# Load optimizer and training state if resuming
if init_from == 'resume':
    optimizer_state_path = os.path.join(resume_dir, 'optimizer.pt')
    if os.path.exists(optimizer_state_path):
        optimizer_state = torch.load(optimizer_state_path, map_location=device)
        optimizer.load_state_dict(optimizer_state['optimizer'])
        iter_num = optimizer_state['iter_num']
        clip_time = optimizer_state['clip_time']
        best_val_loss = optimizer_state['best_val_loss']
        tokens_trained = optimizer_state['tokens_trained']
        if dtype == 'float16' and 'scaler' in optimizer_state and optimizer_state['scaler'] is not None:
            scaler.load_state_dict(optimizer_state['scaler'])
        print(best_val_loss)
        del optimizer_state
    
     # Restore process-specific state
    process_path = os.path.join(resume_dir, f'process_state_rank_{rank_id}.pt')
    if os.path.exists(process_path):
        process_state = torch.load(process_path, map_location='cpu')
        current_offset = process_state['current_offset']
        
        # Restore sequence offsets
        if not hasattr(get_batch, 'seq_offsets'):
            get_batch.seq_offsets = {}
        get_batch.seq_offsets = process_state.get('seq_offsets', {})

        if 'global_val_offset' in process_state:
            get_batch.global_val_offset = process_state['global_val_offset']
        
        print(f"Rank {rank_id}: Resuming from offset {current_offset} with sequence tracking")
    else:
        print(f"Rank {rank_id}: No process state found, starting from beginning")

# Compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# Wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Learning rate decay scheduler (cosine with warmup)
def get_lr(it, schedule='cosine'):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1

    return min_lr + coeff * (learning_rate - min_lr)

# Logging
if wandb_log and master_process:
    wandb_run_id_file = os.path.join(out_dir, 'wandb_run_id.txt')
    if init_from == 'resume' and os.path.exists(wandb_run_id_file):
        with open(wandb_run_id_file, 'r') as f:
            wandb_run_id = f.read().strip()
    else:
        wandb_run_id = wandb.util.generate_id()
        os.makedirs(out_dir, exist_ok=True)
        with open(wandb_run_id_file, 'w') as f:
            f.write(wandb_run_id)
    wandb_config = {
        'model_args': model_args,
        'training_args': {
            'batch_size': batch_size,
            'block_size': block_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'max_iters': max_iters,
            'lr_decay_iters': lr_decay_iters,
            'eval_interval': eval_interval,
            'eval_iters': eval_iters,
            'log_interval': log_interval
        },
        'optimizer_args': {
            'optimizer_name': optimizer_name,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'beta1': beta1,
            'beta2': beta2,
            'grad_clip': grad_clip,
            'decay_lr': decay_lr,
            'warmup_iters': warmup_iters,
            'min_lr': min_lr,
            'schedule': schedule
        }
    }
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=wandb_config,
        id=wandb_run_id,
        resume="must" if init_from == 'resume' else None
    )
    if init_from == 'scratch':
        wandb.log({"param_count_m": param_count_m}, commit=False)

# Training loop
t0 = time.time()
raw_model = model.module if ddp else model # unwrap DDP container if needed
if init_from == 'scratch':
    X, Y = get_batch('train') # fetch the very first batch
elif init_from == 'resume':
    process_path = os.path.join(resume_dir, f'process_state_rank_{rank_id}.pt')
    if os.path.exists(process_path):
        process_state = torch.load(process_path)
        X, Y = process_state['x'], process_state['y']

while True:
    # Determine and set the learning rate for this iteration
    lr = get_lr(iter_num, schedule=schedule) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and iter_num > checkpoint_step + eval_interval // 10:
        if master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: val loss {losses['val']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "val/loss": losses['val'],
                    "lr": lr,
                    "data/current_offset": current_offset,
                }, step=iter_num)
                
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    print(f"saving model to {out_dir}")
                    # Save model
                    raw_model.save_pretrained(out_dir)

        # Periodically save extra checkpoints
        if iter_num % (eval_interval * 5) == 0:
            checkpoint_dir = os.path.join(out_dir, f'checkpoint-{iter_num}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            if master_process:
                # Save model
                raw_model.save_pretrained(checkpoint_dir)
                
                # Save optimizer state
                optimizer_state = {
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'clip_time': clip_time,
                    'best_val_loss': best_val_loss,
                    'tokens_trained': tokens_trained,
                    'scaler': scaler.state_dict() if dtype == 'float16' else None,
                }
                torch.save(optimizer_state, os.path.join(checkpoint_dir, 'optimizer.pt'))
            
            # Save complete data state
            process_state = {
                'current_offset': current_offset,
                'seq_offsets': getattr(get_batch, 'seq_offsets', {}),
            }
            
            if master_process:
                process_state['global_val_offset'] = getattr(get_batch, 'global_val_offset', 512)
            process_state['x'] = X
            process_state['y'] = Y
            
            process_path = os.path.join(checkpoint_dir, f'process_state_rank_{rank_id}.pt')
            torch.save(process_state, process_path)
            
            if master_process:
                print(f"Process states saved to {checkpoint_dir}/process_state_rank_*.pt")
            


    if iter_num == 0 and eval_only:
        break

    # Forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    accumulated_loss = 0.0
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # In DDP training we only need to sync gradients at the last micro step.
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
            accumulated_loss += loss.item()
            
        # Immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        
        # Backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        
    # Clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if total_norm.item() > grad_clip:
            clip_time += 1
            
    # Step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    
    # Flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    # Update total tokens trained
    tokens_trained += tokens_per_iter
    tokens_trained_B = tokens_trained / 1e9  # Convert to billions

    if iter_num % log_interval == 0 and master_process:
        lossf = accumulated_loss # loss as float. note: this is a CPU-GPU sync point
        tokens_per_sec = tokens_per_iter / dt
        tokens_per_sec_M = tokens_per_sec / 1_000_000
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, "
              f"tps (M) {tokens_per_sec_M:.2f}, tokens trained {tokens_trained_B:.2f}B, "
              f"file idx {assigned_indices[current_offset % len(assigned_indices)]}/{len(train_file_list)}")

        params = []
        for (name, p) in model.named_parameters():
            params.append(p)
        total_param_norm = 0
        for p in params:
            param_norm = p.data.norm(2)
            total_param_norm += param_norm.item() ** 2
        total_param_norm = total_param_norm ** 0.5
        
        momentum_norm = 0
        momentum_norm_sq = 0
        LL = len(optimizer.state_dict()['state'])
        for jj in range(LL):
            momentum_norm += (optimizer.state_dict()['state'][jj]['exp_avg'].detach().norm(2)) ** 2
            momentum_norm_sq += (optimizer.state_dict()['state'][jj]['exp_avg_sq'].detach().norm(2)) ** 2
        momentum_norm = torch.sqrt(momentum_norm).item()
        momentum_norm_sq = torch.sqrt(momentum_norm_sq).item()
        momentum_div = momentum_norm/(np.sqrt(momentum_norm_sq)+1e-8)
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "param_norm": total_param_norm,
                "momentum_norm" : momentum_norm,
                "momentum_norm_sq": momentum_norm_sq,
                "momentum_div": momentum_div,
                "train/clip_rate": clip_time / (iter_num + 1),
                "train/grad_norm": total_norm.item() if grad_clip != 0.0 else 0.0,
                "train/iter_time_ms": dt * 1000,
                "train/tokens_per_sec_M": tokens_per_sec_M,
                "train/tokens_trained_B": tokens_trained_B,
                "data/current_offset": current_offset,
                "data/current_file": assigned_indices[current_offset % len(assigned_indices)],
                "gpu/memory_allocated_MB": torch.cuda.memory_allocated() / (1024 * 1024),
                "gpu/max_memory_allocated_MB": torch.cuda.max_memory_allocated() / (1024 * 1024),
            }, step=iter_num)
            
    iter_num += 1

    # Termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()